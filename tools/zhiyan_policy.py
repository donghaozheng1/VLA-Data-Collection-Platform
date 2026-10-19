#!/usr/bin/env python3
"""zhiyan 模型驱动仿真（半闭环）：真机观测 -> 模型预测动作 -> 动作驱动 zhiyan 模型运动。

与 tools/libero_stream.py 同一套 I/O 约定，父进程 core/sim.py 直接可读：
    <run>/frames/f%06d.jpg + status.json + control.txt + rollout.mp4

    <yolov8n python> tools/zhiyan_policy.py --run <dir> --episode 0 --policy <ckpt>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ["MUJOCO_GL"] = "wgl"

ROOT = pathlib.Path(__file__).resolve().parent.parent
HF = ROOT / "storage" / "hf_cache"
os.environ.update({"HF_HOME": str(HF), "HF_HUB_CACHE": str(HF / "hub"),
                   "TORCH_HOME": str(HF / "torch"), "XDG_CACHE_HOME": str(HF),
                   "HF_HUB_DISABLE_SYMLINKS": "1"})
for extra in reversed((ROOT / ".deps_libero", ROOT / ".deps_lerobot",
                       pathlib.Path(r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\src"))):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

XML = ROOT / "storage" / "zhiyan_urdf" / "zhiyan_right_arm_grip.xml"
DS = ROOT / "storage" / "datasets" / "zhiyan_pick_place"


def write_status(run_dir: pathlib.Path, **kw) -> None:
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")
    tmp.replace(run_dir / "status.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--grip-scale", type=float, default=0.06, help="手部通道 -> 行程(米)")
    ap.add_argument("--grip-invert", action="store_true", help="手部 1=张开（默认 1=握紧）")
    args = ap.parse_args()

    import cv2
    import mujoco
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    run_dir = pathlib.Path(args.run).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    control = run_dir / "control.txt"
    control.write_text("run", encoding="utf-8")
    started = time.time()
    write_status(run_dir, state="starting", step=0, steps=0, seconds=0.0)
    video_fps = None

    try:
        # ---- 模型 ----
        import lerobot.policies.act  # noqa: F401  触发注册
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        cfg = PreTrainedConfig.from_pretrained(args.policy)
        policy = get_policy_class(cfg.type).from_pretrained(args.policy)
        policy.eval()
        device = next(policy.parameters()).device
        policy.reset()

        # ---- 仿真模型 ----
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.size, width=args.size)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.azimuth, cam.elevation, cam.distance = 118.0, -8.0, model.stat.extent * 0.75
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_Link")

        # ---- 数据（真机观测）----
        d = pq.read_table(DS / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet").to_pydict()
        states = np.asarray(d["observation.state"], dtype=np.float32)
        experts = np.asarray(d["action"], dtype=np.float32)
        total = args.steps or len(states)
        total = min(total, len(states))
        caps = {}
        for key in ("observation.images.head", "observation.images.hand_right"):
            caps[key] = cv2.VideoCapture(str(DS / "videos" / "chunk-000" / key /
                                             f"episode_{args.episode:06d}.mp4"))
        task = f"zhiyan 模型驱动（episode {args.episode}，{cfg.type}）"
        write_status(run_dir, state="running", step=0, steps=total, seconds=0.0, task=task)

        MEAN = np.array([0.485, 0.456, 0.406], np.float32)
        STD = np.array([0.229, 0.224, 0.225], np.float32)

        def img_tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device)

        preds = []
        for i in range(total):
            mode = control.read_text(encoding="utf-8").strip() if control.is_file() else "run"
            if mode == "stop":
                break
            while mode == "pause":
                time.sleep(0.15)
                mode = control.read_text(encoding="utf-8").strip() if control.is_file() else "run"
                if mode == "stop":
                    break

            batch = {"observation.state": torch.from_numpy(states[i]).unsqueeze(0).to(device)}
            ok_all = True
            for key, cap in caps.items():
                ok, frame = cap.read()
                if not ok:
                    ok_all = False
                    break
                batch[key] = img_tensor(frame)
            if not ok_all:
                break

            with torch.no_grad():
                action = policy.select_action(batch).cpu().numpy().ravel()
            preds.append(action)

            # 动作 -> 仿真：手臂用前 7 维，手部用手部通道
            data.qpos[:7] = np.radians(action[:7])
            grip_src = action[7:14]
            grip_val = float(np.clip(grip_src.mean(), 0.0, 1.0))
            if args.grip_invert:
                grip_val = 1.0 - grip_val
            data.qpos[7] = grip_val * args.grip_scale
            data.qpos[8] = -grip_val * args.grip_scale
            mujoco.mj_forward(model, data)
            if hand_id >= 0:
                cam.lookat[:] = 0.6 * cam.lookat + 0.4 * data.xpos[hand_id]
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                (frames_dir / f"f{i:06d}.jpg").write_bytes(buf.tobytes())
            elapsed = time.time() - started
            write_status(run_dir, state="running", step=i + 1, steps=total, seconds=elapsed,
                         task=task, sps=round((i + 1) / max(elapsed, 0.001), 2))

        for cap in caps.values():
            cap.release()
        preds_arr = np.asarray(preds, dtype=np.float32) if preds else np.zeros((0, 14), np.float32)
        mae_arm = float(np.abs(preds_arr[:, :7] - experts[:len(preds_arr), :7]).mean()) if len(preds_arr) else -1
        state, error = "finished", None
    except Exception as exc:  # noqa: BLE001
        state, error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"

    try:
        jpegs = sorted(frames_dir.glob("f*.jpg"))
        n = len(jpegs)
        if n:
            video_fps = 20.0                    # 回放/仿真都按 20fps（与数据一致）
            first = cv2.imread(str(jpegs[0]))
            if first is not None:
                h, w = first.shape[:2]
                vw = cv2.VideoWriter(str(run_dir / "rollout.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))
                for j in jpegs:
                    im = cv2.imread(str(j))
                    if im is not None:
                        vw.write(im)
                vw.release()
    except Exception as exc:  # noqa: BLE001
        print(f"[zhiyan_policy] mp4 编码失败: {exc}", flush=True)

    total_elapsed = time.time() - started
    write_status(run_dir, state=state, step=n if "n" in dir() else 0, steps=0,
                 seconds=total_elapsed, error=error,
                 sps=round((n if "n" in dir() else 0) / max(total_elapsed, 0.001), 2),
                 video_fps=video_fps, mae_arm=(mae_arm if state == "finished" else None),
                 video=str((run_dir / "rollout.mp4").relative_to(ROOT)) if (run_dir / "rollout.mp4").is_file() else None)
    print(f"[zhiyan_policy] {state} {total_elapsed:.1f}s 臂部动作 MAE={mae_arm if state == 'finished' else '-'} {error or ''}",
          flush=True)
    return 0 if state == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
