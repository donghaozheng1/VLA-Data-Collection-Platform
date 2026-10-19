#!/usr/bin/env python3
"""在**独立进程**里跑 LIBERO 仿真，把帧与状态写成文件（避免在服务进程里 import torch）。

用法（由 core/sim.py 拉起，也可以手工跑）：
    <yolov8n python> tools/libero_stream.py --run storage/sim_runs/xxx --steps 600 --size 256

产出（都写在 --run 目录里）：
    latest.jpg     最新一帧（原子替换：先写 .tmp 再 rename，读侧永远拿到完整文件）
    status.json    状态：state/step/steps/reward/task/error/seconds
    rollout.mp4    录像
    control.txt    父进程写 "pause" / "run" / "stop" 控制它
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEROBOT_SRC = pathlib.Path(r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\src")
for extra in reversed((ROOT / ".deps_libero", ROOT / ".deps_lerobot",
                       ROOT / "storage" / "libero" / "LIBERO-master", LEROBOT_SRC)):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
_HF = ROOT / "storage" / "hf_cache"
os.environ.update(HF_HOME=str(_HF), HF_HUB_CACHE=str(_HF / "hub"),
                  TORCH_HOME=str(_HF / "torch"), XDG_CACHE_HOME=str(_HF),
                  HF_HUB_DISABLE_SYMLINKS="1")
os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "storage" / "libero" / "config"))

TASK_KEY = "alphabet soup and the tomato sauce"


def policy_action(policy, obs, device):
    """把仿真观测映射成模型输入，返回 7 维动作（直接可喂 env.step）。

    8 维 observation.state 的构成是**用真实数据核对出来的**：
        eef_pos(3) + eef 轴角(3) + 夹爪 qpos(2)
    （不是四元数；轴角用 robosuite 自己的 quat2axisangle 保证约定一致）
    """
    import numpy as np
    import torch
    from robosuite.utils.transform_utils import quat2axisangle

    def chw(rgb):
        arr = np.ascontiguousarray(np.asarray(rgb))
        t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
        return t.unsqueeze(0).to(device)

    state = np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(quat2axisangle(np.asarray(obs["robot0_eef_quat"])), dtype=np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ])
    batch = {
        "observation.images.image": chw(obs["agentview_image"]),
        "observation.images.wrist_image": chw(obs["robot0_eye_in_hand_image"]),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.state.joint": torch.from_numpy(
            np.asarray(obs["robot0_joint_pos"], dtype=np.float32)).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        action = policy.select_action(batch)
    return action.cpu().numpy().ravel()


def write_status(run_dir: pathlib.Path, **kw) -> None:
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")
    tmp.replace(run_dir / "status.json")


def scripted_action(step: int, total: int):
    import numpy as np

    phase = step / max(total, 1)
    if phase < 0.3:
        return np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
    if phase < 0.55:
        return np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0])
    if phase < 0.8:
        return np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0])
    return np.array([0.0, 0.35, 0.2, 0.0, 0.0, 0.0, -1.0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()

    import cv2
    import numpy as np

    run_dir = pathlib.Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    control = run_dir / "control.txt"
    control.write_text("run", encoding="utf-8")
    started = time.time()
    write_status(run_dir, state="starting", step=0, steps=args.steps, reward=0.0, seconds=0.0)

    env = None
    last_step = 0
    video_fps = None
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
        index = next(i for i, t in enumerate(langs) if TASK_KEY in t)
        task = langs[index]
        write_status(run_dir, state="running", step=0, steps=args.steps, reward=0.0,
                     task=task, seconds=0.0)
        bddl = suite.get_task_bddl_file_path(index)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size,
                                 camera_widths=args.size, camera_depths=False)
        env.seed(args.seed)
        obs = env.reset()

        # ---- 加载策略（可选）----
        policy = None
        policy_device = None
        if args.policy:
            import torch
            import lerobot.policies.act  # 触发 @PreTrainedConfig.register_subclass("act")
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class

            cfg = PreTrainedConfig.from_pretrained(args.policy)
            policy = get_policy_class(cfg.type).from_pretrained(args.policy)
            policy.eval()
            policy_device = next(policy.parameters()).device
            policy.reset()
            write_status(run_dir, state="running", step=0, steps=args.steps, reward=0.0,
                         task=task, seconds=0.0, policy=args.policy)
            print(f"[libero_stream] 策略已加载 {type(policy).__name__} device={policy_device}",
                  flush=True)

        interval = 1.0 / max(args.fps, 1)
        for step in range(args.steps):
            mode = (control.read_text(encoding="utf-8").strip() if control.is_file() else "run")
            if mode == "stop":
                break
            while mode == "pause":
                time.sleep(0.2)
                mode = (control.read_text(encoding="utf-8").strip() if control.is_file() else "run")
                if mode == "stop":
                    break
            if policy is not None:
                action = policy_action(policy, obs, policy_device)
            else:
                action = scripted_action(step, args.steps)
            obs, reward, done, _info = env.step(np.asarray(action, dtype=float))
            frame = np.asarray(obs["agentview_image"])[::-1].copy()
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                # 每帧写一个编号文件（不改名，避开 Windows 文件锁），收尾按实测帧率编码
                (frames_dir / f"f{step:06d}.jpg").write_bytes(buf.tobytes())
            last_step = step + 1
            write_status(run_dir, state="running", step=last_step, steps=args.steps,
                         reward=float(reward), task=task, seconds=time.time() - started)
            if done:
                break
            time.sleep(interval)
        state = "finished"
        error = None
    except Exception as exc:  # noqa: BLE001
        state = "failed"
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass
        # ★ 收尾：按实测帧率编码 mp4，播放速度 = 真实速度
        try:
            jpegs = sorted(frames_dir.glob("f*.jpg"))
            if jpegs and last_step:
                total = max(time.time() - started, 0.001)
                fps_measured = max(1.0, last_step / total)
                first = cv2.imread(str(jpegs[0]))
                if first is not None:
                    h, w = first.shape[:2]
                    vw = cv2.VideoWriter(str(run_dir / "rollout.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         round(fps_measured, 2), (w, h))
                    for jpg in jpegs:
                        img = cv2.imread(str(jpg))
                        if img is not None:
                            vw.write(img)
                    vw.release()
                    video_fps = round(fps_measured, 2)
        except Exception as exc:  # noqa: BLE001
            print(f"[libero_stream] 编码 mp4 失败: {type(exc).__name__}: {exc}", flush=True)
    total_elapsed = time.time() - started
    write_status(run_dir, state=state, step=last_step, steps=args.steps, reward=0.0,
                 seconds=total_elapsed, error=error,
                 sps=round(last_step / max(total_elapsed, 0.001), 2),
                 video_fps=video_fps,
                 video=str((run_dir / "rollout.mp4").relative_to(ROOT)) if (run_dir / "rollout.mp4").is_file() else None,
                 policy=args.policy)
    print(f"[libero_stream] {state} {time.time() - started:.1f}s {error or ''}", flush=True)
    return 0 if state == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
