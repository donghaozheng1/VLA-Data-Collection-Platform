#!/usr/bin/env python3
"""zhiyan 轨迹回放：把数据集里的真机关节角逐帧喂给 MuJoCo 模型并渲染。

I/O 约定与 tools/libero_stream.py 一致（父进程 core/sim.py 直接可读）：
    <run>/frames/f000000.jpg   每帧一张（读侧取编号最大的）
    <run>/status.json          状态（state/step/steps/seconds/sps/error）
    <run>/control.txt          父进程写 run/pause/stop
    <run>/rollout.mp4          收尾按实测帧率编码

    <yolov8n python> tools/zhiyan_replay.py --run storage/sim_runs/x --episode 0 --steps 399
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ["MUJOCO_GL"] = "wgl"          # Windows 离屏渲染后端

ROOT = pathlib.Path(__file__).resolve().parent.parent
for extra in reversed((ROOT / ".deps_libero", ROOT / ".deps_lerobot")):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

XML = ROOT / "storage" / "zhiyan_urdf" / "zhiyan_right_arm.xml"
DS = ROOT / "storage" / "datasets" / "zhiyan_pick_place"


def write_status(run_dir: pathlib.Path, **kw) -> None:
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")
    tmp.replace(run_dir / "status.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="0=整条 episode")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    import cv2
    import mujoco
    import numpy as np
    import pyarrow.parquet as pq

    run_dir = pathlib.Path(args.run).resolve()      # 必须是绝对路径，后面 relative_to(ROOT) 才不会崩
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    control = run_dir / "control.txt"
    control.write_text("run", encoding="utf-8")
    started = time.time()
    write_status(run_dir, state="starting", step=0, steps=0, seconds=0.0)
    video_fps = None

    try:
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.size, width=args.size)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = model.stat.center
        cam.azimuth, cam.elevation = 118.0, -8.0
        cam.distance = model.stat.extent * 0.75
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_Link")

        pq_path = DS / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
        d = pq.read_table(pq_path).to_pydict()
        states = np.asarray(d["observation.state"], dtype=float)
        total = args.steps or len(states)
        total = min(total, len(states))
        task = "pick_and_place（轨迹回放）"
        write_status(run_dir, state="running", step=0, steps=total, seconds=0.0, task=task)

        for i in range(total):
            mode = control.read_text(encoding="utf-8").strip() if control.is_file() else "run"
            if mode == "stop":
                break
            while mode == "pause":
                time.sleep(0.15)
                mode = control.read_text(encoding="utf-8").strip() if control.is_file() else "run"
                if mode == "stop":
                    break
            data.qpos[:] = np.radians(states[i, :7])       # 数据 7 维关节角 -> qpos
            mujoco.mj_forward(model, data)
            if hand_id >= 0:                               # 相机跟随末端
                cam.lookat[:] = 0.6 * cam.lookat + 0.4 * data.xpos[hand_id]
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                (frames_dir / f"f{i:06d}.jpg").write_bytes(buf.tobytes())
            elapsed = time.time() - started
            write_status(run_dir, state="running", step=i + 1, steps=total, seconds=elapsed,
                         task=task, sps=round((i + 1) / max(elapsed, 0.001), 2))
        state = "finished"
        error = None
    except Exception as exc:  # noqa: BLE001
        state, error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"

    # 收尾：按实测帧率编码 mp4
    try:
        jpegs = sorted(frames_dir.glob("f*.jpg"))
        n = len(jpegs)
        if n:
            video_fps = max(1.0, n / max(time.time() - started, 0.001))
            first = cv2.imread(str(jpegs[0]))
            if first is not None:
                h, w = first.shape[:2]
                vw = cv2.VideoWriter(str(run_dir / "rollout.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), round(video_fps, 2), (w, h))
                for j in jpegs:
                    img = cv2.imread(str(j))
                    if img is not None:
                        vw.write(img)
                vw.release()
    except Exception as exc:  # noqa: BLE001
        print(f"[zhiyan_replay] 编码 mp4 失败: {exc}", flush=True)

    total_elapsed = time.time() - started
    write_status(run_dir, state=state, step=n if "n" in dir() else 0, steps=0,
                 seconds=total_elapsed, error=error, sps=round((n if "n" in dir() else 0) / max(total_elapsed, 0.001), 2),
                 video_fps=video_fps,
                 video=str((run_dir / "rollout.mp4").relative_to(ROOT)) if (run_dir / "rollout.mp4").is_file() else None)
    print(f"[zhiyan_replay] {state} {total_elapsed:.1f}s {error or ''}", flush=True)
    return 0 if state == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
