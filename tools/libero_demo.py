#!/usr/bin/env python3
"""LIBERO 仿真演示/查看器。

用法（在**你自己的终端**里跑，这样才有桌面窗口）：
    # 实时窗口：能直接看到机械臂动（可拖动视角，Esc 退出）
    python tools/libero_demo.py --live --steps 400

    # 离屏录制：无窗口，产出 mp4 + 拼图（我在后台用这个）
    python tools/libero_demo.py --steps 400 --out storage/libero

说明：
  - 需要把 .deps_libero（robosuite 1.4.1 + mujoco 2.3.5）与 .deps_lerobot 一起放进 PYTHONPATH；
  - Windows 上 MUJOCO_GL 必须是 wgl（robosuite 那段 egl 覆盖已在本项目隔离目录里打过补丁）；
  - 不带 --policy 时用内置的脚本动作（证明仿真会动）；带 --policy 时加载 ACT checkpoint。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
# 注意顺序：reversed + insert(0) 之后，.deps_libero 必须排在最前
# （里面有 LIBERO 要求的 robosuite 1.4.1 + mujoco 2.3.5，否则会命中 .deps_lerobot 里的 1.5.2）
for extra in reversed((ROOT / ".deps_libero", ROOT / ".deps_lerobot",
                       ROOT / "storage" / "libero" / "LIBERO-master",
                       pathlib.Path(r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\src"))):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
print("[paths] " + " | ".join(str(p) for p in sys.path[:4]))

os.environ.setdefault("MUJOCO_GL", "wgl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "storage" / "libero" / "config"))
TASK_KEY = "alphabet soup and the tomato sauce"


def make_env(render: bool, size: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
    index = next(i for i, t in enumerate(langs) if TASK_KEY in t)
    bddl = suite.get_task_bddl_file_path(index)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size,
                             camera_depths=False)
    env.seed(0)
    return env, langs[index]


def scripted_action(step: int, total: int):
    """没接模型时的脚本动作：伸向桌面、合拢、抬起 —— 用来证明仿真真的在动。"""
    import numpy as np

    phase = step / max(total, 1)
    if phase < 0.35:
        return np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
    if phase < 0.6:
        return np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0])
    if phase < 0.85:
        return np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    return np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0, -1.0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="打开实时窗口（需要桌面）")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", default="storage/libero")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    env, task = make_env(args.live, args.size)
    print(f"任务: {task}")
    obs = env.reset()
    if args.live:
        env.render()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    frames_for_montage = []
    if not args.live:
        import cv2
        import numpy as np

        montage_every = max(1, args.steps // 8)

    import numpy as np

    started = time.time()
    for step in range(args.steps):
        action = scripted_action(step, args.steps)
        obs, reward, done, info = env.step(np.asarray(action, dtype=float))
        if args.live:
            env.render()
        else:
            import cv2

            frame = obs["agentview_image"][::-1]              # BGR
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "demo_rollout.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            if step % montage_every == 0 and len(frames_for_montage) < 8:
                thumb = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                cv2.rectangle(thumb, (0, 0), (256, 24), (255, 255, 255), -1)
                cv2.putText(thumb, f"step {step}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                frames_for_montage.append(thumb)
        if done:
            print(f"  episode 在第 {step} 步结束")
            break
    elapsed = time.time() - started

    if writer is not None:
        import cv2
        import numpy as np

        writer.release()
        while len(frames_for_montage) < 8:
            frames_for_montage.append(np.zeros((256, 256, 3), np.uint8))
        montage = np.vstack([np.hstack(frames_for_montage[0:4]), np.hstack(frames_for_montage[4:8])])
        montage_path = out_dir / "demo_preview.png"
        cv2.imwrite(str(montage_path), montage)
        print(f"  视频: {out_dir / 'demo_rollout.mp4'}")
        print(f"  拼图: {montage_path}")
    env.close()
    print(f"完成：{args.steps} 步 / {elapsed:.1f}s（{args.steps / max(elapsed, 0.001):.1f} 步/秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
