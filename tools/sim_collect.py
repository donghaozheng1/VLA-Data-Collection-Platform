#!/usr/bin/env python3
"""模拟采集器：没有真机时，用仿真数据把**真实采集链路**跑通。

它替换掉的只有两件事：

- `read_joints()` → 用一条物理上说得通的 7 轴抓放轨迹（正弦缓动 + 关节限位）代替 RealMan SDK
- 相机帧 → 用 cv2 画出"目标块 + 夹爪 + 插槽"的合成画面代替 RealSense

**其余全部是真链路**：走 `POST /api/collect/session` → 逐帧上报 → 提交，服务端做归一化、
mp4 编码、episode 落盘、平台接入与时间同步体检 —— 和接真机时是同一段代码。

数据是仿真的，这一点在任务文本里写清楚（`[仿真]` 前缀），不会和真机数据混淆。

用法：

    # 4 条干净 episode（3 成功 1 失败），10Hz / 6s
    python tools/sim_collect.py --dataset sim-realman7-pickplace \
        --episodes 4 --seconds 6 --fps 10

    # 注入缺陷：第 2 条时间戳回退（质检会以「时间戳单调性」打回）
    python tools/sim_collect.py --dataset sim-broken --episodes 2 --defect backwards@1

    # 注入缺陷：时间跳变（质检以「时间戳步长」警告）
    python tools/sim_collect.py --dataset sim-gap --episodes 1 --defect gap@1
"""
from __future__ import annotations

import argparse
import base64
import math
import time

import cv2
import httpx
import numpy as np

IMAGE_KEY = "observation.image"
BATCH = 10
WIDTH, HEIGHT = 160, 120
JOINTS = [f"joint_{i}" for i in range(1, 8)]

# 7 轴关节限位（弧度，按常见 6+1 协作臂的保守范围）
LIMITS = [(-3.05, 3.05), (-2.0, 2.0), (-2.6, 2.6), (-3.05, 3.05),
          (-2.0, 2.0), (-2.6, 2.6), (0.0, 0.7)]


def clamp(value: float, index: int) -> float:
    low, high = LIMITS[index]
    return max(low, min(high, value))


def joints_at(phase: float, episode: int) -> list[float]:
    """一条抓放轨迹：接近 → 夹取 → 抬起 → 移动到插槽 → 放下。

    phase ∈ [0,1] 表示整条 episode 的进度；不同 episode 用不同的目标位置，
    所以轨迹彼此不同（不会都被重复检测当成同一条）。
    """
    offset = 0.06 * episode
    ease = math.sin(math.pi * phase)                      # 中途摆幅大，两端平缓
    reach = 0.22 + 0.05 * episode
    base = [
        0.30 * math.sin(math.pi * phase) + offset,        # 底座旋转
        -0.45 + 0.35 * ease,                              # 肩
        0.60 - 0.45 * ease,                               # 肘
        0.10 + 0.25 * math.sin(2 * math.pi * phase),      # 腕 1
        -0.20 + 0.30 * ease,                              # 腕 2
        0.15 * math.cos(math.pi * phase),                 # 腕 3
        max(0.02, 0.55 * (1 - abs(2 * phase - 1))),       # 夹爪：中段闭合
    ]
    base[0] += reach * (1 if episode % 2 == 0 else -1)
    return [round(clamp(v, i), 6) for i, v in enumerate(base)]


def render(phase: float, episode: int, success: bool) -> np.ndarray:
    """合成相机画面：底板上有一个目标块、右侧有一个插槽、上方有夹爪。"""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    frame[:, :] = (38, 40, 44)
    cv2.rectangle(frame, (0, HEIGHT - 28), (WIDTH, HEIGHT), (60, 62, 66), -1)   # 台面

    slot_x = 108
    cv2.rectangle(frame, (slot_x, HEIGHT - 52), (slot_x + 26, HEIGHT - 28), (90, 120, 90), 2)

    start_x, end_x = 30, slot_x + 2
    travel = min(1.0, phase / 0.75)
    block_x = int(start_x + (end_x - start_x) * travel)
    # 失败样本：最后阶段块掉落（位置停在半路，且带一点偏移）
    if not success and phase > 0.6:
        block_x = int(start_x + (end_x - start_x) * 0.55) + 6
    block_y = HEIGHT - 52
    cv2.rectangle(frame, (block_x, block_y), (block_x + 24, block_y + 24), (52, 90, 200), -1)

    grip_x = block_x + 12
    grip_y = int(block_y - 6 - 26 * math.sin(math.pi * phase))
    cv2.rectangle(frame, (grip_x - 12, grip_y - 8), (grip_x + 12, grip_y + 8), (200, 200, 210), -1)
    cv2.line(frame, (grip_x, grip_y + 8), (grip_x, grip_y + 18), (200, 200, 210), 2)

    cv2.putText(frame, f"sim ep{episode}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (180, 180, 180), 1)
    cv2.putText(frame, f"{'OK' if success else 'FAIL'} t={phase:.2f}", (4, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 200, 120) if success else (120, 120, 220), 1)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="模拟 RealMan+RealSense 采集（走真实会话接口）")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--failure-every", type=int, default=4,
                        help="每隔 N 条做成失败样本（0 = 全成功）")
    parser.add_argument("--defect", default="",
                        help="注入缺陷：backwards@索引 / gap@索引（索引从 0 起）")
    parser.add_argument("--task", default="抓取蓝色方块并放入右侧插槽")
    parser.add_argument("--no-image", action="store_true")
    args = parser.parse_args()

    defect_kind, defect_at = None, -1
    if args.defect:
        kind, _, index = args.defect.partition("@")
        defect_kind, defect_at = kind.strip(), int(index or 0)

    frames_per_episode = max(4, int(args.seconds * args.fps))
    task_text = f"[仿真] {args.task}"

    with httpx.Client(base_url=args.base, timeout=300) as client:
        health = client.get("/api/overview")
        if health.status_code != 200:
            raise SystemExit(f"平台不可达：{args.base} -> {health.status_code}")

        for episode in range(args.episodes):
            success = not (args.failure_every and (episode + 1) % args.failure_every == 0)
            session = client.post("/api/collect/session", json={
                "dataset_name": args.dataset, "task_text": task_text, "fps": args.fps,
                "robot_type": "sim-realman7dof", "state_dim": 7, "joint_names": JOINTS,
                "image_keys": [] if args.no_image else [IMAGE_KEY],
                "source": "simulator",
                "note": f"仿真第 {episode + 1} 条（{'成功' if success else '失败'}）",
            }).json()
            session_id = session["id"]

            batch: list[dict] = []
            t0 = time.time()
            for index in range(frames_per_episode):
                phase = index / max(1, frames_per_episode - 1)
                stamp = t0 + index / args.fps
                if defect_kind == "backwards" and episode == defect_at and index == frames_per_episode // 2:
                    stamp = t0 + (index - 3) / args.fps          # 时间戳回退
                elif defect_kind == "gap" and episode == defect_at and index == frames_per_episode // 2:
                    stamp = t0 + index / args.fps + 0.5          # 时间跳变
                frame = {"timestamp": stamp, "state": joints_at(phase, episode)}
                if not args.no_image:
                    ok, buf = cv2.imencode(".png", render(phase, episode, success))
                    if ok:
                        frame["images"] = {IMAGE_KEY: base64.b64encode(buf.tobytes()).decode()}
                batch.append(frame)
                if len(batch) >= BATCH:
                    client.post(f"/api/collect/session/{session_id}/frames", json={"frames": batch})
                    batch.clear()
            if batch:
                client.post(f"/api/collect/session/{session_id}/frames", json={"frames": batch})

            result = client.post(f"/api/collect/session/{session_id}/commit",
                                 json={"success": success,
                                       "note": f"仿真采集（{'成功' if success else '失败'}）"}).json()
            print(f"[ep{episode}] {'成功' if success else '失败'} 样本 -> episode "
                  f"{result.get('episode_index')}  {result.get('rows')} 行 / "
                  f"{result.get('video_frames')} 视频帧  步长偏差 {result.get('step_dev_max')}  "
                  f"平台数据集 #{result.get('platform_dataset_id')} v{result.get('platform_version')}")
    print(f"\n完成：数据集 {args.dataset}，共 {args.episodes} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
