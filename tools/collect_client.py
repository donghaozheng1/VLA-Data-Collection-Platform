#!/usr/bin/env python3
"""采集客户端：把**真机**（RealSense 相机 + RealMan 机械臂）或**回放**数据流式上报给平台。

两条数据源：

1. `realtime` —— 真机采集。需要 `pyrealsense2` 与 RealMan SDK（在本机 `envs\\yolov8n` 那类
   已装好驱动的环境里运行）。事件循环：读一帧彩色图 + 读一次关节角 → 打主机时间戳 → 批量上报。

        python tools/collect_client.py realtime --base http://127.0.0.1:8000 \
            --dataset realman7-grasp --task "抓取红色连接器并插入左侧插槽" --seconds 6

2. `replay` —— 回放已有 LeRobot 数据集里的某条 episode（**不需要硬件**，用来验证链路）。
   会真的解出视频帧、按原 fps 逐帧上报，服务端编码成新的 mp4。

        python tools/collect_client.py replay --base http://127.0.0.1:8000 \
            --dataset collect-replay --task "抓取红色连接器并插入左侧插槽" --success \
            --from "C:\\Users\\ll\\Desktop\\VLA\\lerobot\\.hf_cache\\lerobot\\lerobot\\pusht" \
            --episode 0

时间同步：客户端只负责用主机单调时钟打时间戳（Unix 秒），不做任何对齐；
服务端在提交时校验单调性与步长偏差，抖动会在 `/qc` 里被 `时间戳步长` 规则报出。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

IMAGE_KEY = "observation.image"
BATCH = 10


def post(client: httpx.Client, path: str, payload: dict) -> dict:
    resp = client.post(path, json=payload)
    if resp.status_code >= 400:
        raise SystemExit(f"{path} -> {resp.status_code} {resp.text[:300]}")
    return resp.json()


def start_session(client: httpx.Client, args, state_dim: int, joint_names: list[str]) -> int:
    session = post(client, "/api/collect/session", {
        "dataset_name": args.dataset,
        "task_text": args.task,
        "fps": args.fps,
        "robot_type": args.robot_type,
        "state_dim": state_dim,
        "joint_names": joint_names,
        "image_keys": [] if args.no_image else [IMAGE_KEY],
        "source": args.source,
        "note": args.note,
    })
    print(f"[session] #{session['id']} dataset={args.dataset} fps={args.fps} "
          f"state_dim={state_dim} 图像={'关' if args.no_image else '开'}")
    return session["id"]


def flush(client: httpx.Client, session_id: int, batch: list[dict]) -> None:
    if not batch:
        return
    result = post(client, f"/api/collect/session/{session_id}/frames", {"frames": batch})
    print(f"  上报 {result['appended']} 帧（累计 {result['frames']}）")
    batch.clear()


def commit(client: httpx.Client, session_id: int, success: bool | None, note: str | None) -> None:
    result = post(client, f"/api/collect/session/{session_id}/commit",
                  {"success": success, "note": note})
    print("\n[commit]")
    print(f"  episode_index = {result['episode_index']}  rows = {result['rows']}  "
          f"duration = {result['duration_s']}s  video_frames = {result['video_frames']}")
    print(f"  步长偏差 = {result['step_dev_max']}  时间戳单调 = {result['timestamps_monotonic']}")
    print(f"  数据集路径 = {result['dataset_path']}")
    print(f"  已接入平台: dataset #{result.get('platform_dataset_id')} "
          f"v{result.get('platform_version')} hash={str(result.get('platform_content_hash'))[:12]}…"
          f" (ingest {result.get('ingest_status')})")
    print(f"  下一步：/qc 对该数据集执行质检；若步长偏差 > 0.2 会命中「时间戳步长」规则")


# --------------------------------------------------------------------------- 采集源

def run_replay(client: httpx.Client, args) -> None:
    import cv2
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core import lerobot  # 复用平台的读取器，保证字段口径一致

    src = Path(args.from_path)
    meta = lerobot.load_meta(src)
    index = args.episode
    paths = lerobot.episode_paths(meta["info"], src, index)
    table = pq.read_table(paths["parquet"])
    data = table.to_pydict()
    states = [list(row) for row in data["observation.state"]]
    timestamps = [float(t) for t in data["timestamp"]]
    video = next(iter(paths["videos"].values()), None)
    cap = cv2.VideoCapture(str(video)) if (video and not args.no_image) else None

    state_dim = len(states[0])
    session_id = start_session(client, args, state_dim,
                              [f"motor_{i}" for i in range(state_dim)])
    batch: list[dict] = []
    sent = 0
    for i, (state, ts) in enumerate(zip(states, timestamps)):
        frame = {"timestamp": ts, "state": state}
        if cap is not None:
            ok, image = cap.read()
            if not ok:
                print(f"[warn] 视频在第 {i} 帧结束，提前停止")
                break
            ok, buf = cv2.imencode(".png", image)
            frame["images"] = {IMAGE_KEY: base64.b64encode(buf.tobytes()).decode()}
        batch.append(frame)
        sent += 1
        if len(batch) >= BATCH:
            flush(client, session_id, batch)
        if args.delay:
            time.sleep(args.delay)
    flush(client, session_id, batch)
    if cap is not None:
        cap.release()
    print(f"[replay] 共上报 {sent} 帧（源：{src.name} episode {index}）")
    commit(client, session_id, True if args.success else (False if args.failure else None), args.note)


def run_realtime(client: httpx.Client, args) -> None:
    try:
        import pyrealsense2 as rs
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"真机采集需要 pyrealsense2（当前环境缺失：{exc}）。\n"
            f"本机 base 环境已装好 pyrealsense2 2.58.2，用 base 的 python 跑即可；"
            f"envs\\yolov8n 里没有。") from exc

    import cv2
    import numpy as np

    # SDK 在 ≠ 相机在：这里真的枚举一次设备，避免把“没插相机”误报成“SDK 缺失”
    devices = []
    try:
        devices = list(rs.context().query_devices())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 枚举 RealSense 设备失败：{type(exc).__name__}: {exc}")
    if not devices:
        raise SystemExit(
            "RealSense SDK 可用，但**没有检测到相机设备**。\n"
            "  请插上相机（建议 USB3 口）后重试；想先验证数据链路可用回放模式：\n"
            "  python tools/collect_client.py replay --dataset <名> --task \"<指令>\" "
            "--from <数据集目录> --episode 0")
    print(f"[realtime] 检测到 {len(devices)} 个 RealSense 设备："
          + "、".join(str(d.get_info(rs.camera_info.name)) for d in devices))

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core import session as session_mod  # 仅用于 joint 名默认值

    state_dim = args.state_dim or 7
    session_id = start_session(client, args, state_dim, session_mod.DEFAULT_JOINT_NAMES[:state_dim])

    pipeline = rs.pipeline()
    config = rs.config()
    width, height = (1280, 720) if args.d405 else (640, 480)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)
    print(f"[realtime] RealSense {'D405' if args.d405 else 'D435'} {width}x{height}@30 已启动；"
          f"采集 {args.seconds}s @ {args.fps}Hz")

    # 机械臂关节读数：按你们现有脚本（dhz\\realman_ctrl.py / inference_api.py）的方式接入；
    # 这里留出一个明确的钩子，避免猜测具体 SDK 调用。
    def read_joints() -> list[float]:
        raise SystemExit(
            "尚未接入 RealMan 关节读取。请在 read_joints() 里接上你们的 SDK 调用"
            "（参考 dhz\\realman_ctrl.py），或先用 --no-robot 只采图像。")

    period = 1.0 / args.fps
    batch: list[dict] = []
    started = time.monotonic()
    frames = 0
    try:
        while time.monotonic() - started < args.seconds:
            tick = time.monotonic()
            frameset = pipeline.wait_for_frames()
            color = frameset.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data())
            state = ([0.0] * state_dim) if args.no_robot else read_joints()
            ok, buf = cv2.imencode(".png", image)
            batch.append({
                "timestamp": time.time(),
                "state": state,
                "images": {IMAGE_KEY: base64.b64encode(buf.tobytes()).decode()},
            })
            frames += 1
            if len(batch) >= BATCH:
                flush(client, session_id, batch)
            sleep = period - (time.monotonic() - tick)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        pipeline.stop()
    flush(client, session_id, batch)
    print(f"[realtime] 共采集 {frames} 帧")
    commit(client, session_id, True if args.success else (False if args.failure else None), args.note)


def main() -> int:
    parser = argparse.ArgumentParser(description="VLA 数采平台采集客户端")
    parser.add_argument("source", choices=["realtime", "replay"])
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", required=True, help="目标数据集名")
    parser.add_argument("--task", required=True, help="语言指令（会写进 tasks.jsonl）")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--robot-type", default="realman-7dof")
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--no-image", action="store_true", help="不采图像（纯轨迹）")
    parser.add_argument("--success", action="store_true", help="标记为成功样本")
    parser.add_argument("--failure", action="store_true", help="标记为失败样本")
    parser.add_argument("--note", default=None)
    # replay
    parser.add_argument("--from", dest="from_path", help="回放源数据集目录")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0, help="回放时每帧额外延时（秒）")
    # realtime
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--d405", action="store_true", help="用 D405（1280x720 精定位）")
    parser.add_argument("--no-robot", action="store_true", help="只采图像，关节填 0（调试用）")
    args = parser.parse_args()

    if args.source == "replay" and not args.from_path:
        raise SystemExit("replay 模式需要 --from <数据集目录>")

    with httpx.Client(base_url=args.base, timeout=120) as client:
        health = client.get("/api/overview")
        if health.status_code != 200:
            raise SystemExit(f"平台不可达：{args.base} -> {health.status_code}")
        print(f"[base] {args.base} overview={json.dumps(health.json(), ensure_ascii=False)[:120]}")
        if args.source == "replay":
            run_replay(client, args)
        else:
            run_realtime(client, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
