"""实时采集会话：把客户端实时上报的帧，写成**真实的 LeRobot episode**。

时间同步契约
------------
- 客户端用**主机单调时钟**给每帧打 Unix 秒时间戳；服务端在提交时按 `t - t0` 归一成
  episode 内的相对时间（LeRobot 约定：每条 episode 从 0 开始）。
- 提交时服务端**校验**（而不是强行改写）时间戳单调性与步长偏差，把最大偏差写进结果；
  真有抖动/丢帧的数据集会在随后的 `/qc` 里按 `时间戳单调性` / `时间戳步长` 规则被打回，
  再用 `/preprocess` 的修正类预处理（`dedup_ts` / `align_ts` / `resample`）处理。

帧与图像的落盘
--------------
- `storage/sessions/<id>/frames.jsonl`：逐帧 state/action/时间戳
- `storage/sessions/<id>/images/<key>/<frame>.png`：逐帧图像（可选）
- 提交时用 cv2 把 PNG 序列编码成 mp4（LeRobot 的 video feature），并把 PNG 保留备查
"""
from __future__ import annotations

import base64
import json
import math
import shutil
import statistics
from pathlib import Path

from . import db, lerobot

SESSIONS_DIR = db.STORAGE / "sessions"
DEFAULT_IMAGE_KEY = "observation.image"
DEFAULT_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:  # noqa: BLE001
    HAS_CV2 = False


# --------------------------------------------------------------------------- 会话生命周期

def create_session(*, dataset_name: str, task_text: str, fps: float = 10.0,
                   robot_type: str = "realman-7dof", state_dim: int = 7,
                   joint_names: list[str] | None = None, image_keys: list[str] | None = None,
                   source: str = "client", note: str | None = None) -> dict:
    joint_names = joint_names or DEFAULT_JOINT_NAMES[:state_dim]
    image_keys = image_keys or []
    session_id = db.execute(
        "INSERT INTO collect_sessions (dataset_name, robot_type, fps, task_text, state_dim,"
        " joint_names_json, image_keys_json, status, source, note, created_at)"
        " VALUES (?,?,?,?,?,?,?, 'open', ?,?,?)",
        (dataset_name, robot_type, float(fps), task_text, int(state_dim),
         json.dumps(joint_names, ensure_ascii=False), json.dumps(image_keys, ensure_ascii=False),
         source, note, db.now()))
    path = SESSIONS_DIR / str(session_id)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "frames.jsonl").touch()
    db.execute("UPDATE collect_sessions SET path=? WHERE id=?", (str(path), session_id))
    return status(session_id)


def status(session_id: int) -> dict:
    row = db.query_one("SELECT * FROM collect_sessions WHERE id=?", (session_id,))
    if not row:
        raise ValueError(f"会话不存在：{session_id}")
    row["joint_names"] = json.loads(row.get("joint_names_json") or "[]")
    row["image_keys"] = json.loads(row.get("image_keys_json") or "[]")
    if row.get("result_json"):
        row["result"] = json.loads(row["result_json"])
    return row


def sessions(limit: int = 20, status_filter: str | None = None) -> list[dict]:
    if status_filter:
        rows = db.query("SELECT * FROM collect_sessions WHERE status=? ORDER BY id DESC LIMIT ?",
                        (status_filter, limit))
    else:
        rows = db.query("SELECT * FROM collect_sessions ORDER BY id DESC LIMIT ?", (limit,))
    for row in rows:
        row["joint_names"] = json.loads(row.get("joint_names_json") or "[]")
        row["image_keys"] = json.loads(row.get("image_keys_json") or "[]")
    return rows


def append_frames(session_id: int, frames: list[dict]) -> dict:
    """追加一批帧。每帧：{timestamp, state:[...], action:[...]?, images:{key: base64 png}}。"""
    session = status(session_id)
    if session["status"] != "open":
        raise ValueError(f"会话状态为 {session['status']}，不能再追加帧")
    frames_file = Path(session["path"]) / "frames.jsonl"
    written = 0
    with frames_file.open("a", encoding="utf-8") as fh:
        for frame in frames:
            state = frame.get("state")
            if not state:
                raise ValueError("每帧必须有 state")
            if len(state) != session["state_dim"]:
                raise ValueError(f"state 维度不符：期望 {session['state_dim']}，收到 {len(state)}")
            record = {"t": float(frame["timestamp"]), "state": [float(x) for x in state]}
            if frame.get("action") is not None:
                record["action"] = [float(x) for x in frame["action"]]
            images = frame.get("images") or {}
            for key, payload in images.items():
                index = session["frames"] + written
                target = Path(session["path"]) / "images" / key
                target.mkdir(parents=True, exist_ok=True)
                data = base64.b64decode(payload.split(",")[-1])
                (target / f"{index:06d}.png").write_bytes(data)
                record.setdefault("images", {})[key] = f"{key}/{index:06d}.png"
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    db.execute("UPDATE collect_sessions SET frames=frames+? WHERE id=?", (written, session_id))
    return {"session_id": session_id, "appended": written,
            "frames": db.scalar("SELECT frames AS c FROM collect_sessions WHERE id=?", (session_id,))}


def abort_session(session_id: int) -> dict:
    session = status(session_id)
    if session["status"] == "committed":
        raise ValueError("已提交的会话不能中止")
    db.execute("UPDATE collect_sessions SET status='aborted' WHERE id=?", (session_id,))
    if session.get("path"):
        shutil.rmtree(session["path"], ignore_errors=True)
    return status(session_id)


# --------------------------------------------------------------------------- 提交：写进数据集

def commit_session(session_id: int, *, success: bool | None = None,
                   note: str | None = None) -> dict:
    session = status(session_id)
    if session["status"] != "open":
        raise ValueError(f"会话状态为 {session['status']}，不能提交")
    frames_file = Path(session["path"]) / "frames.jsonl"
    records = [json.loads(line) for line in frames_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("会话里没有任何帧，不能提交")

    timestamps = [r["t"] for r in records]
    states = [r["state"] for r in records]
    actions = [r.get("action") for r in records]
    if any(a is None for a in actions):
        actions = states[1:] + [states[-1]]      # 未单独上报动作时，用下一帧状态（遥操作/回放惯例）

    images: dict[str, list[bytes]] = {}
    for key in session["image_keys"]:
        directory = Path(session["path"]) / "images" / key
        files = sorted(directory.glob("*.png"))
        images[key] = [f.read_bytes() for f in files]

    # 时间戳健康度（与质检同口径，先给客户端反馈）
    steps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    expected = 1.0 / float(session["fps"] or 10.0)
    step_dev = round(max(abs(s - expected) / expected for s in steps), 4) if steps else 0.0
    monotonic = all(s >= 0 for s in steps)

    dataset_dir = db.STORAGE / "datasets" / session["dataset_name"]
    existed = (dataset_dir / "meta" / "info.json").is_file()
    appended = append_episode(
        dataset_dir,
        vectors={"observation.state": states, "action": actions},
        timestamps=timestamps,
        images=images or None,
        task_text=session["task_text"],
        fps=float(session["fps"] or 10.0),
        robot_type=session["robot_type"],
        joint_names=session["joint_names"],
        success=success,
    )

    result = {
        "episode_index": appended["episode_index"], "rows": appended["rows"],
        "duration_s": appended["duration_s"], "video_frames": appended["video_frames"],
        "dataset_path": str(dataset_dir), "dataset_existed": existed,
        "step_dev_max": step_dev, "timestamps_monotonic": monotonic,
        "fps": session["fps"], "task_text": session["task_text"],
        "note": note, "success": success,
    }
    db.execute(
        "UPDATE collect_sessions SET status='committed', episode_index=?, committed_at=?,"
        " result_json=?, note=COALESCE(?, note) WHERE id=?",
        (appended["episode_index"], db.now(),
         json.dumps(result, ensure_ascii=False, default=str), note, session_id))
    return {**status(session_id), **result}


def append_episode(dataset_dir: Path, *, vectors: dict[str, list[list[float]]],
                   timestamps: list[float], images: dict[str, list[bytes]] | None,
                   task_text: str, fps: float, robot_type: str,
                   joint_names: list[str], success: bool | None = None,
                   image_key: str = DEFAULT_IMAGE_KEY) -> dict:
    """把一段新采数据追加成数据集里的一条 episode（不存在则按 LeRobot v2.1 新建）。"""
    lerobot.require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_dir = Path(dataset_dir)
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    rows = len(vectors["observation.state"])
    state_dim = len(vectors["observation.state"][0])
    video_keys = list((images or {}).keys())
    # 图像形状：从第一帧解出来，用于 features 的 shape 声明
    image_shape = None
    if video_keys and HAS_CV2:
        frame = cv2.imdecode(np.frombuffer(images[video_keys[0]][0], np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            image_shape = list(frame.shape)

    if (meta_dir / "info.json").is_file():
        info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
        declared = (info.get("features", {}).get("observation.state") or {}).get("shape") or [state_dim]
        if int(declared[0]) != state_dim:
            raise ValueError(f"state 维度与数据集声明不符：数据集 {declared[0]}，本次 {state_dim}")
    else:
        info = {
            "codebase_version": "v2.1", "robot_type": robot_type, "fps": float(fps),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size": 1000, "total_chunks": 1,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [state_dim],
                                      "names": {"joints": joint_names}},
                "action": {"dtype": "float32", "shape": [state_dim],
                           "names": {"joints": joint_names}},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "next.reward": {"dtype": "float32", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
                "next.success": {"dtype": "bool", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
    for key in video_keys:
        feature = info["features"].setdefault(key, {
            "dtype": "video", "shape": image_shape or [0, 0, 3],
            "names": ["height", "width", "channel"],
        })
        feature["shape"] = image_shape or feature.get("shape")
        feature["video_info"] = {"video.fps": float(fps), "video.codec": "mp4v",
                                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                                 "has_audio": False}

    episode_index = int(info.get("total_episodes") or 0)
    t0 = timestamps[0]
    relative = [round(t - t0, 6) for t in timestamps]
    done = [i == rows - 1 for i in range(rows)]
    if success is None:
        success_flags = [False] * rows
        reward = [0.0] * rows
    else:
        success_flags = [False] * (rows - 1) + [bool(success)]
        reward = [0.0] * (rows - 1) + [1.0 if success else 0.0]

    # 任务文本 -> task_index
    tasks_file = meta_dir / "tasks.jsonl"
    tasks: dict[int, str] = {}
    if tasks_file.is_file():
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                tasks[int(row["task_index"])] = row["task"]
    task_index = next((k for k, v in tasks.items() if v == task_text), None)
    if task_index is None:
        task_index = (max(tasks) + 1) if tasks else 0
        tasks[task_index] = task_text

    table = pa.table({
        "observation.state": pa.array(vectors["observation.state"],
                                      type=pa.list_(pa.float32(), state_dim)),
        "action": pa.array(vectors["action"], type=pa.list_(pa.float32(), state_dim)),
        "episode_index": pa.array([episode_index] * rows, type=pa.int64()),
        "frame_index": pa.array(list(range(rows)), type=pa.int64()),
        "timestamp": pa.array(relative, type=pa.float32()),
        "next.reward": pa.array(reward, type=pa.float32()),
        "next.done": pa.array(done, type=pa.bool_()),
        "next.success": pa.array(success_flags, type=pa.bool_()),
        "index": pa.array(list(range(int(info.get("total_frames") or 0),
                                   int(info.get("total_frames") or 0) + rows)), type=pa.int64()),
        "task_index": pa.array([task_index] * rows, type=pa.int64()),
    })
    parquet_path = dataset_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    pq.write_table(table, parquet_path)

    video_frames = None
    for key in video_keys:
        video_dir = dataset_dir / "videos" / "chunk-000" / key
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"episode_{episode_index:06d}.mp4"
        video_frames = _encode_video(images[key], video_path, float(fps))

    with tasks_file.open("w", encoding="utf-8") as fh:
        for k in sorted(tasks):
            fh.write(json.dumps({"task_index": k, "task": tasks[k]}, ensure_ascii=False) + "\n")
    with (meta_dir / "episodes.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"episode_index": episode_index, "tasks": [task_text],
                             "length": rows}, ensure_ascii=False) + "\n")
    with (meta_dir / "episodes_stats.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"episode_index": episode_index,
                             "stats": lerobot._episode_stats(table, info)},
                            ensure_ascii=False) + "\n")

    info["total_episodes"] = episode_index + 1
    info["total_frames"] = int(info.get("total_frames") or 0) + rows
    info["total_videos"] = int(info.get("total_videos") or 0) + len(video_keys)
    info["total_tasks"] = len(tasks)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{episode_index + 1}"}
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False),
                                        encoding="utf-8")

    return {"episode_index": episode_index, "rows": rows, "task_index": task_index,
            "duration_s": round(relative[-1], 3) if relative else 0.0,
            "video_frames": video_frames, "parquet": str(parquet_path),
            "image_shape": image_shape}


def _encode_video(frames: list[bytes], target: Path, fps: float) -> int:
    """把 PNG 帧序列编码成 mp4（LeRobot 的 video feature）。"""
    if not HAS_CV2:
        raise RuntimeError("缺少 opencv(cv2)/numpy，无法编码视频；请 pip install opencv-python")
    first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
                             (width, height))
    written = 0
    try:
        for payload in frames:
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
            written += 1
    finally:
        writer.release()
    return written
