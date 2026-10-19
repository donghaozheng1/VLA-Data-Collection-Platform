"""把机器人自己采集的原始关节轨迹转换成真实 LeRobot v2.1 数据集。

真实结构（实测自 C:\\Users\\ll\\Desktop\\dhz\\hkc\\*.parquet，14 段）：
    timestamp  double   Unix 秒（如 1769068262.0197）
    joint_1..joint_7  double   RealMan 7 轴关节角（度），采样间隔约 0.1s（≈10Hz）
    行数 42 ~ 452

转换约定（沿用 LeRobot 的通用惯例）：
    observation.state = 当前帧关节角 (7)
    action            = 下一帧关节角 (7)，末帧沿用自身
    timestamp         = 相对该 episode 起点的秒数
    next.done         = 末帧 True
    next.success      = False（留待标注）
    无视频特征 —— 属于纯状态/动作轨迹数据集，质检会自动跳过视频类检查。
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from . import db, lerobot

JOINT_COLUMNS = [f"joint_{i}" for i in range(1, 8)]
RAW_COLUMNS = ["timestamp"] + JOINT_COLUMNS

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # noqa: BLE001
    pa = None
    pq = None


def find_raw_episodes(root: str | os.PathLike) -> list[dict]:
    """找出根目录下所有符合「timestamp + joint_1..7」结构的 parquet。"""
    if pq is None:
        lerobot.require_pyarrow()
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(root_path.glob("*.parquet")):
        try:
            schema = pq.read_schema(path)
        except Exception:
            continue
        names = set(schema.names)
        if {"timestamp", *JOINT_COLUMNS} <= names:
            rows = pq.read_metadata(path).num_rows
            out.append({"name": path.stem, "path": str(path), "rows": rows})
    return out


def read_trajectory(path: str | os.PathLike) -> dict:
    """读一段真实关节轨迹。"""
    table = pq.read_table(path, columns=RAW_COLUMNS)
    data = table.to_pydict()
    timestamps = [float(t) for t in data["timestamp"]]
    joints = [[float(data[col][i]) for col in JOINT_COLUMNS] for i in range(table.num_rows)]
    steps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    median_step = statistics.median(steps) if steps else 0.1
    return {
        "rows": table.num_rows,
        "timestamps": timestamps,
        "joints": joints,
        "fps": round(1.0 / median_step, 3) if median_step > 0 else round(1.0 / 0.1, 3),
        "duration_s": round(timestamps[-1] - timestamps[0], 3) if timestamps else 0.0,
        "bytes": Path(path).stat().st_size,
    }


def convert(files: list[dict], out_dir: str | os.PathLike, name: str,
            task_text: str = "抓取放置任务（待标注）", fps: float | None = None) -> dict:
    """把若干段原始轨迹写成一个真实可用的 LeRobot v2.1 数据集。"""
    lerobot.require_pyarrow()
    if not files:
        raise ValueError("没有可转换的原始轨迹")

    out = Path(out_dir)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # 先读一遍，确定全局 fps
    parsed = []
    for item in files:
        traj = read_trajectory(item["path"])
        traj["name"] = item["name"]
        traj["path"] = item["path"]
        parsed.append(traj)
    fps = float(fps or statistics.median(t["fps"] for t in parsed) or 10.0)

    episodes_meta: list[dict] = []
    stats_out: list[dict] = []
    total_frames = 0
    global_index = 0

    for episode_index, traj in enumerate(parsed):
        joints = traj["joints"]
        rows = len(joints)
        if rows == 0:
            continue
        actions = joints[1:] + [joints[-1]]
        t0 = traj["timestamps"][0]
        rel_ts = [round(t - t0, 6) for t in traj["timestamps"]]

        table = pa.table({
            "observation.state": pa.array(joints, type=pa.list_(pa.float32(), len(JOINT_COLUMNS))),
            "action": pa.array(actions, type=pa.list_(pa.float32(), len(JOINT_COLUMNS))),
            "episode_index": pa.array([episode_index] * rows, type=pa.int64()),
            "frame_index": pa.array(list(range(rows)), type=pa.int64()),
            "timestamp": pa.array(rel_ts, type=pa.float32()),
            "next.reward": pa.array([0.0] * rows, type=pa.float32()),
            "next.done": pa.array([i == rows - 1 for i in range(rows)], type=pa.bool_()),
            "next.success": pa.array([False] * rows, type=pa.bool_()),
            "index": pa.array(list(range(global_index, global_index + rows)), type=pa.int64()),
            "task_index": pa.array([0] * rows, type=pa.int64()),
        })
        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet")

        episodes_meta.append({"episode_index": episode_index, "tasks": [task_text], "length": rows})
        stats_out.append({"episode_index": episode_index, "stats": lerobot._episode_stats(table, _info(fps))})
        total_frames += rows
        global_index += rows

    info = _info(fps)
    info.update({
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "splits": {"train": f"0:{len(episodes_meta)}"},
    })
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"task_index": 0, "task": task_text}, ensure_ascii=False) + "\n")
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in episodes_meta:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as fh:
        for row in stats_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"name": name, "path": str(out), "episodes": len(episodes_meta),
            "frames": total_frames, "fps": fps, "bytes": size,
            "sources": [t["name"] for t in parsed]}


def _info(fps: float) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "realman-7dof",
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "",
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [len(JOINT_COLUMNS)],
                "names": {"joints": JOINT_COLUMNS},
            },
            "action": {
                "dtype": "float32", "shape": [len(JOINT_COLUMNS)],
                "names": {"joints": JOINT_COLUMNS},
            },
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
