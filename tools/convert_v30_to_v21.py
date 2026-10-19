"""把 LeRobot v3.0 数据集转换为 v2.1（供平台接入/训练）。

背景：平台的读取器只认 v2.1（每 episode 一个 parquet + jsonl 元数据），
而 zhiyan_dataset 是 v3.0（多 episode 合一个 parquet + parquet 元数据 + 时间戳切分视频）。

    python tools/convert_v30_to_v21.py \
        --src zhiyan_dataset/zhiyan_pusht_dataset \
        --dest storage/datasets/zhiyan_pick_place

另外会**修正 action**：v3.0 这份数据的 action 与 state 完全相同（自复制），
训练会退化成恒等映射。默认改成 action[t] = state[t+1]（末帧复制），可用 --keep-action 关闭。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from core.lerobot import _episode_stats  # noqa: E402  复用平台的统计函数


def read_jsonlines(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="zhiyan_dataset/zhiyan_pusht_dataset")
    ap.add_argument("--dest", default="storage/datasets/zhiyan_pick_place")
    ap.add_argument("--keep-action", action="store_true", help="不修正 action（保留原始自复制）")
    ap.add_argument("--no-video", action="store_true", help="跳过视频切段（只转数据）")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dest = pathlib.Path(args.dest)
    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        print(f"[警告] 源数据集版本是 {info.get('codebase_version')}，本脚本针对 v3.0")

    fps = float(info["fps"])
    features = info["features"]
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    scalar_cols = [k for k, v in features.items()
                   if v.get("dtype") in ("int64", "float32", "bool") and k not in ("task_index",)]
    print(f"== 源: {src}")
    print(f"   v{info.get('codebase_version')} / {info['total_episodes']} 条 / {info['total_frames']} 帧 / {fps:g}fps")
    print(f"   相机: {video_keys}")
    print(f"   标量列: {scalar_cols}")

    # ---- 任务表 ----
    tasks_tbl = pq.read_table(src / "meta" / "tasks.parquet").to_pydict()
    tasks = {int(tasks_tbl["task_index"][i]): tasks_tbl["task"][i] for i in range(len(tasks_tbl["task"]))}
    print(f"   任务: {tasks}")

    # ---- 全局 stats（v3.0 的 stats.json 可给视频特征沿用）----
    global_stats: dict = {}
    stats_file = src / "meta" / "stats.json"
    if stats_file.is_file():
        try:
            global_stats = json.loads(stats_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            global_stats = {}

    # ---- episode 元数据 ----
    ep_files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    episodes: list[dict] = []
    for f in ep_files:
        d = pq.read_table(f).to_pydict()
        for i in range(len(d["episode_index"])):
            episodes.append({k: d[k][i] for k in d})
    episodes.sort(key=lambda e: int(e["episode_index"]))
    print(f"\n== 准备写出 {len(episodes)} 条 episode 到 {dest}")

    if dest.exists():
        shutil.rmtree(dest)
    (dest / "meta").mkdir(parents=True)
    (dest / "data" / "chunk-000").mkdir(parents=True)
    for key in video_keys:
        (dest / "videos" / "chunk-000" / key).mkdir(parents=True)

    # ---- 一次性读入全部数据（2647 行，很小）----
    data_files = sorted((src / "data").rglob("*.parquet"))
    tables = [pq.read_table(f) for f in data_files]
    data = pa.concat_tables(tables).to_pydict()
    total_rows = len(data["episode_index"])
    print(f"   读入 {total_rows} 行数据（{len(data_files)} 个文件）")

    episodes_meta: list[dict] = []
    stats_out: list[dict] = []
    global_index = 0
    total_frames_out = 0

    for new_index, ep in enumerate(episodes):
        old_index = int(ep["episode_index"])
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"]) + 1
        rows = end - start
        task_text = str(ep.get("task") or tasks.get(0) or "task")
        task_index = next((k for k, v in tasks.items() if v == task_text), 0)

        states = np.asarray(data["observation.state"][start:end], dtype=np.float32)
        actions = np.asarray(data["action"][start:end], dtype=np.float32)
        action_note = "原样"
        if not args.keep_action and rows > 1 and np.allclose(actions, states, atol=1e-6):
            actions = np.vstack([states[1:], states[-1:]])      # action[t] = state[t+1]
            action_note = "已修正为下一帧"

        state_dim = states.shape[1]
        columns = {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), state_dim)),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), state_dim)),
            "timestamp": pa.array([float(t) for t in data["timestamp"][start:end]], type=pa.float32()),
            "frame_index": pa.array(list(range(rows)), type=pa.int64()),
            "episode_index": pa.array([new_index] * rows, type=pa.int64()),
            "index": pa.array(list(range(global_index, global_index + rows)), type=pa.int64()),
            "task_index": pa.array([task_index] * rows, type=pa.int64()),
        }
        for col in ("next.reward", "next.done", "next.success"):
            if col in data:
                values = data[col][start:end]
                dtype = pa.bool_() if isinstance(values[0], (bool, np.bool_)) else pa.float32()
                columns[col] = pa.array(values, type=dtype)

        table = pa.table(columns)
        out_pq = dest / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet"
        pq.write_table(table, out_pq)

        episodes_meta.append({"episode_index": new_index, "tasks": [task_text], "length": rows})
        # 注意：必须把**含视频特征**的 features 一起传进去，否则图像统计会缺失，
        # LeRobot 建数据集时会报 KeyError: 'observation.images.xxx'（踩过一次）
        stats_out.append({"episode_index": new_index,
                          "stats": _episode_stats(table, {"features": features},
                                                  inherited=global_stats)})

        # ---- 视频切段 ----
        if not args.no_video:
            for key in video_keys:
                src_video_dir = src / "videos" / key
                vfiles = sorted(src_video_dir.rglob("*.mp4"))
                chunk_index = int(ep.get(f"videos/{key}/chunk_index", 0) or 0)
                file_index = int(ep.get(f"videos/{key}/file_index", 0) or 0)
                video_path = next((v for v in vfiles if v.stem == f"file-{file_index:03d}"), None)
                if video_path is None and vfiles:
                    video_path = vfiles[0]
                if video_path is None:
                    continue
                from_ts = float(ep.get(f"videos/{key}/from_timestamp", 0.0) or 0.0)
                to_ts = float(ep.get(f"videos/{key}/to_timestamp", from_ts + rows / fps) or 0.0)
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(from_ts * fps)))
                out_video = dest / "videos" / "chunk-000" / key / f"episode_{new_index:06d}.mp4"
                writer, written = None, 0
                for i in range(rows):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None:
                        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
                    written += 1
                cap.release()
                if writer is not None:
                    writer.release()
                print(f"   ep{old_index}->{new_index} [{key}] {written}/{rows} 帧  "
                      f"({from_ts:.2f}s→{to_ts:.2f}s, {video_path.name})", flush=True)

        print(f"   ep{old_index}->{new_index}: {rows} 帧 state/action {state_dim} 维 "
              f"（action {action_note}）task={task_text}", flush=True)
        global_index += rows
        total_frames_out += rows

    # ---- v2.1 元数据 ----
    out_features = {}
    for k, v in features.items():
        if k in ("episode_index", "frame_index", "timestamp", "index", "task_index", "next.reward",
                 "next.done", "next.success"):
            continue
        if v.get("dtype") == "float32" and k == "observation.state":
            out_features[k] = {"dtype": "float32", "shape": [state_dim], "names": None}
        elif k == "action":
            out_features[k] = {"dtype": "float32", "shape": [state_dim], "names": None}
        elif v.get("dtype") in ("video", "image"):
            # ★ names 必须是维度名列表：LeRobot 用 names[2] 判断通道轴做 HWC->CHW
            #   （写成 None 会 TypeError: 'NoneType' object is not subscriptable，踩过一次）
            out_features[k] = {"dtype": v.get("dtype"), "shape": v.get("shape"),
                               "names": ["height", "width", "channels"]}
        else:
            out_features[k] = v
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        out_features[k] = {"dtype": "float32" if k == "timestamp" else "int64", "shape": [1], "names": None}

    out_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type"),
        "total_episodes": len(episodes),
        "total_frames": total_frames_out,
        "total_tasks": len(tasks),
        "total_videos": len(video_keys) * len(episodes),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": out_features,
        "source": {"converted_from": "v3.0", "original": str(src),
                   "action_fixed": not args.keep_action},
    }
    (dest / "meta" / "info.json").write_text(json.dumps(out_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in episodes_meta) + "\n", encoding="utf-8")
    (dest / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": k, "task": v}, ensure_ascii=False) for k, v in sorted(tasks.items())) + "\n",
        encoding="utf-8")
    (dest / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in stats_out) + "\n", encoding="utf-8")
    (dest / "meta" / "stats.json").write_text(json.dumps(global_stats, ensure_ascii=False), encoding="utf-8")

    print(f"\n== 完成 → {dest}")
    print(f"   {len(episodes)} 条 / {total_frames_out} 帧 / {len(video_keys)} 路相机 / {fps:g}fps")
    print(f"   meta: {sorted(p.name for p in (dest / 'meta').iterdir())}")
    for key in video_keys:
        n = len(list((dest / "videos" / "chunk-000" / key).glob("*.mp4")))
        print(f"   视频 [{key}]: {n} 个")
    print(f"   action 修正: {'否（保留原样）' if args.keep_action else '是（action[t]=state[t+1]）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
