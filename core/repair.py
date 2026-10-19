"""修正类预处理：把质检发现的问题**真正修掉**，产出一个能通过质检的新数据集。

与 preprocess（按规则筛 episode）的区别：这里会**改帧**——删帧、修时间戳、重建视频。

支持的操作（可组合）：
    drop_broken        丢弃视频缺失/损坏的 episode（这类问题无法修，只能剔）
    drop_dup_episodes  跨 episode 去重：轨迹（重采样后逐点比较）近似相同的只留最早一条
    drop_invalid       剔除 state/action 里含 NaN/Inf 的帧
    dedup_ts           剔除时间戳重复或回退的帧
    align_ts           时间戳对齐到 1/fps 的均匀网格（不改帧数）
    resample           按 fps 把轨迹线性插值到目标长度（会**生成中间帧**，默认目标取声明长度）
    sync_video         按最终保留的帧重建视频，使「行数 == 视频帧数」重新成立
                       （只要有任何 episode 重建，就统一重建全部，避免同一数据集里编码混杂）

写出的新数据集沿用 LeRobot v2.1 结构；`meta/episodes.jsonl` 的 length 会按修复后的真实长度重写，
所以修复后「长度一致性」检查会通过。
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from . import db, lerobot

try:
    import cv2
    HAS_CV2 = True
except Exception:  # noqa: BLE001
    HAS_CV2 = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # noqa: BLE001
    pa = None
    pq = None

OPS = {
    "drop_broken": "丢弃视频缺失/损坏的 episode（不可修）",
    "drop_dup_episodes": "跨 episode 去重（轨迹近似相同只留最早一条）",
    "drop_invalid": "剔除 NaN/Inf 帧",
    "dedup_ts": "剔除重复/回退时间戳的帧",
    "align_ts": "时间戳对齐到 1/fps 均匀网格（不改帧数）",
    "resample": "按 fps 线性插值到目标长度（会生成中间帧）",
    "sync_video": "按保留帧重建视频，使行数==帧数",
}
DEFAULT_OPS = ["drop_broken", "drop_dup_episodes", "drop_invalid", "dedup_ts", "align_ts", "sync_video"]


# --------------------------------------------------------------------------- 小工具

def _interp(series: list[float], src_ts: list[float], dst_ts: list[float]) -> list[float]:
    """按时间线性插值（dst_ts 递增）。"""
    out: list[float] = []
    j = 0
    n = len(src_ts)
    for t in dst_ts:
        while j + 1 < n and src_ts[j + 1] < t:
            j += 1
        if j + 1 >= n:
            out.append(series[-1])
            continue
        t0, t1 = src_ts[j], src_ts[j + 1]
        if t1 == t0:
            out.append(series[j])
            continue
        w = (t - t0) / (t1 - t0)
        out.append(series[j] * (1.0 - w) + series[j + 1] * w)
    return out


def _traj_signature(vectors: list[list[float]], samples: int = 16) -> list[float]:
    """把一条轨迹压成固定采样点的签名，用于跨 episode 去重比较。"""
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    idx = [0] if n == 1 else [round(i * (n - 1) / (samples - 1)) for i in range(samples)]
    return [vectors[i][d] for i in idx for d in range(dim)]


def _max_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _rebuild_video(src_video: Path, dst_video: Path, source_indices: list[int], fps: float) -> dict:
    """按给定的源帧序号重建视频（序号非递减；重复序号表示该帧被复用）。"""
    if not HAS_CV2:
        raise RuntimeError("缺少 opencv(cv2)，无法重建视频；请 pip install opencv-python")
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"源视频打不开：{src_video.name}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height) if width and height else None
    writer = cv2.VideoWriter(str(dst_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps) or 10.0, size)
    written = 0
    src_idx = 0
    ti = 0
    try:
        while ti < len(source_indices):
            ok, frame = cap.read()
            if not ok:
                break
            if source_indices[ti] == src_idx:
                while ti < len(source_indices) and source_indices[ti] == src_idx:
                    writer.write(frame)
                    written += 1
                    ti += 1
            src_idx += 1
    finally:
        cap.release()
        writer.release()
    return {"frames": written, "width": width, "height": height,
            "short": written < len(source_indices)}


# --------------------------------------------------------------------------- 主流程

def repair_dataset(src_path, meta: dict, out_dir, name: str, ops: list[str] | None = None,
                   *, resample_to: int | None = None, dup_tol: float = 1e-6,
                   annotations: dict | None = None) -> dict:
    lerobot.require_pyarrow()
    ops = list(ops or DEFAULT_OPS)
    unknown = [op for op in ops if op not in OPS]
    if unknown:
        raise ValueError(f"未知的修复操作：{unknown}（可用：{list(OPS)}）")

    src = Path(src_path)
    info = dict(meta["info"])
    fps = float(info.get("fps") or 0) or 10.0
    video_keys = lerobot.video_keys(info)
    vector_keys = lerobot.vector_keys(info)
    annotations = annotations or {}

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for key in video_keys:
        (out / "videos" / "chunk-000" / key).mkdir(parents=True, exist_ok=True)

    episodes = sorted(meta["episodes"]) or list(range(int(info.get("total_episodes") or 0)))

    # ---- 1) 逐条探测 + episode 级筛选（丢弃不可修 / 重复）----
    plan: list[dict] = []
    dropped: list[dict] = []
    seen_sig: list[tuple[int, list[float]]] = []

    for episode_index in episodes:
        record = lerobot.probe_episode(src, meta, episode_index, with_video=True)
        paths = lerobot.episode_paths(info, src, episode_index)
        reason = None

        if not record["data_exists"]:
            reason = "缺少 parquet"
        elif "drop_broken" in ops and video_keys:
            if not all(v.get("exists") for v in record["videos"].values()):
                reason = "视频缺失"
            elif record.get("video_ok") is False:
                reason = "视频损坏不可解码"
        if reason:
            dropped.append({"episode_index": episode_index, "reason": reason})
            continue

        if "drop_dup_episodes" in ops and vector_keys:
            table = pq.read_table(paths["parquet"], columns=vector_keys)
            first_key = vector_keys[0]
            sig = _traj_signature([list(v) for v in table.to_pydict()[first_key]])
            duplicate_of = next((idx for idx, other in seen_sig
                                 if sig and _max_diff(sig, other) <= dup_tol), None)
            if duplicate_of is not None:
                dropped.append({"episode_index": episode_index,
                                "reason": f"与 ep{duplicate_of} 轨迹近似重复（最大差 ≤ {dup_tol}）"})
                continue
            seen_sig.append((episode_index, sig))

        plan.append({"episode_index": episode_index, "record": record, "paths": paths})

    # ---- 2) 逐条做帧级修复 ----
    episodes_meta: list[dict] = []
    stats_out: list[dict] = []
    report_rows: list[dict] = []
    total_frames = 0
    global_index = 0
    video_rebuilt = 0
    task_of_episode: dict[int, str] = {}

    for new_index, item in enumerate(plan):
        episode_index = item["episode_index"]
        paths = item["paths"]
        table = pq.read_table(paths["parquet"])
        rows_before = table.num_rows
        data = table.to_pydict()
        timestamps_all = [float(t) for t in (data.get("timestamp") or [])]
        vectors_all = {key: [list(v) for v in (data.get(key) or [])] for key in vector_keys}

        # (a) 帧级掩码：保留的**原始行号**
        keep = list(range(rows_before))
        if "drop_invalid" in ops and vector_keys:
            keep = [i for i in keep if all(
                value is not None and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
                for key in vector_keys for value in vectors_all[key][i])]
        if "dedup_ts" in ops and timestamps_all:
            kept: list[int] = []
            last = None
            for i in keep:
                if last is None or timestamps_all[i] > last:
                    kept.append(i)
                    last = timestamps_all[i]
            keep = kept
        dropped_frames = rows_before - len(keep)

        # (b) 目标长度与「最终行 -> 原始行」映射（视频与标量列都按它取）
        declared = (meta["episodes"].get(episode_index) or {}).get("length")
        target_len = len(keep)
        if "resample" in ops:
            target_len = max(1, int(resample_to or declared
                                    or item["record"].get("video_frames") or len(keep)))

        if "resample" in ops and target_len != len(keep) and keep:
            src_ts = [timestamps_all[i] for i in keep]
            t0, t1 = src_ts[0], src_ts[-1]
            dst_ts = ([t0] * target_len if t1 <= t0 else
                      [t0 + (t1 - t0) * k / max(1, target_len - 1) for k in range(target_len)])
            mapping = [keep[_nearest(src_ts, t)] for t in dst_ts]
            vectors = {}
            for key in vector_keys:
                dim = len(vectors_all[key][keep[0]]) if vectors_all[key] else 0
                rows_out: list[list[float]] = [[] for _ in range(target_len)]
                for d in range(dim):
                    column = [vectors_all[key][i][d] for i in keep]
                    for k, value in enumerate(_interp(column, src_ts, dst_ts)):
                        rows_out[k].append(value)
                vectors[key] = rows_out
            timestamps = dst_ts
        else:
            mapping = list(keep)
            vectors = {key: [vectors_all[key][i] for i in keep] for key in vector_keys}
            timestamps = ([round(k / fps, 6) for k in range(len(keep))]
                          if "align_ts" in ops else [timestamps_all[i] for i in keep])

        final_rows = len(mapping)
        if final_rows == 0:
            dropped.append({"episode_index": episode_index, "reason": "修复后没有可用帧"})
            continue

        # (c) 组装所有列：向量列用修复后的，其余标量列一律按 mapping 重取（长度必须一致）
        columns: dict[str, list] = {}
        for col in table.column_names:
            original = data.get(col) or []
            if col in vector_keys:
                columns[col] = vectors[col]
            elif col == "episode_index":
                columns[col] = [new_index] * final_rows
            elif col == "frame_index":
                columns[col] = list(range(final_rows))
            elif col == "timestamp":
                columns[col] = [float(t) for t in timestamps]
            elif col == "index":
                columns[col] = list(range(global_index, global_index + final_rows))
            elif col == "next.done":
                columns[col] = [i == final_rows - 1 for i in range(final_rows)]
            else:
                columns[col] = [original[i] for i in mapping]

        ann = annotations.get(episode_index) or {}
        if ann.get("success") is not None and "next.success" in columns:
            columns["next.success"] = [False] * (final_rows - 1) + [bool(ann["success"])]
            if "next.reward" in columns:
                columns["next.reward"] = [0.0] * (final_rows - 1) + [1.0 if ann["success"] else 0.0]

        new_table = pa.table({col: pa.array(values, type=table.schema.field(col).type)
                              for col, values in columns.items()})
        pq.write_table(new_table, out / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet")

        rebuilt = []
        for key in video_keys:
            src_video = paths["videos"].get(key)
            if not src_video or not src_video.is_file():
                continue
            dst_video = out / "videos" / "chunk-000" / key / f"episode_{new_index:06d}.mp4"
            rebuilt.append(_rebuild_video(src_video, dst_video, mapping, fps))
            video_rebuilt += 1

        declared_tasks = (meta["episodes"].get(episode_index) or {}).get("tasks") or []
        task_text = ((ann.get("task_text") or "").strip() or (declared_tasks[0] if declared_tasks else None)
                      or (list(meta["tasks"].values())[0] if meta["tasks"] else "未命名任务"))
        task_of_episode[new_index] = task_text
        episodes_meta.append({"episode_index": new_index, "tasks": [task_text], "length": final_rows})
        stats_out.append({"episode_index": new_index, "stats": lerobot._episode_stats(new_table, info)})
        report_rows.append({
            "episode_index": episode_index, "new_index": new_index,
            "rows_before": rows_before, "rows_after": final_rows,
            "dropped_frames": dropped_frames,
            "resampled": "resample" in ops and final_rows != len(keep),
            "video_frames_before": item["record"].get("video_frames"),
            "video_frames_after": rebuilt[0]["frames"] if rebuilt else None,
            "video_short": any(r["short"] for r in rebuilt),
        })
        total_frames += final_rows
        global_index += final_rows

    # ---- 3) 写 meta ----
    for key in video_keys:
        vinfo = info["features"][key].setdefault("video_info", {})
        vinfo["video.codec"] = "mp4v" if video_rebuilt else vinfo.get("video.codec")
        vinfo["video.fps"] = fps
    info["total_episodes"] = len(episodes_meta)
    info["total_frames"] = total_frames
    info["total_videos"] = len(episodes_meta) * len(video_keys)
    info["total_tasks"] = len(set(task_of_episode.values())) or 1
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(episodes_meta)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False),
                                            encoding="utf-8")

    task_texts = sorted(set(task_of_episode.values())) or ["未命名任务"]
    task_index = {text: i for i, text in enumerate(task_texts)}
    for row in report_rows:
        path = out / "data" / "chunk-000" / f"episode_{row['new_index']:06d}.parquet"
        tbl = pq.read_table(path)
        if "task_index" in tbl.column_names:
            value = task_index[task_of_episode[row["new_index"]]]
            tbl = tbl.set_column(tbl.schema.get_field_index("task_index"), "task_index",
                                 pa.array([value] * tbl.num_rows, type=pa.int64()))
            pq.write_table(tbl, path)

    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for i, text in enumerate(task_texts):
            fh.write(json.dumps({"task_index": i, "task": text}, ensure_ascii=False) + "\n")
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in episodes_meta:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as fh:
        for row in stats_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {
        "name": name, "path": str(out), "ops": ops, "op_labels": [OPS[o] for o in ops],
        "episodes_in": len(episodes), "episodes_out": len(episodes_meta),
        "frames_out": total_frames, "bytes": size, "fps": fps,
        "dropped_episodes": dropped, "per_episode": report_rows,
        "frames_dropped": sum(r["dropped_frames"] for r in report_rows),
        "videos_rebuilt": video_rebuilt,
    }

def _nearest(sorted_values: list[float], target: float) -> int:
    if not sorted_values:
        return 0
    lo, hi = 0, len(sorted_values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and abs(sorted_values[lo - 1] - target) <= abs(sorted_values[lo] - target):
        best = lo - 1
    return best
