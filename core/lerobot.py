"""LeRobot v2.1 数据集读写与质检规则。

真实契约（实测自 C:\\Users\\ll\\Desktop\\VLA\\lerobot 的 pusht 数据集）：

    meta/info.json      总览 + features 声明（observation.image=video、action/state 维度、fps）
    meta/tasks.jsonl    {"task_index":0,"task":"Push the T-shaped block onto the T-shaped target."}
    meta/episodes.jsonl {"episode_index":0,"tasks":[...],"length":161}
    data/chunk-XXX/episode_XXXXXX.parquet
                        observation.state / action (fixed_size_list<float>[N])
                        episode_index / frame_index / timestamp / next.reward / next.done
                        / next.success / index / task_index
    videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4

本模块只做「格式 + 事实」，不做编排；任务编排在 core/pipeline.py。
外部数据集一律只读，导出写到 storage/exports 下。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
    PYARROW_ERROR = ""
except Exception as exc:  # noqa: BLE001
    HAS_PYARROW = False
    PYARROW_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import cv2
    HAS_CV2 = True
except Exception:  # noqa: BLE001
    HAS_CV2 = False

INFO_NAME = "info.json"
TASKS_NAME = "tasks.jsonl"
EPISODES_NAME = "episodes.jsonl"
EPISODES_STATS_NAME = "episodes_stats.jsonl"

# 逐帧标量列（非视频、非向量）
SCALAR_COLUMNS = ["episode_index", "frame_index", "timestamp", "next.reward",
                  "next.done", "next.success", "index", "task_index"]


def require_pyarrow() -> None:
    if not HAS_PYARROW:
        raise RuntimeError(
            "缺少 pyarrow，无法读写 parquet（LeRobot 数据本体）。"
            f"请安装：pip install pyarrow。当前错误：{PYARROW_ERROR}"
        )


# --------------------------------------------------------------------------- 发现与元信息

def find_datasets(root: str | os.PathLike, max_depth: int = 6) -> list[str]:
    """在 root 下找出所有 LeRobot 数据集目录（含 meta/info.json）。"""
    root_path = Path(root)
    if not root_path.exists():
        return []
    found: list[str] = []
    skip = {".git", "node_modules", "__pycache__", ".cache"}
    base_depth = len(root_path.parts)
    for current, dirs, _files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in skip]
        current_path = Path(current)
        if len(current_path.parts) - base_depth > max_depth:
            dirs[:] = []
            continue
        if (current_path / "meta" / INFO_NAME).is_file():
            found.append(str(current_path))
            dirs[:] = []
    return sorted(found)


def load_meta(ds_path: str | os.PathLike) -> dict:
    """读 info.json / tasks.jsonl / episodes.jsonl，返回统一结构。"""
    ds = Path(ds_path)
    info_file = ds / "meta" / INFO_NAME
    if not info_file.is_file():
        raise FileNotFoundError(f"不是 LeRobot 数据集（缺 meta/info.json）：{ds}")
    info = json.loads(info_file.read_text(encoding="utf-8"))

    tasks: dict[int, str] = {}
    tasks_file = ds / "meta" / TASKS_NAME
    if tasks_file.is_file():
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                tasks[int(row["task_index"])] = row["task"]

    episodes: dict[int, dict] = {}
    episodes_file = ds / "meta" / EPISODES_NAME
    if episodes_file.is_file():
        for line in episodes_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                episodes[int(row["episode_index"])] = row

    return {"info": info, "tasks": tasks, "episodes": episodes}


def video_keys(info: dict) -> list[str]:
    return [k for k, v in (info.get("features") or {}).items()
            if isinstance(v, dict) and v.get("dtype") == "video"]


def vector_keys(info: dict) -> list[str]:
    """非视频的向量类特征（action / observation.state 等）。"""
    out = []
    for key, spec in (info.get("features") or {}).items():
        # 相机特征（video / 图像内嵌 parquet 的 image）都不是"向量列"，必须一起排除，
        # 否则内嵌图像数据集会把图像列当成 observation.state 那样处理。
        if not isinstance(spec, dict) or spec.get("dtype") in ("video", "image"):
            continue
        if key in SCALAR_COLUMNS:
            continue
        out.append(key)
    return out


def episode_paths(info: dict, ds_path: str | os.PathLike, index: int) -> dict:
    """按 info.json 里的模板拼出该 episode 的 parquet / 视频路径。"""
    ds = Path(ds_path)
    chunks_size = int(info.get("chunks_size") or 1000)
    chunk = index // chunks_size
    data_rel = (info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_rel = (info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")

    parquet = ds / data_rel.format(episode_chunk=chunk, episode_index=index)
    videos = {
        key: ds / video_rel.format(episode_chunk=chunk, episode_index=index, video_key=key)
        for key in video_keys(info)
    }
    return {"parquet": parquet, "videos": videos}


# --------------------------------------------------------------------------- 逐条 episode 探测

def probe_episode(ds_path: str | os.PathLike, meta: dict, index: int,
                  with_video: bool = True) -> dict:
    """读取一条 episode 的真实事实：行数、维度、NaN、成功位、视频帧数/fps/可解码性。"""
    require_pyarrow()
    info = meta["info"]
    paths = episode_paths(info, ds_path, index)
    parquet_path: Path = paths["parquet"]
    declared = meta["episodes"].get(index) or {}
    declared_length = declared.get("length")

    record: dict = {
        "episode_index": index,
        "declared_length": declared_length,
        "declared_tasks": declared.get("tasks") or [],
        "parquet_path": str(parquet_path),
        "data_exists": parquet_path.is_file(),
        "data_rows": None,
        "data_bytes": parquet_path.stat().st_size if parquet_path.is_file() else 0,
        "action_dim": None,
        "state_dim": None,
        "nan_values": None,
        "timestamp_start": None,
        "timestamp_end": None,
        "timestamp_monotonic": None,
        "timestamp_step_max_dev": None,
        "success_count": None,
        "done_count": None,
        "reward_sum": None,
        "task_index": None,
        "task_text": None,
        "action_constant": None,
        "videos": {},
        "video_bytes": 0,
        "video_frames": None,
        "video_fps": None,
        "video_ok": None,
        "errors": [],
    }

    if not parquet_path.is_file():
        record["errors"].append(f"缺少数据文件 {parquet_path.name}")
        return record

    table = pq.read_table(parquet_path)
    record["data_rows"] = table.num_rows
    data = table.to_pydict()

    vectors = vector_keys(info)
    for key in vectors:
        values = data.get(key)
        if not values:
            continue
        first = values[0]
        dim = len(first) if isinstance(first, (list, tuple)) else 1
        record["action_dim" if key == "action" else f"{key.replace('.', '_')}_dim"] = dim

    # NaN / Inf
    nan_count = 0
    for key in vectors:
        for row in data.get(key) or []:
            items = row if isinstance(row, (list, tuple)) else [row]
            for value in items:
                if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    nan_count += 1
    record["nan_values"] = nan_count

    # 动作是否全程不变
    actions = data.get("action") or []
    if actions:
        record["action_constant"] = all(list(a) == list(actions[0]) for a in actions[1:])
    else:
        record["action_constant"] = None

    # 标量列
    timestamps = data.get("timestamp") or []
    if timestamps:
        record["timestamp_start"] = float(timestamps[0])
        record["timestamp_end"] = float(timestamps[-1])
        diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        record["timestamp_monotonic"] = all(d >= 0 for d in diffs)
        fps = float(info.get("fps") or 0) or None
        if fps and diffs:
            expected = 1.0 / fps
            record["timestamp_step_max_dev"] = round(
                max(abs(d - expected) / expected for d in diffs), 4)
    if data.get("next.success") is not None:
        record["success_count"] = int(sum(bool(v) for v in data["next.success"]))
    if data.get("next.done") is not None:
        record["done_count"] = int(sum(bool(v) for v in data["next.done"]))
    if data.get("next.reward") is not None:
        record["reward_sum"] = round(float(sum(data["next.reward"])), 4)
    if data.get("task_index"):
        task_index = int(data["task_index"][0])
        record["task_index"] = task_index
        record["task_text"] = meta["tasks"].get(task_index)

    # 视频：只读头部 + 试解一帧，不解码整段
    if with_video:
        for key, video_path in paths["videos"].items():
            entry = {"path": str(video_path), "exists": video_path.is_file()}
            if entry["exists"]:
                entry["bytes"] = video_path.stat().st_size
                record["video_bytes"] += entry["bytes"]
            if entry["exists"] and HAS_CV2:
                cap = cv2.VideoCapture(str(video_path))
                entry["opened"] = cap.isOpened()
                if entry["opened"]:
                    entry["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    entry["fps"] = round(float(cap.get(cv2.CAP_PROP_FPS)), 3)
                    entry["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    entry["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ok, frame = cap.read()
                    entry["decodable"] = bool(ok) and frame is not None
                    if record["video_frames"] is None:
                        record["video_frames"] = entry["frames"]
                        record["video_fps"] = entry["fps"]
                        record["video_ok"] = entry["decodable"]
                else:
                    entry["decodable"] = False
                    record["video_ok"] = False
                cap.release()
            elif entry["exists"]:
                entry["skipped"] = "未安装 opencv，跳过视频检查"
            record["videos"][key] = entry

    return record


# --------------------------------------------------------------------------- 质检规则

def check_episode(record: dict, info: dict) -> list[dict]:
    """对一条 episode 的探测结果跑规则，返回 findings（rule / severity / detail）。"""
    findings: list[dict] = []

    def add(rule: str, severity: str, detail: str, measured=None, expected=None):
        findings.append({"rule": rule, "severity": severity, "detail": detail,
                         "measured": None if measured is None else str(measured),
                         "expected": None if expected is None else str(expected)})

    for err in record.get("errors") or []:
        add("文件完整性", "error", err)

    # 视频文件缺失/跳过：必须显式报出来，否则「行数==帧数」检查会因拿不到帧数而静默跳过
    for key, entry in (record.get("videos") or {}).items():
        if not entry.get("exists"):
            add("文件完整性", "error", f"缺少视频文件 {Path(entry.get('path', key)).name}")
        elif entry.get("skipped"):
            add("视频帧数一致性", "info", f"{key}: {entry['skipped']}")
        elif entry.get("frames") is None:
            add("视频帧数一致性", "info", f"{key}: 读不到帧数，无法与 parquet 行数比对")

    # 纯状态/动作数据集（没有视频特征）：VLA 缺少视觉观测，值得提示
    if not video_keys(info):
        add("视觉输入完整性", "info",
            "该数据集没有视频特征，属于纯状态/动作轨迹数据集（VLA 缺少视觉观测）")

    rows = record.get("data_rows")
    declared = record.get("declared_length")
    frames = record.get("video_frames")

    if rows == 0:
        add("空 episode", "error", "parquet 没有任何帧")
    if rows is not None and declared is not None and rows != declared:
        add("长度一致性", "error",
            "parquet 行数与 episodes.jsonl 声明长度不一致",
            measured=f"{rows} 行", expected=f"{declared} 行")
    if rows is not None and frames is not None and rows != frames:
        add("视频帧数一致性", "error",
            "parquet 行数与视频帧数不一致（VLA 训练最常见的坏样本）",
            measured=f"{rows} 行 / {frames} 帧", expected="两者相等")
    if record.get("video_ok") is False:
        add("视频可解码性", "error", "视频打不开或首帧无法解码")
    if rows and frames is None and not HAS_CV2:
        add("视频帧数一致性", "info", "未安装 opencv（cv2），跳过视频帧数检查")

    fps_declared = info.get("fps")
    vid_fps = record.get("video_fps")
    if fps_declared and vid_fps and abs(float(vid_fps) - float(fps_declared)) > 0.1:
        add("帧率一致性", "warning", "视频 fps 与 info.json 声明不一致",
            measured=vid_fps, expected=fps_declared)

    for key, spec in (info.get("features") or {}).items():
        if not isinstance(spec, dict) or spec.get("dtype") == "video":
            continue
        shape = spec.get("shape") or []
        if len(shape) != 1:
            continue
        dim_key = "action_dim" if key == "action" else f"{key.replace('.', '_')}_dim"
        actual = record.get(dim_key)
        if actual is not None and int(shape[0]) != int(actual):
            add("特征维度一致性", "error", f"{key} 维度与 info.json 声明不一致",
                measured=actual, expected=shape[0])

    if record.get("nan_values"):
        add("数值有效性", "error", f"轨迹里存在 {record['nan_values']} 个 NaN/Inf")
    if record.get("timestamp_monotonic") is False:
        # 时间戳回退会让训练算错 delta（策略用相邻帧差），不属于"可接受的抖动"，
        # 而是采集时钟坏了 —— 判 error 打回，走修正类预处理（dedup_ts / align_ts）。
        add("时间戳单调性", "error", "timestamp 非单调递增（时钟回退，训练会用错 delta）")
    dev = record.get("timestamp_step_max_dev")
    if dev is not None and dev > 0.2:
        add("时间戳步长", "warning", "timestamp 步长偏离 1/fps 超过 20%（可能丢帧）",
            measured=f"最大偏差 {dev:.1%}", expected="< 20%")
    if record.get("action_constant") is True:
        add("动作有效性", "warning", "action 全程不变，该条大概率没有有效操作")
    if not record.get("declared_tasks") and not record.get("task_text"):
        add("语言指令完整性", "warning", "该 episode 没有任务描述（VLA 的语言输入缺失）")

    return findings


def summarize(findings: list[dict]) -> dict:
    errors = sum(1 for f in findings if f["severity"] == "error")
    warnings = sum(1 for f in findings if f["severity"] == "warning")
    infos = sum(1 for f in findings if f["severity"] == "info")
    return {"errors": errors, "warnings": warnings, "infos": infos,
            "result": "打回重采" if errors else ("通过（含警告）" if warnings else "通过")}


# --------------------------------------------------------------------------- 导出（真实写 LeRobot）

def export_subset(ds_path: str | os.PathLike, meta: dict, episode_indices: list[int],
                  out_dir: str | os.PathLike, name: str | None = None,
                  annotations: dict | None = None) -> dict:
    """把选中的 episode 导出成一个真实的 LeRobot v2.1 数据集。

    - parquet 重编号（episode_index / index / task_index）
    - 视频原样拷贝（不转码）
    - 重新生成 info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl
    - annotations: {原始 episode_index: {"task_text": str|None, "success": 0|1|None}}
      **标注优先于源数据**：语言指令写进 tasks.jsonl 与 task_index；
      成功位按 LeRobot 惯例写在末帧的 next.success，并把 next.reward 置为 1.0/0.0。
    """
    require_pyarrow()
    import shutil

    src = Path(ds_path)
    info = dict(meta["info"])
    out = Path(out_dir)
    annotations = annotations or {}
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    keys = video_keys(info)
    for key in keys:
        (out / "videos" / "chunk-000" / key).mkdir(parents=True, exist_ok=True)

    fps = float(info.get("fps") or 0)

    # ---- 第一遍：定下每条 episode 的任务文本与成功位（标注优先），并建立任务索引表 ----
    plan: list[dict] = []
    for new_index, episode_index in enumerate(sorted(episode_indices)):
        paths = episode_paths(info, src, episode_index)
        if not paths["parquet"].is_file():
            raise FileNotFoundError(f"缺少数据文件：{paths['parquet']}")
        declared = (meta["episodes"].get(episode_index) or {}).get("tasks") or []
        ann = annotations.get(episode_index) or {}
        text = (ann.get("task_text") or "").strip() or (declared[0] if declared else None)
        if not text:
            # 源数据没写进 episodes.jsonl 时，回落到 parquet 的 task_index -> tasks.jsonl
            column = pq.read_table(paths["parquet"], columns=["task_index"]).to_pydict().get("task_index")
            if column:
                text = meta["tasks"].get(int(column[0]))
        plan.append({"new_index": new_index, "episode_index": episode_index,
                     "task": text, "success": ann.get("success"), "paths": paths})

    task_texts = sorted({p["task"] for p in plan if p["task"]} or set(meta["tasks"].values()))
    if not task_texts:
        task_texts = ["未命名任务"]
    task_index_map = {text: i for i, text in enumerate(task_texts)}

    # ---- 第二遍：写 parquet、拷视频 ----
    episodes_meta: list[dict] = []
    stats_out: list[dict] = []
    # 源数据集的统计：图像/视频特征平台不重算，直接沿用源数据里的（避免丢归一化统计）
    source_stats: dict[int, dict] = {}
    src_stats_file = src / "meta" / "episodes_stats.jsonl"
    if src_stats_file.is_file():
        for line in src_stats_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                source_stats[int(row["episode_index"])] = row.get("stats") or {}
            except Exception:  # noqa: BLE001
                continue
    total_frames = 0
    global_index = 0
    annotations_applied = 0

    for item in plan:
        new_index = item["new_index"]
        task_text = item["task"] or task_texts[0]
        paths = item["paths"]

        table = pq.read_table(paths["parquet"])
        rows = table.num_rows

        table = table.set_column(
            table.schema.get_field_index("episode_index"), "episode_index",
            pa.array([new_index] * rows, type=pa.int64()))
        if "index" in table.column_names:
            table = table.set_column(
                table.schema.get_field_index("index"), "index",
                pa.array(list(range(global_index, global_index + rows)), type=pa.int64()))
        if "task_index" in table.column_names:
            table = table.set_column(
                table.schema.get_field_index("task_index"), "task_index",
                pa.array([task_index_map[task_text]] * rows, type=pa.int64()))

        # 标注的成功位优先：写在末帧（LeRobot 用末帧表示该 episode 的结果）
        if item["success"] is not None and rows and "next.success" in table.column_names:
            success = bool(item["success"])
            table = table.set_column(
                table.schema.get_field_index("next.success"), "next.success",
                pa.array([False] * (rows - 1) + [success], type=pa.bool_()))
            if "next.reward" in table.column_names:
                table = table.set_column(
                    table.schema.get_field_index("next.reward"), "next.reward",
                    pa.array([0.0] * (rows - 1) + [1.0 if success else 0.0], type=pa.float32()))
            annotations_applied += 1

        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet")

        for key in keys:
            src_video = paths["videos"].get(key)
            if src_video and src_video.is_file():
                shutil.copy2(src_video,
                             out / "videos" / "chunk-000" / key / f"episode_{new_index:06d}.mp4")

        episodes_meta.append({"episode_index": new_index, "tasks": [task_text], "length": rows})
        stats_out.append({
            "episode_index": new_index,
            "stats": _episode_stats(table, info, inherited=source_stats.get(episode_index)),
        })
        total_frames += rows
        global_index += rows

    info["total_episodes"] = len(episodes_meta)
    info["total_frames"] = total_frames
    info["total_videos"] = len(episodes_meta) * len(keys)
    info["total_tasks"] = len(task_texts)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(episodes_meta)}"}
    (out / "meta" / INFO_NAME).write_text(json.dumps(info, indent=4, ensure_ascii=False),
                                          encoding="utf-8")

    with (out / "meta" / TASKS_NAME).open("w", encoding="utf-8") as fh:
        for i, text in enumerate(task_texts):
            fh.write(json.dumps({"task_index": i, "task": text}, ensure_ascii=False) + "\n")

    with (out / "meta" / EPISODES_NAME).open("w", encoding="utf-8") as fh:
        for row in episodes_meta:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "meta" / EPISODES_STATS_NAME).open("w", encoding="utf-8") as fh:
        for row in stats_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"name": name or out.name, "path": str(out), "episodes": len(episodes_meta),
            "frames": total_frames, "fps": fps, "bytes": size,
            "duration_s": round(total_frames / fps, 2) if fps else None,
            "tasks": task_texts, "annotations_applied": annotations_applied}

def _episode_stats(table, info: dict, inherited: dict | None = None) -> dict:
    """按 LeRobot 约定算每个特征的 min/max/mean/std/count（供训练归一化用）。"""
    stats: dict = {}
    data = table.to_pydict()
    for key, spec in (info.get("features") or {}).items():
        if not isinstance(spec, dict):
            continue
        if spec.get("dtype") in ("video", "image"):
            # 图像/视频不是数值特征：能沿用源数据集的统计就沿用，没有就跳过。
            # 绝不能把内嵌图像的 dict 丢给 float()（那会 TypeError）。
            if inherited and key in inherited:
                stats[key] = inherited[key]
            continue
        values = data.get(key)
        if not values:
            continue
        if not isinstance(values[0], (list, tuple, int, float)):
            continue                     # 非数值列（结构体等）不猜，直接跳过
        if isinstance(values[0], (list, tuple)):
            if not values[0] or not isinstance(values[0][0], (int, float)):
                continue
            dim = len(values[0])
            cols = [[float(row[d]) for row in values] for d in range(dim)]
        else:
            cols = [[float(v) for v in values]]
        entry = {}
        for stat_name, fn in (("min", min), ("max", max),
                              ("mean", lambda c: sum(c) / len(c)),
                              ("std", _std)):
            entry[stat_name] = [round(fn(col), 6) for col in cols]
        entry["count"] = [len(cols[0])]
        stats[key] = entry
    return stats


def dataset_fingerprint(path: str | os.PathLike) -> dict:
    """对数据集内容做确定性指纹，用于版本管理。

    meta/ 与 data/ 按**文件内容**参与哈希（这两部分决定数据语义）；
    videos/ 只用（相对路径, 字节数）参与，避免为了算哈希把上百 MB 视频全读一遍。
    同一个数据集重复计算结果一致；内容一变（哪怕只改一条标注）哈希就会变。
    """
    import hashlib

    root = Path(path)
    digest = hashlib.sha256()
    files = 0
    total = 0

    for sub in ("meta", "data"):
        directory = root / sub
        if not directory.is_dir():
            continue
        for file in sorted(f for f in directory.rglob("*") if f.is_file()):
            digest.update(file.relative_to(root).as_posix().encode("utf-8"))
            digest.update(file.read_bytes())
            files += 1
            total += file.stat().st_size

    videos = root / "videos"
    if videos.is_dir():
        for file in sorted(f for f in videos.rglob("*") if f.is_file()):
            stat = file.stat()
            digest.update(file.relative_to(root).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            files += 1
            total += stat.st_size

    return {"sha256": digest.hexdigest(), "files": files, "bytes": total}


def _std(col: list[float]) -> float:
    if len(col) < 2:
        return 0.0
    mean = sum(col) / len(col)
    return math.sqrt(sum((v - mean) ** 2 for v in col) / len(col))
