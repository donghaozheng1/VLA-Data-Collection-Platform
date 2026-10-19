"""画布类标注：目标框 / 关键点 / 阶段切分。

和语言指令、成功位的区别：这些是**结构化**标注，落在 `spatial_annotations` 表里，
导出时写进 `meta/annotations.json`（平台约定 —— LeRobot v2.1 契约里没有承载空间标注的字段，
所以**不改** info.json 的 features，避免伪造一个不存在的契约）。

这个模块的重点是**校验**：坐标越界、宽高非正、阶段起止反了、帧号超出 episode 长度，
全部在入库前拒掉，并把原因说清楚。乱数据一旦进了库，后面所有统计都会被污染。
"""
from __future__ import annotations

import json
from pathlib import Path

from . import audit, db, lerobot

KINDS = ("bbox", "keypoint", "phase")
BBOX_FORMAT = "normalized xywh（0~1，左上角为原点）"


def _episode_length(dataset_id: int, episode_index: int) -> int:
    row = db.query_one("SELECT data_rows, video_frames FROM episodes WHERE dataset_id=?"
                       " AND episode_index=?", (dataset_id, episode_index))
    if not row:
        raise ValueError(f"数据集 #{dataset_id} 不存在 episode {episode_index}")
    return int(row["data_rows"] or row["video_frames"] or 0)


def _as_float(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字，收到 {value!r}") from exc


def validate(dataset_id: int, episode_index: int, kind: str, label: str | None,
             data: dict, frame_index: int | None = None) -> dict:
    """校验一条标注，返回规范化后的 data。任何不合法都抛 ValueError。"""
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一，收到 {kind!r}")
    if not (label or "").strip():
        raise ValueError("label 不能为空（框/关键点/阶段都要有名字，否则统计没有意义）")
    if not isinstance(data, dict):
        raise ValueError("data 必须是对象")

    length = _episode_length(dataset_id, episode_index)
    clean: dict = {}

    if kind in ("bbox", "keypoint"):
        index = 0 if frame_index is None else int(frame_index)
        if length and not 0 <= index < length:
            raise ValueError(f"frame_index {index} 超出 episode 帧范围 0~{max(length - 1, 0)}")
        clean["frame_index"] = index

    if kind == "bbox":
        x, y = _as_float(data.get("x"), "x"), _as_float(data.get("y"), "y")
        w, h = _as_float(data.get("w"), "w"), _as_float(data.get("h"), "h")
        if w <= 0 or h <= 0:
            raise ValueError(f"框的宽高必须为正（收到 w={w}, h={h}）")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"框左上角必须归一化在 0~1（收到 x={x}, y={y}）")
        if x + w > 1.001 or y + h > 1.001:
            raise ValueError(f"框超出图像边界（x+w={x + w:.3f}, y+h={y + h:.3f} 应 ≤ 1）")
        clean.update({"x": round(x, 6), "y": round(y, 6),
                      "w": round(min(w, 1 - x), 6), "h": round(min(h, 1 - y), 6)})
    elif kind == "keypoint":
        x, y = _as_float(data.get("x"), "x"), _as_float(data.get("y"), "y")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"关键点必须归一化在 0~1（收到 x={x}, y={y}）")
        clean.update({"x": round(x, 6), "y": round(y, 6)})
        if data.get("visibility") is not None:
            visibility = _as_float(data["visibility"], "visibility")
            if not 0 <= visibility <= 1:
                raise ValueError(f"visibility 必须在 0~1（收到 {visibility}）")
            clean["visibility"] = round(visibility, 4)
    else:  # phase
        start = int(_as_float(data.get("start"), "start"))
        end = int(_as_float(data.get("end"), "end"))
        if start >= end:
            raise ValueError(f"阶段起点必须小于终点（收到 start={start}, end={end}）")
        if length and end > length:
            raise ValueError(f"阶段终点 {end} 超出 episode 长度 {length}")
        clean.update({"start": start, "end": end})
    return clean


def put(dataset_id: int, episode_index: int, kind: str, label: str, data: dict,
        frame_index: int | None = None, annotator: str = "admin") -> dict:
    clean = validate(dataset_id, episode_index, kind, label, data, frame_index)
    row = db.query_one("SELECT name FROM datasets WHERE id=?", (dataset_id,))
    annotation_id = db.execute(
        "INSERT INTO spatial_annotations (dataset_id, episode_index, kind, label, data_json,"
        " frame_index, annotator, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (dataset_id, episode_index, kind, label.strip(), json.dumps(clean, ensure_ascii=False),
         clean.get("frame_index"), annotator, db.now()))
    audit.log("INFO", "数据标注",
              f"数据集 #{dataset_id} ep{episode_index} 新增 {kind} 标注「{label}」",
              ref_type="episode", ref_id=episode_index, actor=annotator,
              detail={"kind": kind, "label": label, "data": clean})
    return {"id": annotation_id, "dataset_id": dataset_id, "dataset": row["name"] if row else None,
            "episode_index": episode_index, "kind": kind, "label": label, "data": clean}


def items(dataset_id: int | None = None, episode_index: int | None = None,
          kind: str | None = None, limit: int = 1000) -> list[dict]:
    sql = "SELECT * FROM spatial_annotations WHERE 1=1"
    params: list = []
    if dataset_id is not None:
        sql += " AND dataset_id=?"
        params.append(dataset_id)
    if episode_index is not None:
        sql += " AND episode_index=?"
        params.append(episode_index)
    if kind:
        sql += " AND kind=?"
        params.append(kind)
    sql += " ORDER BY episode_index, kind, id LIMIT ?"
    params.append(max(1, min(int(limit), 5000)))
    rows = db.query(sql, tuple(params))
    for row in rows:
        try:
            row["data"] = json.loads(row.pop("data_json") or "{}")
        except Exception:  # noqa: BLE001
            row["data"] = {}
    return rows


def delete(annotation_id: int) -> dict:
    row = db.query_one("SELECT * FROM spatial_annotations WHERE id=?", (annotation_id,))
    if not row:
        raise ValueError(f"标注不存在：{annotation_id}")
    db.execute("DELETE FROM spatial_annotations WHERE id=?", (annotation_id,))
    audit.log("WARNING", "数据标注",
              f"删除 {row['kind']} 标注「{row['label']}」（数据集 #{row['dataset_id']} "
              f"ep{row['episode_index']}）", ref_id=annotation_id)
    return {"id": annotation_id, "deleted": True}


def summary(dataset_id: int) -> dict:
    rows = items(dataset_id, limit=5000)
    total_episodes = db.scalar("SELECT COUNT(*) AS c FROM episodes WHERE dataset_id=?", (dataset_id,))
    by_kind: dict[str, int] = {}
    by_label: dict[str, int] = {}
    episodes: set[int] = set()
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
        by_label[f"{row['kind']}:{row['label']}"] = by_label.get(f"{row['kind']}:{row['label']}", 0) + 1
        episodes.add(row["episode_index"])
    return {"dataset_id": dataset_id, "total": len(rows), "by_kind": by_kind,
            "labels": sorted(by_label.items(), key=lambda kv: -kv[1])[:20],
            "episodes_with_spatial": len(episodes), "episodes_total": total_episodes,
            "coverage": round(len(episodes) / total_episodes, 4) if total_episodes else 0.0}


def payload(dataset_id: int, episode_indices: list[int] | None = None) -> dict:
    """导出用：组装写进 meta/annotations.json 的结构。"""
    rows = items(dataset_id, limit=5000)
    if episode_indices is not None:
        wanted = set(episode_indices)
        rows = [r for r in rows if r["episode_index"] in wanted]
    dataset = db.query_one("SELECT name FROM datasets WHERE id=?", (dataset_id,))
    episodes: dict[str, dict] = {}
    for row in rows:
        slot = episodes.setdefault(str(row["episode_index"]), {"bbox": [], "keypoint": [], "phase": []})
        slot[row["kind"]].append({"label": row["label"], "frame_index": row["frame_index"],
                                  "data": row["data"], "annotator": row["annotator"]})
    return {"tool": "hkc-platform", "dataset": dataset["name"] if dataset else None,
            "created_at": db.now(), "bbox_format": BBOX_FORMAT,
            "note": "平台约定的空间/阶段标注；LeRobot v2.1 契约无此字段，故不改 info.json 的 features",
            "counts": {kind: sum(len(v[kind]) for v in episodes.values()) for kind in KINDS},
            "episodes": episodes}


def write_meta(dataset_dir: str | Path, dataset_id: int,
               episode_indices: list[int] | None = None) -> dict:
    """把标注写进数据集目录的 meta/annotations.json（尽力而为，目录只读时如实报告）。"""
    meta_dir = Path(dataset_dir) / "meta"
    if not meta_dir.is_dir():
        return {"written": None, "error": "目录里没有 meta/"}
    data = payload(dataset_id, episode_indices)
    if not any(data["counts"].values()):
        return {"written": None, "error": None, "skipped": "该数据集没有画布类标注"}
    try:
        target = meta_dir / "annotations.json"
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"written": str(target), "counts": data["counts"]}
    except OSError as exc:
        return {"written": None, "error": f"{type(exc).__name__}: {exc}"}


def photo_of_episode(dataset_id: int, episode_index: int, key: str | None = None) -> str | None:
    """取该 episode 的视频路径（画布从视频里抓帧）。"""
    row = db.query_one("SELECT path FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError(f"数据集不存在：{dataset_id}")
    meta = lerobot.load_meta(row["path"])
    paths = lerobot.episode_paths(meta["info"], row["path"], episode_index)
    if key:
        return paths["videos"].get(key)
    return next(iter(paths["videos"].values()), None)
