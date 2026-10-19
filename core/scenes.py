"""场景/任务视图：按任务聚合真实统计 + 任务写法归一。

为什么需要归一：实测同一任务在库里存在多种写法 —— 英文 LeRobot 元数据原文
（`Push the T-shaped block onto the T-shaped target`）、平台标注写的中文
（`把 T 形块推到 T 形目标上（平台标注）`）、以及测试产生的 `推动 T 形块 1/2/3`。
平台**不做静默合并**：归一靠 `task_aliases` 表显式映射，未映射的写法单独列出。
"""
from __future__ import annotations

import re

from . import db

TASK_SQL = """
SELECT e.task_text AS task,
       COUNT(*) AS episodes,
       COUNT(DISTINCT e.dataset_id) AS datasets,
       COALESCE(SUM(e.data_rows), 0) AS frames,
       ROUND(COALESCE(SUM(e.duration_s), 0), 1) AS duration_s,
       SUM(CASE WHEN e.success_count > 0 THEN 1 ELSE 0 END) AS success_by_data,
       SUM(CASE WHEN e.nan_values > 0 THEN 1 ELSE 0 END) AS with_nan,
       GROUP_CONCAT(DISTINCT e.dataset_id) AS dataset_ids
FROM episodes e
GROUP BY e.task_text
ORDER BY episodes DESC
"""


def aliases() -> dict[str, str]:
    return {r["alias"]: r["canonical"] for r in db.query("SELECT * FROM task_aliases")}


def canonical_of(task: str, mapping: dict[str, str] | None = None) -> str:
    mapping = mapping if mapping is not None else aliases()
    seen = set()
    current = task
    while current in mapping and current not in seen:      # 防止别名环
        seen.add(current)
        current = mapping[current]
    return current


def set_alias(alias: str, canonical: str, note: str | None = None) -> dict:
    if alias == canonical:
        raise ValueError("别名与目标相同")
    if canonical_of(canonical) == alias or canonical_of(alias) != alias:
        raise ValueError("别名链会形成环，请直接指向最终的任务名")
    db.execute("INSERT INTO task_aliases (alias, canonical, note, created_at) VALUES (?,?,?,?)"
               " ON CONFLICT(alias) DO UPDATE SET canonical=excluded.canonical, note=excluded.note",
               (alias, canonical, note, db.now()))
    return {"alias": alias, "canonical": canonical}


def delete_alias(alias: str) -> dict:
    db.execute("DELETE FROM task_aliases WHERE alias=?", (alias,))
    return {"alias": alias, "deleted": True}


def overview() -> dict:
    """按规范化任务聚合；每个任务保留其全部原始写法。"""
    mapping = aliases()
    dataset_names = {r["id"]: r["name"] for r in
                     db.query("SELECT id, name FROM datasets")}
    annotation_stats = {r["task"]: r for r in db.query("""
        SELECT e.task_text AS task, COUNT(*) AS annotated,
               SUM(CASE WHEN a.success = 1 THEN 1 ELSE 0 END) AS annotated_success,
               SUM(CASE WHEN a.success = 0 THEN 1 ELSE 0 END) AS annotated_fail
        FROM annotations a
        JOIN episodes e ON e.dataset_id = a.dataset_id AND e.episode_index = a.episode_index
        GROUP BY e.task_text
    """)}
    badcase_stats = {r["task"]: r for r in db.query("""
        SELECT e.task_text AS task, COUNT(*) AS cases,
               SUM(CASE WHEN b.source = 'qc' THEN 1 ELSE 0 END) AS qc_cases,
               SUM(CASE WHEN b.status = 'open' THEN 1 ELSE 0 END) AS open_cases
        FROM bad_cases b
        JOIN episodes e ON e.dataset_id = b.dataset_id AND e.episode_index = b.episode_index
        WHERE b.dataset_id IS NOT NULL
        GROUP BY e.task_text
    """)}

    grouped: dict[str, dict] = {}
    for row in db.query(TASK_SQL):
        task = row["task"] or "(无任务文本)"
        canonical = canonical_of(task, mapping)
        bucket = grouped.setdefault(canonical, {
            "task": canonical, "variants": [], "episodes": 0, "datasets": set(),
            "frames": 0, "duration_s": 0.0, "success_by_data": 0, "with_nan": 0,
            "annotated": 0, "annotated_success": 0, "annotated_fail": 0,
            "bad_cases": 0, "qc_cases": 0, "open_cases": 0,
        })
        ids = [int(x) for x in str(row["dataset_ids"] or "").split(",") if x.strip()]
        # 同一数据集可能在多个 variant 下出现，用集合去重
        variant = {
            "task": task, "episodes": row["episodes"], "datasets": len(ids),
            "frames": row["frames"], "duration_s": row["duration_s"],
            "success_by_data": row["success_by_data"] or 0,
            "dataset_names": sorted(dataset_names.get(i, str(i)) for i in ids),
            "is_canonical": task == canonical,
        }
        bucket["variants"].append(variant)
        bucket["episodes"] += row["episodes"]
        bucket["datasets"].update(ids)
        bucket["frames"] += row["frames"]
        bucket["duration_s"] = round(bucket["duration_s"] + (row["duration_s"] or 0), 1)
        bucket["success_by_data"] += row["success_by_data"] or 0
        bucket["with_nan"] += row["with_nan"] or 0
        ann = annotation_stats.get(task) or {}
        bucket["annotated"] += ann.get("annotated") or 0
        bucket["annotated_success"] += ann.get("annotated_success") or 0
        bucket["annotated_fail"] += ann.get("annotated_fail") or 0
        bc = badcase_stats.get(task) or {}
        bucket["bad_cases"] += bc.get("cases") or 0
        bucket["qc_cases"] += bc.get("qc_cases") or 0
        bucket["open_cases"] += bc.get("open_cases") or 0

    tasks = []
    for bucket in grouped.values():
        bucket["datasets"] = sorted(bucket["datasets"])
        bucket["dataset_count"] = len(bucket["datasets"])
        bucket["dataset_names"] = sorted({n for v in bucket["variants"] for n in v["dataset_names"]})
        bucket["variant_count"] = len(bucket["variants"])
        bucket["coverage"] = round(bucket["annotated"] / bucket["episodes"], 3) if bucket["episodes"] else 0.0
        bucket["aliased"] = bucket["variant_count"] > 1
        bucket["variants"].sort(key=lambda v: -v["episodes"])
        tasks.append(bucket)
    tasks.sort(key=lambda b: -b["episodes"])

    return {
        "tasks": tasks,
        "totals": {
            "tasks": len(tasks),
            "raw_variants": sum(t["variant_count"] for t in tasks),
            "episodes": sum(t["episodes"] for t in tasks),
            "frames": sum(t["frames"] for t in tasks),
            "datasets": len({d for t in tasks for d in t["datasets"]}),
            "annotated": sum(t["annotated"] for t in tasks),
            "success_by_data": sum(t["success_by_data"] for t in tasks),
            "aliases": len(mapping),
        },
        "aliases": mapping,
        "suggestions": suggest_duplicates(tasks),
    }


def _normalize(text: str) -> str:
    """粗归一：去掉空白/标点/大小写差异与结尾序号。仅用于"疑似同一任务"的提示。"""
    cleaned = re.sub(r"[\s\-_/（）()【】\[\]、,，.。:：;；!！?？\"']+", "", str(text or "").lower())
    cleaned = re.sub(r"\d+$", "", cleaned)
    return cleaned


def suggest_duplicates(tasks: list[dict] | None = None) -> list[dict]:
    """找出**写法高度相似但尚未归一**的任务，交给人工确认。

    只做近文本匹配（同语言、去标点/序号后相同），跨语言（英文元数据 vs 中文标注）
    不会自动配对 —— 那需要人判断，平台不猜。
    """
    tasks = tasks if tasks is not None else overview()["tasks"]
    buckets: dict[str, list[str]] = {}
    for item in tasks:
        buckets.setdefault(_normalize(item["task"]), []).append(item["task"])
    return [{"normalized": key, "tasks": members}
            for key, members in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
            if len(members) > 1]


def merge(tasks: list[str], canonical: str, note: str | None = None) -> dict:
    """把多个任务名归并到 canonical（写 task_aliases，不静默合并）。"""
    if canonical not in tasks:
        tasks = list(tasks) + [canonical]
    done = []
    for task in tasks:
        if task == canonical:
            continue
        db.execute("INSERT INTO task_aliases (alias, canonical, note, created_at) VALUES (?,?,?,?)"
                   " ON CONFLICT(alias) DO UPDATE SET canonical=excluded.canonical, note=excluded.note",
                   (task, canonical, note, db.now()))
        done.append(task)
    return {"canonical": canonical, "aliased": done}
