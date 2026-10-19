"""跨数据集重复 episode 检测。

数据平台很容易踩的坑：同一批 episode 在多个数据集里各存了一份 —— 有的是**有意的派生**
（preprocess / repair / export 的产物，血缘里写得清清楚楚），有的是**意外的重复**
（同一批数据被接入两次、导出后又被当新数据接入）。

平台的判断分两级，都基于**内容**而不是文件名：

1. `content_sha`：对 state / action / timestamp 三列的**取值**做哈希（浮点先归一，绕开
   parquet 写入差异）—— 语义相同即同一条，用于精确重复。
2. `sample_sha`：均匀抽 16 行、取 3 位小数再哈希 —— 用于**近似**重复（例如 repair 掉了两帧，
   长度从 161 变 159，但采样行一致）。

指纹按 episode 缓存在 `episode_hashes` 表里，重复扫描不必重算。

**平台不自动删任何东西**：只报告。删除数据是不可逆的，判断哪个副本该留必须由人来做；
平台能做的是把「哪些是血缘可解释的派生、哪些是无血缘的可疑重复」分清楚。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import db, jobs, lerobot

SAMPLE_ROWS = 16


def _norm(value, digits: int = 5):
    """浮点归一，避免 float32/float64 与写入差异造成的假不同。"""
    if isinstance(value, (list, tuple)):
        return [_norm(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def _column_values(table, key: str) -> list | None:
    if key not in table.column_names:
        return None
    return [_norm(v) for v in table.column(key).to_pylist()]


def fingerprint(dataset_id: int, episode_index: int, parquet_path: str | Path) -> dict:
    """算一条 episode 的三种指纹。"""
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    file_bytes = path.read_bytes()
    table = pq.read_table(path)

    payload = {}
    for key in ("observation.state", "action", "timestamp"):
        values = _column_values(table, key)
        if values is not None:
            payload[key] = values
    content_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    rows = table.num_rows
    step = max(1, rows // SAMPLE_ROWS) if rows else 1
    sample = {key: values[::step][:SAMPLE_ROWS] for key, values in payload.items()}
    sample_sha = hashlib.sha256(
        json.dumps({"rows": rows, "sample": sample}, sort_keys=True, default=str)
        .encode("utf-8")).hexdigest()
    near_sha = hashlib.sha256(
        json.dumps({"sample": {k: _norm(v, 3) for k, v in sample.items()}},
                   sort_keys=True, default=str).encode("utf-8")).hexdigest()

    return {"dataset_id": dataset_id, "episode_index": episode_index,
            "exact_sha": hashlib.sha256(file_bytes).hexdigest(),
            "content_sha": content_sha, "sample_sha": sample_sha, "near_sha": near_sha,
            "rows": rows, "bytes": len(file_bytes)}


@jobs.handler("dedup")
def dedup_job(job: jobs.JobContext, payload: dict) -> dict:
    """后台任务：算指纹 + 出报告。"""
    dataset_ids = payload.get("dataset_ids")
    force = bool(payload.get("force"))
    return build(dataset_ids=dataset_ids, force=force, progress=job.progress)


def build(*, dataset_ids: list[int] | None = None, force: bool = False,
          progress=None) -> dict:
    """计算并缓存指纹，然后出重复报告。"""
    sql = "SELECT * FROM datasets ORDER BY id"
    datasets = db.query(sql)
    if dataset_ids:
        wanted = set(int(x) for x in dataset_ids)
        datasets = [d for d in datasets if d["id"] in wanted]

    computed = 0
    skipped = 0
    errors: list[str] = []
    total = len(datasets)
    for position, dataset in enumerate(datasets, start=1):
        if progress:
            progress(position, total, f"扫描数据集 #{dataset['id']} {dataset['name']}")
        if not force:
            known = db.scalar("SELECT COUNT(*) AS c FROM episode_hashes WHERE dataset_id=?",
                              (dataset["id"],))
            if known >= (dataset["total_episodes"] or 0) and known > 0:
                skipped += 1
                continue
        try:
            meta = lerobot.load_meta(dataset["path"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"数据集 #{dataset['id']}: {type(exc).__name__}: {exc}")
            continue
        rows = db.query("SELECT episode_index, parquet_path FROM episodes"
                        " WHERE dataset_id=? ORDER BY episode_index", (dataset["id"],))
        for row in rows:
            if not row["parquet_path"] or not Path(row["parquet_path"]).is_file():
                continue
            try:
                data = fingerprint(dataset["id"], row["episode_index"], row["parquet_path"])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"#{dataset['id']}/ep{row['episode_index']}: "
                              f"{type(exc).__name__}: {exc}")
                continue
            db.execute(
                "INSERT INTO episode_hashes (dataset_id, episode_index, exact_sha, content_sha,"
                " sample_sha, near_sha, rows, bytes, computed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(dataset_id, episode_index) DO UPDATE SET"
                " exact_sha=excluded.exact_sha, content_sha=excluded.content_sha,"
                " sample_sha=excluded.sample_sha, near_sha=excluded.near_sha,"
                " rows=excluded.rows, bytes=excluded.bytes, computed_at=excluded.computed_at",
                (data["dataset_id"], data["episode_index"], data["exact_sha"], data["content_sha"],
                 data["sample_sha"], data["near_sha"], data["rows"], data["bytes"], db.now()))
            computed += 1
    return {"computed": computed, "skipped_datasets": skipped, "errors": errors[:10],
            "report": report(dataset_ids)}


def components() -> dict[int, int]:
    """血缘族：把 parent_dataset_id 连成的图求连通分量。

    为什么用连通分量而不是"直接祖先"：`#22` 派生于 `#2`，而 `#2` 与 `#1` 内容相同 ——
    按直接祖先判，`#1` 与 `#22` 会被算成"无血缘"，于是同一个组里冒出一堆传递性的假告警。
    按族判，`{#1,#21,#23}` 与 `{#2,#22,#26}` 是两个族，跨族共享内容才是**一条**真告警。
    """
    rows = db.query("SELECT id, parent_dataset_id FROM datasets")
    parent = {r["id"]: r["id"] for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in rows:
        child, ancestor = row["id"], row["parent_dataset_id"]
        if ancestor and ancestor in parent:
            root_a, root_c = find(ancestor), find(child)
            if root_a != root_c:
                parent[root_c] = root_a
    return {dataset_id: find(dataset_id) for dataset_id in parent}


def report(dataset_ids: list[int] | None = None, limit: int = 200) -> dict:
    """按内容指纹分组，输出重复/近似重复组并分类。"""
    sql = ("SELECT h.*, d.name AS dataset_name, d.parent_dataset_id, d.derived_by"
           " FROM episode_hashes h JOIN datasets d ON d.id = h.dataset_id")
    params: list = []
    if dataset_ids:
        placeholders = ",".join("?" * len(dataset_ids))
        sql += f" WHERE h.dataset_id IN ({placeholders})"
        params.extend(int(x) for x in dataset_ids)
    rows = db.query(sql, tuple(params))
    component_of = components()

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["content_sha"],), []).append(row)
    near_groups: dict[tuple, list[dict]] = {}
    for row in rows:
        near_groups.setdefault((row["near_sha"],), []).append(row)

    def classify(items: list[dict]) -> dict:
        """按**血缘族**判定，一个组只给一条结论。"""
        dataset_ids_in = sorted({i["dataset_id"] for i in items})
        families: dict[int, list[int]] = {}
        for dataset_id in dataset_ids_in:
            families.setdefault(component_of.get(dataset_id, dataset_id), []).append(dataset_id)
        family_list = sorted(families.values(), key=lambda f: f[0])
        if len(dataset_ids_in) == 1:
            label = "同数据集内重复"
        elif len(family_list) == 1:
            label = "派生重复（同一血缘族）"
        else:
            label = "跨血缘族重复（需人工判断）"
        return {"classification": label, "families": family_list,
                "cross_family": len(family_list) > 1 and len(dataset_ids_in) > 1}

    def as_group(key, items, kind) -> dict:
        lengths = sorted({i["rows"] for i in items})
        verdict = classify(items)
        byte_identical = len({i["exact_sha"] for i in items}) == 1
        return {
            "kind": kind,
            "byte_identical": byte_identical,
            "sameness": "字节完全相同" if byte_identical else "内容相同（文件字节不同）",
            "dataset_count": len({i["dataset_id"] for i in items}),
            "count": len(items),
            **verdict,
            "lengths": lengths,
            "members": [{"dataset_id": i["dataset_id"], "dataset": i["dataset_name"],
                         "episode_index": i["episode_index"], "rows": i["rows"],
                         "derived_by": i["derived_by"],
                         "parent_dataset_id": i["parent_dataset_id"]} for i in items],
        }

    exact = [as_group(k, v, "精确重复") for k, v in groups.items() if len(v) > 1]
    exact_member_sets = [{(i["dataset_id"], i["episode_index"]) for i in v}
                         for k, v in groups.items() if len(v) > 1]
    near = []
    for key, v in near_groups.items():
        if len(v) < 2:
            continue
        member_set = {(i["dataset_id"], i["episode_index"]) for i in v}
        if any(member_set == s for s in exact_member_sets):
            continue          # 与某个精确组是同一批 episode，不重复报告
        near.append(as_group(key, v, "近似重复"))

    def sort_key(g):
        return (-g["count"], g["classification"])

    exact.sort(key=sort_key)
    near.sort(key=sort_key)
    by_class: dict[str, int] = {}
    for group in exact + near:
        by_class[group["classification"]] = by_class.get(group["classification"], 0) + 1

    warnings = []
    for group in exact + near:
        if not group.get("cross_family"):
            continue
        names = {}
        for member in group["members"]:
            names.setdefault(member["dataset_id"], member["dataset"])
        described = " 与 ".join(
            "族[" + "、".join(f"#{d} {names.get(d, d)}" for d in family) + "]"
            for family in group["families"])
        warnings.append(
            f"{described} 共享完全相同的 episode（{group['count']} 条，长度 {group['lengths']}），"
            f"但血缘上属于不同的族 —— 若同时用于训练/评估会造成数据泄漏，请人工确认保留哪份")

    return {
        "hashed_episodes": len(rows),
        "datasets_scanned": len({r["dataset_id"] for r in rows}),
        "exact_groups": len(exact),
        "near_groups": len(near),
        "by_classification": by_class,
        "unexplained_groups": sum(1 for g in exact + near if g.get("cross_family")),
        "warnings": warnings[:20],
        "groups": (exact + near)[:limit],
        "note": "平台只报告不删除：派生重复（血缘可解释）是预期内的；"
                "无血缘重复需要人工确认保留哪份。",
    }
