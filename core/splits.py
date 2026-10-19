"""可复现的数据划分（train / val / test）。

为什么需要：实测本机所有数据集的 `info.json` 里 `splits` 都是 `{"train": "0:N"}` ——
训练与评估没有隔离，**评估结果无法复现**（评估集一直在训练数据里）。这里补上真正的划分。

两种模式：

- `random`：归属由 `sha256(f"{salt}:{episode_index}")` 决定，**只靠 salt 就能复现**。
  LeRobot 的 `info.json["splits"]` 只能表达连续区间，装不下随机划分，所以这里写
  `meta/splits.json`（平台约定，含 salt/ratios/名单，可审计）。
- `ordered`：按 episode 顺序切连续区间，同时写进 `info.json["splits"]` —— 这是 LeRobot 原生格式，
  但只在该数据集的 episode 下标恰好是 `0..N-1` 连续时才写（否则区间会指错，宁可不写）。

平台不猜：salt / mode / ratios / 每个 episode 的归属全部落库，任何一次划分都能复现与被审计。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import audit, db, lerobot

RATIOS_DEFAULT = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_ORDER = ("train", "val", "test")


def _normalize_ratios(ratios: dict | None) -> dict:
    ratios = {k: float(v) for k, v in (ratios or RATIOS_DEFAULT).items() if float(v) > 0}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("划分比例之和必须大于 0")
    return {k: round(v / total, 6) for k, v in ratios.items()}


def _bucket(u: float, ratios: dict) -> str:
    """u ∈ [0,1) -> 落在哪个 split。"""
    acc = 0.0
    for name in SPLIT_ORDER:
        if name not in ratios:
            continue
        acc += ratios[name]
        if u < acc:
            return name
    return next(reversed([k for k in SPLIT_ORDER if k in ratios]))


def assign(dataset_id: int, *, ratios: dict | None = None, salt: str = "v1",
           mode: str = "random", write_meta: bool = True) -> dict:
    """给数据集划分 train/val/test（可复现）。"""
    if mode not in ("random", "ordered"):
        raise ValueError("mode 只支持 random / ordered")
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError(f"数据集不存在：{dataset_id}")
    ratios = _normalize_ratios(ratios)

    indices = sorted(lerobot.load_meta(row["path"])["episodes"]) or \
        list(range(int(row["total_episodes"] or 0)))
    if not indices:
        raise ValueError("数据集里没有 episode")

    assignment: dict[int, str] = {}
    if mode == "ordered":
        train_n = int(len(indices) * ratios.get("train", 0.8))
        val_n = int(len(indices) * ratios.get("val", 0.0))
        if len(indices) >= 3:                      # 小数据集也尽量三类都有
            train_n = max(1, train_n)
            val_n = max(1, val_n) if "val" in ratios else 0
            if train_n + val_n >= len(indices):
                train_n = len(indices) - val_n - 1
        for position, index in enumerate(indices):
            if position < train_n:
                assignment[index] = "train"
            elif position < train_n + val_n:
                assignment[index] = "val"
            else:
                assignment[index] = "test"
    else:
        for index in indices:
            digest = hashlib.sha256(f"{salt}:{index}".encode("utf-8")).hexdigest()
            u = int(digest[:8], 16) / 0xFFFFFFFF
            assignment[index] = _bucket(u, ratios)

    db.execute("DELETE FROM dataset_splits WHERE dataset_id=?", (dataset_id,))
    db.execute_many(
        "INSERT INTO dataset_splits (dataset_id, episode_index, split, salt, mode, created_at)"
        " VALUES (?,?,?,?,?,?)",
        [(dataset_id, index, split_name, salt, mode, db.now())
         for index, split_name in sorted(assignment.items())])

    counts = {name: sum(1 for v in assignment.values() if v == name) for name in ratios}
    result = {"dataset_id": dataset_id, "name": row["name"], "mode": mode, "salt": salt,
              "ratios": ratios, "counts": counts, "episodes": len(indices),
              "assignment": assignment, "meta_written": [], "warnings": []}
    for name, count in counts.items():
        if count == 0:
            result["warnings"].append(f"split '{name}' 为空（数据集只有 {len(indices)} 条，比例太小）")

    # 元数据写入是"尽力而为"：数据集可能位于只读位置或工作区之外（例如 HuggingFace 缓存），
    # 平台不该假设自己能改数据源目录。**划分的权威记录在平台库里**，写不进去就如实报告。
    if write_meta:
        meta_dir = Path(row["path"]) / "meta"
        if not meta_dir.is_dir():
            result["warnings"].append(f"数据集目录里没有 meta/，跳过元数据写入：{row['path']}")
        else:
            try:
                _write_meta(meta_dir, row, salt, mode, ratios, counts, assignment, result)
            except OSError as exc:
                result["warnings"].append(
                    f"元数据写入失败（{type(exc).__name__}）：{meta_dir} —— "
                    f"划分仍以平台库为准，按划分导出不受影响")
    audit.log("INFO", "数据划分",
              f"数据集 #{dataset_id} {row['name']} 划分（{mode}, salt={salt}）："
              + "、".join(f"{k} {v}" for k, v in counts.items()),
              ref_type="dataset", ref_id=dataset_id,
              detail={"mode": mode, "salt": salt, "ratios": ratios, "counts": counts,
                      "meta_written": result["meta_written"], "warnings": result["warnings"]})
    return result


def _write_meta(meta_dir: Path, row: dict, salt: str, mode: str, ratios: dict,
                counts: dict, assignment: dict, result: dict) -> None:
    """写 meta/splits.json（平台约定）；ordered 且下标连续时同时写 info.json 原生 splits。"""
    if True:
        if True:
            manifest = {
                "tool": "hkc-platform", "dataset": row["name"], "salt": salt, "mode": mode,
                "ratios": ratios, "counts": counts, "created_at": db.now(),
                "note": "平台约定的划分清单；LeRobot 原生 splits 字段只支持连续区间",
                "splits": {name: sorted(i for i, v in assignment.items() if v == name)
                           for name in ratios},
            }
            (meta_dir / "splits.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            result["meta_written"].append(str(meta_dir / "splits.json"))

            # ordered 且下标连续 -> 同时写成 LeRobot 原生格式
            indices = sorted(assignment)
            if mode == "ordered" and indices == list(range(len(indices))):
                info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
                ranges, cursor = {}, 0
                for name in SPLIT_ORDER:
                    if name not in counts:
                        continue
                    ranges[name] = f"{cursor}:{cursor + counts[name]}"
                    cursor += counts[name]
                info["splits"] = ranges
                (meta_dir / "info.json").write_text(
                    json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
                result["meta_written"].append(str(meta_dir / "info.json"))
                result["info_splits"] = ranges


def overview(dataset_id: int) -> dict:
    rows = db.query("SELECT * FROM dataset_splits WHERE dataset_id=? ORDER BY episode_index",
                    (dataset_id,))
    if not rows:
        return {"dataset_id": dataset_id, "assigned": False, "splits": {}}
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return {"dataset_id": dataset_id, "assigned": True, "salt": rows[0]["salt"],
            "mode": rows[0]["mode"], "created_at": rows[0]["created_at"],
            "counts": counts, "total": len(rows),
            "splits": {name: [r["episode_index"] for r in rows if r["split"] == name]
                       for name in sorted(counts)}}


def episodes_of(dataset_id: int, split: str) -> list[int]:
    return [r["episode_index"] for r in db.query(
        "SELECT episode_index FROM dataset_splits WHERE dataset_id=? AND split=? ORDER BY episode_index",
        (dataset_id, split))]


def all_overviews() -> list[dict]:
    out = []
    for row in db.query("SELECT id, name, total_episodes FROM datasets ORDER BY id"):
        item = overview(row["id"])
        item.update({"name": row["name"], "total_episodes": row["total_episodes"]})
        out.append(item)
    return out
