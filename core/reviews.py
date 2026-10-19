"""标注质量抽检：抽检批次、交叉复核、一致性指标（含 Cohen's Kappa）、争议复核。

流程：
1. `sample()` 从「已标注」的 episode 里按**可复现**规则抽一批（写进 review_batches，可审计；
   同一个 salt 抽出的结果稳定，换 salt 即换一批）
2. `put_review()` 由复核员提交**独立**标签（界面默认盲标：不展示主标结果）
3. `agreement()` 算一致率与 Cohen's Kappa（成功位）+ 指令文本一致率，并列出争议项
4. `resolve()` 由第三方裁决，把最终标签写回主标注表，并把该复核标记为已解决

Kappa 口径（二分类，成功位）：
    po = (a+d)/n, pe = ((a+b)/n)((a+c)/n) + ((c+d)/n)((b+d)/n), kappa = (po-pe)/(1-pe)
其中 a=双方都判成功, b=仅主标成功, c=仅复核成功, d=双方都判失败。
"""
from __future__ import annotations

import hashlib
import random
import json

from . import audit, db


def _sample_key(dataset_id: int, episode_index: int, salt: str) -> str:
    return hashlib.sha256(f"{dataset_id}:{episode_index}:{salt}".encode("utf-8")).hexdigest()


def annotated_episodes(dataset_id: int) -> list[dict]:
    return db.query(
        "SELECT episode_index, task_text, success, annotator, updated_at FROM annotations"
        " WHERE dataset_id=? ORDER BY episode_index", (dataset_id,))


def sample(dataset_id: int, *, ratio: float | None = None, count: int | None = None,
           salt: str = "", note: str | None = None) -> dict:
    """按可复现规则抽检，并记录批次。默认抽 10%。"""
    pool = annotated_episodes(dataset_id)
    if not pool:
        raise ValueError("该数据集还没有任何标注，无法抽检")
    if count is None:
        ratio = 0.1 if ratio is None else max(0.0, min(1.0, float(ratio)))
        count = max(1, round(len(pool) * ratio))
    count = max(1, min(int(count), len(pool)))
    ordered = sorted(pool, key=lambda e: _sample_key(dataset_id, e["episode_index"], salt))
    picked = ordered[:count]
    batch_id = db.execute(
        "INSERT INTO review_batches (dataset_id, salt, ratio, requested, episodes_json, note,"
        " created_at) VALUES (?,?,?,?,?,?,?)",
        (dataset_id, salt, ratio, count,
         json.dumps([e["episode_index"] for e in picked]), note, db.now()))
    return {"batch_id": batch_id, "dataset_id": dataset_id, "pool": len(pool),
            "sampled": len(picked), "episodes": [e["episode_index"] for e in picked],
            "salt": salt, "ratio": ratio, "note": note, "created_at": db.now()}


def batches(dataset_id: int | None = None, limit: int = 20) -> list[dict]:
    if dataset_id:
        rows = db.query("SELECT * FROM review_batches WHERE dataset_id=? ORDER BY id DESC LIMIT ?",
                        (dataset_id, limit))
    else:
        rows = db.query("SELECT * FROM review_batches ORDER BY id DESC LIMIT ?", (limit,))
    for row in rows:
        row["episodes"] = json.loads(row.get("episodes_json") or "[]")
        row["sampled"] = len(row["episodes"])
        row["reviewed_total"] = db.scalar(
            "SELECT COUNT(*) AS c FROM annotation_reviews WHERE dataset_id=?", (row["dataset_id"],))
    return rows


def put_review(dataset_id: int, episode_index: int, *, reviewer: str, task_text: str | None,
               success: int | None, note: str | None = None) -> dict:
    if not db.query_one("SELECT id FROM annotations WHERE dataset_id=? AND episode_index=?",
                        (dataset_id, episode_index)):
        raise ValueError("该 episode 还没有主标注，不能复核")
    db.execute(
        """INSERT INTO annotation_reviews (dataset_id, episode_index, reviewer, task_text, success,
                note, resolved, created_at) VALUES (?,?,?,?,?,?,0,?)
           ON CONFLICT(dataset_id, episode_index, reviewer) DO UPDATE SET
                task_text=excluded.task_text, success=excluded.success, note=excluded.note,
                resolved=0, created_at=excluded.created_at""",
        (dataset_id, episode_index, reviewer, task_text,
         None if success is None else int(bool(success)), note, db.now()))
    audit.log("INFO", "标注抽检",
              f"数据集 #{dataset_id} ep{episode_index} 提交复核（{reviewer}）："
              f"成功位={success} 指令={(task_text or '')[:40]!r}",
              ref_type="episode", ref_id=episode_index, actor=reviewer,
              detail={"reviewer": reviewer, "dataset_id": dataset_id})
    return db.query_one(
        "SELECT * FROM annotation_reviews WHERE dataset_id=? AND episode_index=? AND reviewer=?",
        (dataset_id, episode_index, reviewer))


def _norm(text: str | None) -> str:
    return " ".join((text or "").split()).strip().lower()



def _kappa_from_pairs(pairs: list[tuple[bool, bool]]) -> float | None:
    """由配对标签直接算 Cohen's Kappa（成功位）。"""
    n = len(pairs)
    if not n:
        return None
    a = sum(1 for p, r in pairs if p and r)
    b = sum(1 for p, r in pairs if p and not r)
    c = sum(1 for p, r in pairs if not p and r)
    d = sum(1 for p, r in pairs if not p and not r)
    po = (a + d) / n
    pe = (a + b) / n * (a + c) / n + (c + d) / n * (b + d) / n
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def _bootstrap_kappa(pairs: list[tuple[bool, bool]], resamples: int = 2000,
                     alpha: float = 0.05, seed: int = 20260101) -> list[float] | None:
    """Kappa 的 bootstrap 百分位置信区间。

    固定随机种子 -> 同一份数据每次得到完全相同的区间（可复现、可写进报告）。
    样本太少或标签退化（pe>=1）时返回 None，不硬凑一个看起来精确的数字。
    """
    n = len(pairs)
    if n < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        kappa = _kappa_from_pairs(sample)
        if kappa is not None:
            values.append(kappa)
    if not values:
        return None
    values.sort()
    lo = values[max(0, int(len(values) * (alpha / 2)))]
    hi = values[min(len(values) - 1, int(len(values) * (1 - alpha / 2)))]
    return [round(lo, 4), round(hi, 4)]

def agreement(dataset_id: int, reviewer: str | None = None) -> dict:
    """一致率 + Cohen's Kappa（成功位）+ 文本一致率 + 争议清单。"""
    sql = ("SELECT a.episode_index, a.task_text AS primary_task, a.success AS primary_success,"
           "       r.task_text AS review_task, r.success AS review_success, r.reviewer, r.resolved"
           " FROM annotation_reviews r JOIN annotations a"
           "   ON a.dataset_id = r.dataset_id AND a.episode_index = r.episode_index"
           " WHERE r.dataset_id = ?")
    params: list = [dataset_id]
    if reviewer:
        sql += " AND r.reviewer = ?"
        params.append(reviewer)
    rows = db.query(sql, tuple(params))

    a = b = c = d = 0
    text_same = text_total = 0
    disputes: list[dict] = []
    pairs: list[tuple[bool, bool]] = []
    for row in rows:
        ps, rs = row["primary_success"], row["review_success"]
        if ps is not None and rs is not None:
            pairs.append((bool(ps), bool(rs)))
            if ps and rs:
                a += 1
            elif ps and not rs:
                b += 1
            elif not ps and rs:
                c += 1
            else:
                d += 1
        if row["primary_task"] is not None or row["review_task"] is not None:
            text_total += 1
            if _norm(row["primary_task"]) == _norm(row["review_task"]):
                text_same += 1
        success_diff = ps is not None and rs is not None and bool(ps) != bool(rs)
        text_diff = _norm(row["primary_task"]) != _norm(row["review_task"])
        if success_diff or text_diff:
            disputes.append({**row, "success_diff": success_diff, "text_diff": text_diff})

    n = a + b + c + d
    po = (a + d) / n if n else None
    pe = kappa = None
    if n:
        p_yes = (a + b) / n * (a + c) / n
        p_no = (c + d) / n * (b + d) / n
        pe = p_yes + p_no
        kappa = None if pe >= 1 else round((po - pe) / (1 - pe), 4)
    return {
        "dataset_id": dataset_id, "reviewer": reviewer,
        "reviewed": len(rows), "success_pairs": n,
        "agreement": round(po, 4) if po is not None else None,
        "expected_agreement": round(pe, 4) if pe is not None else None,
        "kappa": kappa,
        "kappa_level": _kappa_level(kappa),
        "kappa_ci": _bootstrap_kappa(pairs),
        "kappa_ci_note": "bootstrap 百分位区间（2000 次重采样，固定随机种子，同数据可复现）",
        "reliable": n >= 30,
        "reliable_note": None if n >= 30 else "样本量 < 30，Kappa 与区间只作参考",
        "text_agreement": round(text_same / text_total, 4) if text_total else None,
        "table": {"both_ok": a, "primary_ok_only": b, "review_ok_only": c, "both_bad": d},
        "disputes": disputes,
        "dispute_count": len(disputes),
        "unresolved": sum(1 for x in disputes if not x["resolved"]),
    }


def _kappa_level(kappa: float | None) -> str:
    if kappa is None:
        return "样本不足或无可变标签"
    if kappa >= 0.8:
        return "几乎完全一致"
    if kappa >= 0.6:
        return "高度一致"
    if kappa >= 0.4:
        return "中等一致"
    if kappa >= 0.2:
        return "一般一致"
    return "一致性差，需重训标注规范"


def resolve(dataset_id: int, episode_index: int, *, reviewer: str, decision: str,
            task_text: str | None = None, success: int | None = None,
            note: str | None = None, resolver: str = "admin") -> dict:
    """裁决争议：最终标签写回主标注表，并把该复核标记为已解决。"""
    current = db.query_one("SELECT * FROM annotations WHERE dataset_id=? AND episode_index=?",
                           (dataset_id, episode_index))
    review = db.query_one("SELECT * FROM annotation_reviews WHERE dataset_id=? AND episode_index=?"
                          " AND reviewer=?", (dataset_id, episode_index, reviewer))
    if review is None:
        raise ValueError("找不到该复核记录")
    if decision == "primary":
        if current is None:
            raise ValueError("没有主标注可采纳")
        final_task, final_success = current["task_text"], current["success"]
    elif decision == "review":
        final_task, final_success = review["task_text"], review["success"]
    elif decision == "custom":
        final_task = task_text if task_text is not None else (current or {}).get("task_text")
        final_success = success if success is not None else (current or {}).get("success")
    else:
        raise ValueError("decision 只能是 primary / review / custom")

    db.execute(
        """INSERT INTO annotations (dataset_id, episode_index, task_text, success, note, annotator,
                updated_at) VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(dataset_id, episode_index) DO UPDATE SET
                task_text=excluded.task_text, success=excluded.success, note=excluded.note,
                annotator=excluded.annotator, updated_at=excluded.updated_at""",
        (dataset_id, episode_index, final_task,
         None if final_success is None else int(bool(final_success)),
         note or f"争议裁决（{decision}）by {resolver}", resolver, db.now()))
    db.execute("UPDATE annotation_reviews SET resolved=1, resolution_note=? WHERE id=?",
               (note or f"裁决采纳 {decision}", review["id"]))
    audit.log("WARNING", "标注抽检",
              f"数据集 #{dataset_id} ep{episode_index} 争议裁决：采纳 {decision}（{resolver}）",
              ref_type="episode", ref_id=episode_index,
              detail={"decision": decision, "final_success": final_success, "note": note})
    return {"dataset_id": dataset_id, "episode_index": episode_index, "decision": decision,
            "final_task_text": final_task, "final_success": final_success, "resolver": resolver}
