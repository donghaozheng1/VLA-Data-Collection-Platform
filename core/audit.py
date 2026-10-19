"""平台操作审计日志：把真实发生的操作记下来，供 /log 与 /system 展示。

与 /log 页面原来的随机 mock 日志不同，这里记录的是**平台自己真实做过的事**：
任务创建/完成/失败、数据集登记与版本、标注与复核、实验登记、bad case 回灌、采集会话提交等。
"""
from __future__ import annotations

import json
from datetime import datetime

from . import db

LEVELS = ["INFO", "WARNING", "ERROR", "DEBUG"]

# 固定模块名（界面下拉用；实际写入以调用方传入为准）
MODULES = ["数据接入", "数据预处理", "数据质检", "数据标注", "数据集导出", "修正处理",
           "采集会话", "版本管理", "模型训练", "标注抽检", "数据划分", "存储",
           "标注工时", "推理服务", "系统"]


def log(level: str, module: str, message: str, *, ref_type: str | None = None,
        ref_id: int | None = None, detail: dict | None = None,
        actor: str | None = None) -> None:
    """写一条审计日志（`actor` = 操作人，用于统计工时）。
    日志本身不能影响主流程，所以这里吞掉异常。"""
    try:
        db.execute(
            "INSERT INTO audit_log (ts, level, module, message, ref_type, ref_id, actor,"
            " detail_json) VALUES (?,?,?,?,?,?,?,?)",
            (db.now(), (level or "INFO").upper(), module, message[:500], ref_type, ref_id,
             (actor or None), json.dumps(detail, ensure_ascii=False, default=str) if detail else None))
    except Exception:  # noqa: BLE001
        pass


def query(level: str | None = None, module: str | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM audit_log WHERE 1=1"
    params: list = []
    if level and level != "ALL":
        sql += " AND level=?"
        params.append(level)
    if module and module != "ALL":
        sql += " AND module=?"
        params.append(module)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.query(sql, tuple(params))
    for row in rows:
        row["action"] = row["message"]          # 兼容原日志页面的字段名
        row["time"] = row["ts"]
        if row.get("detail_json"):
            try:
                row["detail"] = json.loads(row["detail_json"])
            except Exception:  # noqa: BLE001
                row["detail"] = None
        row.pop("detail_json", None)
    return rows


def stats() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "total": db.scalar("SELECT COUNT(*) AS c FROM audit_log"),
        "error": db.scalar("SELECT COUNT(*) AS c FROM audit_log WHERE level='ERROR'"),
        "warning": db.scalar("SELECT COUNT(*) AS c FROM audit_log WHERE level='WARNING'"),
        "info": db.scalar("SELECT COUNT(*) AS c FROM audit_log WHERE level='INFO'"),
        "today": db.scalar("SELECT COUNT(*) AS c FROM audit_log WHERE ts LIKE ?", (f"{today}%",)),
    }


def modules() -> list[str]:
    """实际出现过的模块 + 固定清单，去重保序。"""
    seen = [r["module"] for r in db.query("SELECT DISTINCT module FROM audit_log ORDER BY module")
            if r["module"]]
    return seen + [m for m in MODULES if m not in seen]


def backfill_if_empty() -> dict:
    """从已有记录回填历史审计（只在 audit_log 为空时执行一次）。

    平台在引入审计日志之前已经跑过大量真实操作（任务、版本、质检、采集、标注…），
    这些记录仍在各自表里，按原时间戳回填即可，不需要编造。
    """
    if db.scalar("SELECT COUNT(*) AS c FROM audit_log"):
        return {"skipped": True, "reason": "已有审计记录"}
    from . import jobs as jobs_mod

    rows: list[tuple] = []

    def add(ts, level, module, message, ref_type=None, ref_id=None):
        rows.append((ts or db.now(), level, module, message, ref_type, ref_id))

    for j in db.query("SELECT * FROM jobs ORDER BY id"):
        label = jobs_mod.KIND_LABELS.get(j["kind"], j["kind"])
        level = "ERROR" if j["status"] == "failed" else ("WARNING" if j["status"] == "queued" else "INFO")
        add(j["created_at"], level, label,
            f"任务 #{j['id']}（{label}）{j['status']}" + (f"：{(j['error'] or '')[:120]}" if j["error"] else ""),
            "job", j["id"])
    for v in db.query("SELECT * FROM dataset_versions ORDER BY id"):
        add(v["created_at"], "INFO", "版本管理",
            f"数据集 #{v['dataset_id']} 登记 v{v['version']}（{v['kind']}）hash={str(v['content_hash'])[:12]}",
            "dataset", v["dataset_id"])
    for r in db.query("SELECT * FROM qc_runs ORDER BY id"):
        add(r["created_at"], "WARNING" if (r["failed"] or 0) else "INFO", "数据质检",
            f"数据集 #{r['dataset_id']} 质检：{r['passed']} 通过 / {r['failed']} 打回（run{r['id']}）",
            "dataset", r["dataset_id"])
    for s in db.query("SELECT * FROM collect_sessions ORDER BY id"):
        add(s["created_at"], "INFO", "采集会话",
            f"会话 #{s['id']}（{s['source']}）{s['status']}：{s['frames']} 帧 → episode {s['episode_index']}",
            "session", s["id"])
    for e in db.query("SELECT * FROM experiments ORDER BY id"):
        add(e["created_at"], "INFO", "模型训练",
            f"登记实验 #{e['id']} {e['name']}：policy={e['policy']} steps={e['steps']}", "experiment", e["id"])
    for a in db.query("SELECT * FROM annotations ORDER BY id"):
        add(a["updated_at"], "INFO", "数据标注",
            f"数据集 #{a['dataset_id']} ep{a['episode_index']} 标注：成功位={a['success']}",
            "episode", a["episode_index"])
    for rv in db.query("SELECT * FROM annotation_reviews ORDER BY id"):
        add(rv["created_at"], "INFO", "标注抽检",
            f"数据集 #{rv['dataset_id']} ep{rv['episode_index']} 复核（{rv['reviewer']}）成功位={rv['success']}",
            "episode", rv["episode_index"])
    for b in db.query("SELECT * FROM bad_cases ORDER BY id"):
        add(b["created_at"], "WARNING", "模型训练",
            f"bad case（{b['source']}）：{b['reason']}", "episode", b["episode_index"])

    rows.sort(key=lambda r: r[0])
    db.execute_many("INSERT INTO audit_log (ts, level, module, message, ref_type, ref_id)"
                    " VALUES (?,?,?,?,?,?)", rows)
    return {"backfilled": len(rows)}
