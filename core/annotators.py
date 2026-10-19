"""标注账号与工时。

现状问题：`annotations.annotator` / `annotation_reviews.reviewer` 只是自由字符串 ——
算不出"谁的标注质量如何""这条数据花了多久"，也没法按人分工。

这里的做法**不假装做了一套认证系统**（平台登录仍是 demo 级），只解决两件真实的事：

1. **花名册**：`annotators` 表登记人名与角色（标注员 / 复核员 / 管理员）。已有的自由字符串
   会被自动登记进花名册（不打断既有流程），并在统计里标出"未登记/已停用"。
2. **工时**：`work_sessions` 记录上工/收工区间；区间内**真实做了多少操作**从审计日志里数
   （`audit_log.actor`），不靠人填。没点「收工」的会话会被单独标出来，不计入总工时 ——
   宁可显示"有一个未结束的会话"，也不虚报时长。
"""
from __future__ import annotations

from . import audit, db

ROLES = ("标注员", "复核员", "管理员")


def sync_from_records() -> dict:
    """把既有标注/复核记录里出现过的名字补登进花名册（幂等）。

    花名册是引入账号之后才有的，历史记录里的 `annotator` / `reviewer` 都是自由字符串；
    不补登的话，"谁的标注质量如何"这类统计一开始就是空的。
    """
    names: set[str] = set()
    for table, column in (("annotations", "annotator"),
                          ("annotation_reviews", "reviewer"),
                          ("spatial_annotations", "annotator")):
        for row in db.query(f"SELECT DISTINCT {column} AS name FROM {table}"
                            f" WHERE {column} IS NOT NULL AND {column} != ''"):
            names.add(str(row["name"]).strip())
    added = []
    for name in sorted(names):
        if not db.query_one("SELECT id FROM annotators WHERE name=?", (name,)):
            register(name, "标注员", note="由既有记录补登")
            added.append(name)
    return {"discovered": len(names), "added": added}


def roster(active_only: bool = False) -> list[dict]:
    sync_from_records()
    sql = "SELECT * FROM annotators"
    if active_only:
        sql += " WHERE active=1"
    sql += " ORDER BY active DESC, name"
    rows = db.query(sql)
    for row in rows:
        row["open_sessions"] = db.scalar(
            "SELECT COUNT(*) AS c FROM work_sessions WHERE annotator=? AND ended_at IS NULL",
            (row["name"],))
        row["annotations"] = db.scalar(
            "SELECT COUNT(*) AS c FROM annotations WHERE annotator=?", (row["name"],))
        row["reviews"] = db.scalar(
            "SELECT COUNT(*) AS c FROM annotation_reviews WHERE reviewer=?", (row["name"],))
        row["spatial"] = db.scalar(
            "SELECT COUNT(*) AS c FROM spatial_annotations WHERE annotator=?", (row["name"],))
    return rows


def register(name: str, role: str = "标注员", note: str | None = None,
             active: bool = True) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("人名不能为空")
    if role not in ROLES:
        raise ValueError(f"角色必须是 {ROLES} 之一")
    db.execute("INSERT INTO annotators (name, role, active, note, created_at)"
               " VALUES (?,?,?,?,?)"
               " ON CONFLICT(name) DO UPDATE SET role=excluded.role, active=excluded.active,"
               " note=COALESCE(excluded.note, annotators.note)",
               (name, role, 1 if active else 0, note, db.now()))
    audit.log("INFO", "标注工时", f"登记标注员 {name}（{role}）", actor=name)
    return db.query_one("SELECT * FROM annotators WHERE name=?", (name,))


def ensure_known(name: str) -> dict:
    """自由字符串自动进花名册（不打断既有流程），返回花名册记录。"""
    name = (name or "admin").strip() or "admin"
    row = db.query_one("SELECT * FROM annotators WHERE name=?", (name,))
    if row:
        return row
    return register(name, "标注员", note="由标注记录自动登记")


def start_work(annotator: str, dataset_id: int | None = None,
               note: str | None = None) -> dict:
    row = ensure_known(annotator)
    if not row["active"]:
        raise ValueError(f"{row['name']} 已停用，不能上工")
    open_row = db.query_one("SELECT * FROM work_sessions WHERE annotator=? AND ended_at IS NULL",
                            (row["name"],))
    if open_row:
        raise ValueError(f"{row['name']} 已有一个未结束的会话（#{open_row['id']}，"
                         f"开始于 {open_row['started_at']}），先收工再上工")
    session_id = db.execute(
        "INSERT INTO work_sessions (annotator, dataset_id, started_at, note, created_at)"
        " VALUES (?,?,?,?,?)", (row["name"], dataset_id, db.now(), note, db.now()))
    audit.log("INFO", "标注工时", f"{row['name']} 上工", actor=row["name"],
              detail={"session_id": session_id, "dataset_id": dataset_id})
    return db.query_one("SELECT * FROM work_sessions WHERE id=?", (session_id,))


def stop_work(annotator: str, note: str | None = None) -> dict:
    name = (annotator or "").strip()
    open_row = db.query_one("SELECT * FROM work_sessions WHERE annotator=? AND ended_at IS NULL",
                            (name,))
    if not open_row:
        raise ValueError(f"{name} 没有进行中的会话")
    db.execute("UPDATE work_sessions SET ended_at=?, note=COALESCE(?, note) WHERE id=?",
               (db.now(), note, open_row["id"]))
    session = db.query_one("SELECT * FROM work_sessions WHERE id=?", (open_row["id"],))
    counts = _counts_in_window(name, session["started_at"], session["ended_at"])
    audit.log("INFO", "标注工时",
              f"{name} 收工：{_duration_minutes(session):.1f} 分钟，"
              f"区间内 {sum(counts.values())} 次标注操作", actor=name,
              detail={"session_id": session["id"], "counts": counts})
    return {**session, "counts": counts, "minutes": round(_duration_minutes(session), 2)}


def _duration_minutes(session: dict) -> float:
    from datetime import datetime

    try:
        start = datetime.fromisoformat(str(session["started_at"]))
        end = datetime.fromisoformat(str(session["ended_at"] or db.now()))
    except ValueError:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 60.0)


def _counts_in_window(annotator: str, start: str, end: str | None) -> dict:
    """区间内该人的真实操作数（从审计日志数，不靠人填）。"""
    end = end or db.now()
    rows = db.query(
        "SELECT module, COUNT(*) AS c FROM audit_log WHERE actor=? AND ts >= ? AND ts <= ?"
        " AND module='数据标注' GROUP BY module", (annotator, start, end))
    return {row["module"]: row["c"] for row in rows}


def open_sessions() -> list[dict]:
    return db.query("SELECT * FROM work_sessions WHERE ended_at IS NULL ORDER BY id")


def stats(days: int = 7, annotator: str | None = None) -> dict:
    """按人汇总工时与产出。**未结束的会话不计入总工时**，单独列出来。"""
    from datetime import datetime, timedelta

    since = (datetime.now() - timedelta(days=max(1, days))).strftime("%Y-%m-%d %H:%M:%S")
    sql = "SELECT * FROM work_sessions WHERE started_at >= ?"
    params: list = [since]
    if annotator:
        sql += " AND annotator=?"
        params.append(annotator)
    sessions = db.query(sql + " ORDER BY id", tuple(params))

    people: dict[str, dict] = {}
    for session in sessions:
        person = people.setdefault(session["annotator"], {
            "annotator": session["annotator"], "sessions": 0, "open_sessions": 0,
            "minutes": 0.0, "operations": 0, "datasets": set(),
        })
        if session["ended_at"]:
            person["sessions"] += 1
            person["minutes"] += _duration_minutes(session)
            counts = _counts_in_window(session["annotator"], session["started_at"],
                                       session["ended_at"])
            person["operations"] += sum(counts.values())
        else:
            person["open_sessions"] += 1
        if session["dataset_id"]:
            person["datasets"].add(session["dataset_id"])

    roster_map = {r["name"]: r for r in roster()}
    out = []
    for name, person in people.items():
        row = roster_map.get(name) or {}
        minutes = round(person["minutes"], 2)
        out.append({
            "annotator": name, "role": row.get("role"), "registered": bool(row),
            "active": bool(row.get("active", 1)),
            "sessions": person["sessions"], "open_sessions": person["open_sessions"],
            "minutes": minutes, "hours": round(minutes / 60, 2),
            "operations": person["operations"],
            "per_hour": round(person["operations"] / (minutes / 60), 1) if minutes >= 1 else None,
            "datasets": sorted(person["datasets"]),
            "annotations_total": row.get("annotations", 0),
            "reviews_total": row.get("reviews", 0),
            "spatial_total": row.get("spatial", 0),
        })
    out.sort(key=lambda x: -x["minutes"])
    return {
        "days": days,
        "people": out,
        "totals": {
            "people": len(out),
            "minutes": round(sum(p["minutes"] for p in out), 2),
            "operations": sum(p["operations"] for p in out),
            "open_sessions": sum(p["open_sessions"] for p in out),
        },
        "note": "工时按上工/收工区间统计，操作数从审计日志数（不靠人填）；"
                "未点「收工」的会话不计入工时，单独标出。",
    }
