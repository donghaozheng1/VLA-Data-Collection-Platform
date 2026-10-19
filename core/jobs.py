"""后台任务引擎：耗时操作放线程里跑，进度写进 SQLite，前端轮询 jobs 表。

用法：
    from core import jobs

    @jobs.handler("qc")
    def run_qc(job, payload):
        job.progress(1, 10, "开始")
        ...
        return {"ok": True}          # 写入 result_json

    job_id = jobs.submit("qc", {"dataset_id": 1}, target_type="dataset", target_id=1)
"""
from __future__ import annotations

import json
import threading
import traceback

from . import audit, db

HANDLERS: dict = {}
KIND_LABELS = {
    "ingest": "数据接入",
    "convert": "原始轨迹转换",
    "preprocess": "数据预处理",
    "repair": "修正类预处理",
    "qc": "数据质检",
    "export": "数据集导出",
}


class JobContext:
    """任务句柄：handler 用它汇报进度。"""

    def __init__(self, job_id: int):
        self.id = job_id

    def progress(self, done: int, total: int = 0, message: str = "") -> None:
        pct = int(done / total * 100) if total else min(99, done)
        db.execute(
            "UPDATE jobs SET progress=?, total=?, message=?, status='running' WHERE id=?",
            (max(0, min(100, pct)), total, message[:500], self.id),
        )

    def log(self, message: str) -> None:
        db.execute("UPDATE jobs SET message=? WHERE id=?", (message[:500], self.id))


def handler(kind: str):
    def decorator(fn):
        HANDLERS[kind] = fn
        return fn
    return decorator


def submit(kind: str, payload: dict | None = None, target_type: str | None = None,
           target_id: int | None = None, wait: bool = False) -> int:
    """建一个任务；默认丢到后台线程执行，wait=True 则同步执行（测试用）。"""
    job_id = db.execute(
        "INSERT INTO jobs (kind, target_type, target_id, payload_json, status, progress, total,"
        " message, created_at) VALUES (?,?,?,?, 'queued', 0, 0, '已排队', ?)",
        (kind, target_type, target_id, json.dumps(payload or {}, ensure_ascii=False), db.now()),
    )
    audit.log("INFO", KIND_LABELS.get(kind, kind),
              f"创建任务 #{job_id}（{KIND_LABELS.get(kind, kind)}）",
              ref_type=target_type, ref_id=target_id)
    if wait:
        _run(job_id, kind, payload or {})
    else:
        threading.Thread(target=_run, args=(job_id, kind, payload or {}), daemon=True).start()
    return job_id


def _run(job_id: int, kind: str, payload: dict) -> None:
    fn = HANDLERS.get(kind)
    db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (db.now(), job_id))
    if fn is None:
        db.execute(
            "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
            (f"没有注册 {kind} 的处理器", db.now(), job_id),
        )
        return
    try:
        result = fn(JobContext(job_id), payload)
        db.execute(
            "UPDATE jobs SET status='succeeded', progress=100, message='完成', result_json=?,"
            " finished_at=? WHERE id=?",
            (json.dumps(result or {}, ensure_ascii=False, default=str), db.now(), job_id),
        )
        audit.log("INFO", KIND_LABELS.get(kind, kind),
                  f"任务 #{job_id}（{KIND_LABELS.get(kind, kind)}）完成", ref_id=job_id,
                  detail=result if isinstance(result, dict) else None)
    except Exception as exc:  # noqa: BLE001 - 任务失败要落库给前端看
        db.execute(
            "UPDATE jobs SET status='failed', error=?, message='失败', finished_at=? WHERE id=?",
            (f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}", db.now(), job_id),
        )
        audit.log("ERROR", KIND_LABELS.get(kind, kind),
                  f"任务 #{job_id}（{KIND_LABELS.get(kind, kind)}）失败：{type(exc).__name__}: {exc}",
                  ref_id=job_id)


def get(job_id: int):
    job = db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if job:
        job["kind_label"] = KIND_LABELS.get(job["kind"], job["kind"])
    return job


def recent(limit: int = 20) -> list[dict]:
    rows = db.query("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
    for row in rows:
        row["kind_label"] = KIND_LABELS.get(row["kind"], row["kind"])
    return rows


def running_count() -> int:
    return db.scalar("SELECT COUNT(*) AS c FROM jobs WHERE status IN ('queued','running')")
