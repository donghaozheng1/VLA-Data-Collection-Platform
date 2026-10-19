"""系统管理：真实运行状态（存储用量、数据规模、推理服务、最近的告警）。

原实现是写死的日志列表。
"""
from __future__ import annotations

import os
import platform
import sys

import httpx
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.templating import Jinja2Templates

from core import audit, db, scheduler, storage

router = APIRouter()
api_router = APIRouter()
templates = Jinja2Templates(directory="templates")

INFER_URL = os.environ.get("HKC_INFER_URL", "http://127.0.0.1:8100")


def _dir_stat(sub: str) -> dict:
    directory = db.STORAGE / sub
    if not directory.exists():
        return {"files": 0, "bytes": 0}
    files = [f for f in directory.rglob("*") if f.is_file()]
    return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}


@router.get("")
async def system_page(request: Request):
    storage = {name: _dir_stat(name) for name in
               ("datasets", "exports", "reports", "sessions", "testdata")}
    try:
        infer = {**httpx.get(f"{INFER_URL}/health", timeout=5).json(), "running": True}
    except Exception as exc:  # noqa: BLE001
        infer = {"running": False, "error": f"{type(exc).__name__}: {exc}"}

    counts = {table: db.scalar(f"SELECT COUNT(*) AS c FROM {table}") for table in
              ("datasets", "episodes", "dataset_versions", "collect_sessions", "experiments",
               "evaluations", "bad_cases", "jobs", "artifacts", "annotations",
               "annotation_reviews", "audit_log")}
    return templates.TemplateResponse(request, "system.html", {
        "request": request,
        "platform": {
            "python": sys.version.split()[0],
            "os": f"{platform.system()} {platform.release()}",
            "db_path": str(db.DB_PATH),
            "db_bytes": db.DB_PATH.stat().st_size if db.DB_PATH.exists() else 0,
            "storage_root": str(db.STORAGE),
        },
        "storage": storage,
        "storage_total": sum(v["bytes"] for v in storage.values()),
        "counts": counts,
        "infer": infer,
        "alerts": audit.query(limit=20),
        "stats": audit.stats(),
    })


@api_router.get("/storage/plan")
def storage_plan():
    """存储清理计划（只算账，不删）。"""
    return storage.scan()


@api_router.post("/storage/cleanup")
def storage_cleanup(payload: dict = Body(default={})):
    """执行清理：kinds 限定类别、paths 限定路径；dry_run=True 只回显。"""
    return storage.apply(kinds=payload.get("kinds"), paths=payload.get("paths"),
                         dry_run=bool(payload.get("dry_run")))


@api_router.get("/storage/policy")
def storage_policy_get():
    return {**storage.policy(), "scheduler": scheduler.status()}


@api_router.post("/storage/policy")
def storage_policy_set(payload: dict = Body(default={})):
    try:
        return storage.set_policy(enabled=payload.get("enabled"),
                                  threshold_mb=payload.get("threshold_mb"),
                                  kinds=payload.get("kinds"),
                                  interval_s=payload.get("interval_s"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api_router.post("/storage/check")
def storage_check_now():
    """立刻跑一次策略检查（调度线程平时也在跑）。"""
    return scheduler.tick()
