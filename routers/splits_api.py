"""数据划分接口（/api/splits/*）。导出仍走既有的 /api/jobs/export（mode=split）。"""
from fastapi import APIRouter, Body, HTTPException

from core import db, jobs, splits

api_router = APIRouter()


@api_router.get("/splits/overview")
def overview_all():
    return splits.all_overviews()


@api_router.get("/splits/{dataset_id}")
def overview_one(dataset_id: int):
    return splits.overview(dataset_id)


@api_router.post("/splits/assign")
def do_assign(payload: dict = Body(default={})):
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(400, "缺少 dataset_id")
    try:
        result = splits.assign(int(dataset_id), ratios=payload.get("ratios"),
                               salt=payload.get("salt") or "v1",
                               mode=payload.get("mode") or "random")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result.pop("assignment", None)          # 名单走 GET /api/splits/{id}
    return result


@api_router.delete("/splits/{dataset_id}")
def clear(dataset_id: int):
    db.execute("DELETE FROM dataset_splits WHERE dataset_id=?", (dataset_id,))
    return {"dataset_id": dataset_id, "cleared": True}


@api_router.post("/splits/export")
def export_split(payload: dict = Body(default={})):
    """把某个划分导出成独立数据集（异步任务，前端轮询）。"""
    dataset_id, split = payload.get("dataset_id"), payload.get("split")
    if not dataset_id or not split:
        raise HTTPException(400, "需要 dataset_id 与 split")
    indices = splits.episodes_of(int(dataset_id), split)
    if not indices:
        raise HTTPException(400, f"该数据集还没有 '{split}' 划分")
    row = db.query_one("SELECT name FROM datasets WHERE id=?", (int(dataset_id),))
    job_id = jobs.submit("export", {
        "dataset_id": int(dataset_id), "mode": "split", "split": split,
        "name": payload.get("name") or f"{row['name']}_{split}",
    }, target_type="dataset", target_id=int(dataset_id))
    return {"job_id": job_id, "episodes": len(indices), "split": split}
