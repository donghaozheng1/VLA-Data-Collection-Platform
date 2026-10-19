"""标注质量抽检页 + JSON 接口。

- 页面 `/reviews`：生成抽检批次 → 盲标复核 → 看一致性指标与争议裁决
- 接口 `/api/reviews/*`：抽检、待复核队列、提交复核、一致性、争议裁决、批次历史
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.templating import Jinja2Templates

from core import db, reviews

router = APIRouter()
api_router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("")
async def reviews_page(request: Request):
    return templates.TemplateResponse(request, "reviews.html", {
        "request": request,
        "datasets": db.query(
            "SELECT d.id, d.name, d.total_episodes,"
            " (SELECT COUNT(*) FROM annotations a WHERE a.dataset_id=d.id) AS annotated,"
            " (SELECT COUNT(*) FROM annotation_reviews r WHERE r.dataset_id=d.id) AS reviewed"
            " FROM datasets d ORDER BY d.id"),
    })


# --------------------------------------------------------------------------- 抽检

@api_router.get("/reviews/summary")
def summary():
    """每个数据集的标注/复核概况，供页面下拉里显示。"""
    return db.query(
        "SELECT d.id, d.name, d.total_episodes,"
        " (SELECT COUNT(*) FROM annotations a WHERE a.dataset_id=d.id) AS annotated,"
        " (SELECT COUNT(*) FROM annotations a WHERE a.dataset_id=d.id"
        "   AND a.success IS NOT NULL) AS success_labeled,"
        " (SELECT COUNT(*) FROM annotation_reviews r WHERE r.dataset_id=d.id) AS reviewed,"
        " (SELECT COUNT(*) FROM review_batches b WHERE b.dataset_id=d.id) AS batches"
        " FROM datasets d ORDER BY d.id")


@api_router.post("/reviews/sample")
def do_sample(payload: dict = Body(default={})):
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(400, "缺少 dataset_id")
    try:
        return reviews.sample(int(dataset_id),
                              ratio=payload.get("ratio"),
                              count=payload.get("count"),
                              salt=str(payload.get("salt") or ""),
                              note=payload.get("note"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api_router.get("/reviews/batches")
def list_batches(dataset_id: int | None = None, limit: int = 20):
    return reviews.batches(dataset_id, limit)


# --------------------------------------------------------------------------- 待复核队列

@api_router.get("/reviews/queue")
def queue(dataset_id: int, reviewer: str = "annotator_B", batch_id: int | None = None,
          only_pending: bool = False, limit: int = 200):
    """待复核清单。默认**盲标**：只给原始指令与视频，不给主标注结果。"""
    episode_filter = ""
    params: list = [reviewer, reviewer, dataset_id]
    if batch_id:
        batch = db.query_one("SELECT * FROM review_batches WHERE id=?", (batch_id,))
        if not batch:
            raise HTTPException(404, "批次不存在")
        episodes = json.loads(batch.get("episodes_json") or "[]")
        if not episodes:
            return {"batch_id": batch_id, "reviewer": reviewer, "items": []}
        episode_filter = f" AND a.episode_index IN ({','.join('?' * len(episodes))})"
        params.extend(episodes)

    sql = f"""SELECT a.episode_index,
                     e.task_text AS source_task, e.data_rows, e.duration_s, e.success_count,
                     (SELECT COUNT(*) FROM annotation_reviews r2
                       WHERE r2.dataset_id=a.dataset_id AND r2.episode_index=a.episode_index
                         AND r2.reviewer=?) AS reviewer_count,
                     (SELECT COUNT(*) FROM annotation_reviews r3
                       WHERE r3.dataset_id=a.dataset_id AND r3.episode_index=a.episode_index) AS review_total,
                     r.task_text AS review_task, r.success AS review_success, r.note AS review_note,
                     r.resolved AS review_resolved
              FROM annotations a
              LEFT JOIN episodes e
                ON e.dataset_id=a.dataset_id AND e.episode_index=a.episode_index
              LEFT JOIN annotation_reviews r
                ON r.dataset_id=a.dataset_id AND r.episode_index=a.episode_index AND r.reviewer=?
              WHERE a.dataset_id=?{episode_filter}
              ORDER BY a.episode_index LIMIT ?"""
    params.append(limit)
    items = db.query(sql, tuple(params))
    if only_pending:
        items = [x for x in items if not x["reviewer_count"]]
    return {"dataset_id": dataset_id, "reviewer": reviewer, "batch_id": batch_id,
            "blind": True, "items": items}


@api_router.put("/reviews/{dataset_id}/{episode_index}")
def put_review(dataset_id: int, episode_index: int, payload: dict = Body(default={})):
    try:
        return reviews.put_review(dataset_id, episode_index,
                                  reviewer=payload.get("reviewer") or "annotator_B",
                                  task_text=payload.get("task_text"),
                                  success=payload.get("success"),
                                  note=payload.get("note"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------- 一致性与争议

@api_router.get("/reviews/agreement")
def get_agreement(dataset_id: int, reviewer: str | None = None):
    return reviews.agreement(dataset_id, reviewer)


@api_router.post("/reviews/{dataset_id}/{episode_index}/resolve")
def resolve(dataset_id: int, episode_index: int, payload: dict = Body(default={})):
    try:
        return reviews.resolve(dataset_id, episode_index,
                               reviewer=payload.get("reviewer") or "annotator_B",
                               decision=payload.get("decision") or "review",
                               task_text=payload.get("task_text"),
                               success=payload.get("success"),
                               note=payload.get("note"),
                               resolver=payload.get("resolver") or "admin")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
