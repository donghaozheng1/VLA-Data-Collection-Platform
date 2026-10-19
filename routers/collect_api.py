"""实时采集会话接口（/api/collect/*）。

客户端（真机或回放）调这些接口：建会话 → 批量上报帧 → 提交（落成真实 LeRobot episode）。
提交后会顺带把数据集接入平台（登记版本与血缘 kind=collect）。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Body, HTTPException

from core import audit, db, jobs, session
from core import pipeline  # noqa: F401  确保 ingest 处理器已注册

api_router = APIRouter()


@api_router.post("/collect/session")
def create_session(payload: dict = Body(default={})):
    if not payload.get("dataset_name"):
        raise HTTPException(400, "缺少 dataset_name")
    if not payload.get("task_text"):
        raise HTTPException(400, "缺少 task_text（语言指令，采集时必须给定）")
    try:
        return session.create_session(
            dataset_name=payload["dataset_name"],
            task_text=payload["task_text"],
            fps=float(payload.get("fps") or 10.0),
            robot_type=payload.get("robot_type") or "realman-7dof",
            state_dim=int(payload.get("state_dim") or 7),
            joint_names=payload.get("joint_names"),
            image_keys=payload.get("image_keys") or [],
            source=payload.get("source") or "client",
            note=payload.get("note"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api_router.get("/collect/sessions")
def list_sessions(limit: int = 20, status: str | None = None):
    return session.sessions(limit, status)


@api_router.get("/collect/session/{session_id}")
def get_session(session_id: int):
    try:
        return session.status(session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@api_router.post("/collect/session/{session_id}/frames")
def push_frames(session_id: int, payload: dict = Body(default={})):
    frames = payload.get("frames") or []
    if not frames:
        raise HTTPException(400, "frames 为空")
    try:
        return session.append_frames(session_id, frames)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api_router.post("/collect/session/{session_id}/commit")
def commit(session_id: int, payload: dict = Body(default={})):
    """提交会话：写成一个真实 episode，并把数据集接入平台（版本 kind=collect）。"""
    try:
        result = session.commit_session(
            session_id,
            success=payload.get("success"),
            note=payload.get("note"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    dataset = db.query_one("SELECT id FROM datasets WHERE path=?", (result["dataset_path"],))
    origin = None if dataset else {"kind": "collect", "derived_by": None,
                                   "note": f"采集会话 #{session_id}"}
    job_id = jobs.submit("ingest",
                         {"path": result["dataset_path"], "name": result["dataset_name"],
                          "with_video": bool(result.get("video_frames")), "origin": origin,
                          "incremental": True, "episode_indices": [result["episode_index"]]},
                         target_type="session", target_id=session_id, wait=True)
    job = jobs.get(job_id)
    dataset = db.query_one("SELECT * FROM datasets WHERE path=?", (result["dataset_path"],))
    audit.log("INFO", "采集会话",
              f"会话 #{session_id} 提交为 episode {result['episode_index']}：{result['rows']} 行 / "
              f"{result['video_frames']} 视频帧，步长偏差 {result['step_dev_max']}",
              ref_type="session", ref_id=session_id,
              detail={"dataset": result["dataset_name"], "episode_index": result["episode_index"],
                      "rows": result["rows"], "step_dev_max": result["step_dev_max"]})
    return {**result,
            "platform_dataset_id": dataset["id"] if dataset else None,
            "platform_version": dataset["version"] if dataset else None,
            "platform_content_hash": dataset["content_hash"] if dataset else None,
            "ingest_job": job_id, "ingest_status": job["status"] if job else None,
            "ingest_error": (job["error"] if job else None)}


@api_router.post("/collect/session/{session_id}/abort")
def abort(session_id: int):
    try:
        return session.abort_session(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
