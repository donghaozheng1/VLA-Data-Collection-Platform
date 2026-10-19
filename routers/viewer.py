"""数据集浏览 —— 逐帧播放。

**为什么不用 `<video>` 标签**：实测平台里的视频编码是 `mp4v`(MPEG-4 Part 2) 和 `av01`，
其中 mp4v **浏览器一律不支持**（17 个有视频的数据集里 11 个是 mp4v，包括全部 zhiyan 真机数据）。
转码要额外磁盘和一次全量处理，而实测"按需抽帧"（打开 + seek + 解码 + 缩放 + JPEG 编码）
在 1280x720 上平均 **29ms 且与帧位置无关** —— 20fps 播放需要 50ms/帧，所以逐帧播放
就能做到实时，**零转码、零缓存、不占额外磁盘，而且对所有编码都有效**。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

from core import db

router = APIRouter()
#: 页面与接口分开注册：页面在 /viewer，接口在 /api/viewer/*
#: （照平台上 reviews / collect / training 的惯例，避免页面路径和接口路径混在一起）
api_router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("")
async def viewer_page(request: Request):
    return templates.TemplateResponse(request, "viewer.html", {
        "request": request,
        "datasets": db.query(
            "SELECT id,name,total_episodes,total_frames,fps,codebase_version,codec"
            " FROM datasets ORDER BY id"),
    })


@api_router.get("/viewer/episodes/{dataset_id}")
def viewer_episodes(dataset_id: int):
    """播放器需要的每 episode 摘要：帧数、任务、问题、建议。

    与「挑出要排除的 episode」共用 `qc_checks.episode_problem_summary`，
    所以两边对"数据集级问题"的处理完全一致（否则这里每一条都会被同一个
    数据集级告警染成"待定"，看不出真正的问题）。
    """
    from core import qc_checks

    data = qc_checks.episode_problem_summary(dataset_id)
    if data.get("error"):
        raise HTTPException(404, data["error"])
    return {"dataset": data["dataset"], "run": data.get("run"),
            "episodes": data["episodes"], "dataset_level": data.get("dataset_level") or []}
