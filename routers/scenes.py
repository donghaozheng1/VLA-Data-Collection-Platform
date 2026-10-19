"""场景/任务页与接口。"""
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.templating import Jinja2Templates

from core import audit, db, scenes

router = APIRouter()
api_router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("")
async def scenes_page(request: Request):
    return templates.TemplateResponse(request, "scenes.html", {
        "request": request,
        "overview": scenes.overview(),
    })


@api_router.get("/scenes/overview")
def scenes_overview():
    return scenes.overview()


@api_router.get("/scenes/aliases")
def list_aliases():
    return scenes.aliases()


@api_router.post("/scenes/alias")
def create_alias(payload: dict = Body(default={})):
    alias, canonical = payload.get("alias"), payload.get("canonical")
    if not alias or not canonical:
        raise HTTPException(400, "需要 alias 与 canonical")
    try:
        result = scenes.set_alias(alias, canonical, payload.get("note"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    audit.log("INFO", "数据标注", f"任务归一：{alias!r} -> {canonical!r}",
              detail={"alias": alias, "canonical": canonical})
    return result


@api_router.get("/scenes/suggestions")
def suggestions():
    return scenes.suggest_duplicates()


@api_router.post("/scenes/merge")
def merge_tasks(payload: dict = Body(default={})):
    tasks, canonical = payload.get("tasks") or [], payload.get("canonical")
    if not canonical or len(tasks) < 2:
        raise HTTPException(400, "需要至少两个任务名与 canonical")
    result = scenes.merge(tasks, canonical, payload.get("note"))
    audit.log("INFO", "数据标注",
              f"任务归一：{'、'.join(result['aliased'])} -> {canonical}",
              detail=result)
    return result


@api_router.delete("/scenes/alias")
def remove_alias(alias: str):
    audit.log("WARNING", "数据标注", f"取消任务归一：{alias!r}")
    return scenes.delete_alias(alias)
