from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from core import db
from routers.training_api import DEFAULT_OUTPUT_ROOT

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("")
async def training_page(request: Request):
    """真实训练与离线评估：实验台账 + 指标 + 评估 + bad case 回灌。"""
    return templates.TemplateResponse(request, "training.html", {
        "request": request,
        "default_root": DEFAULT_OUTPUT_ROOT,
        "stats": {
            "experiments": db.scalar("SELECT COUNT(*) AS c FROM experiments"),
            "metrics": db.scalar("SELECT COUNT(*) AS c FROM experiment_metrics"),
            "evaluations": db.scalar("SELECT COUNT(*) AS c FROM evaluations"),
            "bad_open": db.scalar("SELECT COUNT(*) AS c FROM bad_cases WHERE status='open'"),
        },
        "datasets": db.query("SELECT id, name, total_episodes FROM datasets ORDER BY id"),
    })
