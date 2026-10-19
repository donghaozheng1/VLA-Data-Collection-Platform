from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from core import db, pipeline

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("")
async def collect_page(request: Request):
    """采集接入：扫描磁盘上的真实数据集并登记入库。"""
    return templates.TemplateResponse(request, "collect.html", {
        "request": request,
        "roots": pipeline.DEFAULT_SCAN_ROOTS,
        "datasets": db.query("SELECT * FROM datasets ORDER BY id DESC"),
        "recent_artifacts": db.query("SELECT * FROM artifacts ORDER BY id DESC LIMIT 8"),
    })
