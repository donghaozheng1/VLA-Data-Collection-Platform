from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("")
async def dashboard_page(request: Request):
    stats = {
        "datasets": 12,
        "raw_hours": 86.5,
        "annotated": 8,
        "qc_passed": 6,
        "training": 3,
    }
    recent = [
        {"name": "H4展厅导览采集", "robot": "人形-A1", "status": "采集中", "progress": 72},
        {"name": "灵巧手抓取采集", "robot": "人形-A2", "status": "质检中", "progress": 100},
        {"name": "按电梯按钮采集", "robot": "AMR-巡检者", "status": "标注中", "progress": 45},
        {"name": "触摸屏操作采集", "robot": "人形-A1", "status": "已完成", "progress": 100},
    ]
    return templates.TemplateResponse(request, "dashboard.html", {
        "request": request, "stats": stats, "recent": recent
    })