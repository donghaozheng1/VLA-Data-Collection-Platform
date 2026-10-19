from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

DATASETS = [
    {"id": 1, "name": "H4展厅导览数据集", "robot": "人形-A1", "episodes": 320, "hours": 12.5, "format": "MCAP", "status": "已入库", "qc": "通过"},
    {"id": 2, "name": "灵巧手抓取数据集", "robot": "人形-A2", "episodes": 580, "hours": 18.2, "format": "MCAP", "status": "已入库", "qc": "通过"},
    {"id": 3, "name": "按电梯按钮数据集", "robot": "AMR-巡检者", "episodes": 210, "hours": 6.8, "format": "MCAP", "status": "质检中", "qc": "待检"},
    {"id": 4, "name": "触摸屏操作数据集", "robot": "人形-A1", "episodes": 140, "hours": 4.2, "format": "MCAP", "status": "采集中", "qc": "-"},
]

@router.get("")
async def data_page(request: Request):
    return templates.TemplateResponse(request, "data.html", {
        "request": request, "datasets": DATASETS
    })