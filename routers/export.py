from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

EXPORTS = [
    {"id": 1, "dataset": "H4展厅导览数据集", "format": "LeRobot", "size": "2.4GB", "status": "已完成", "time": "2026-09-20 14:00"},
    {"id": 2, "dataset": "灵巧手抓取数据集", "format": "HDF5", "size": "3.8GB", "status": "已完成", "time": "2026-09-21 09:00"},
    {"id": 3, "dataset": "按电梯按钮数据集", "format": "RLDS", "size": "-", "status": "排队中", "time": "-"},
]

FORMATS = ["LeRobot", "HDF5", "RLDS", "MCAP", "JSON", "CSV"]

@router.get("")
async def export_page(request: Request):
    return templates.TemplateResponse(request, "export.html", {
        "request": request, "exports": EXPORTS, "formats": FORMATS
    })