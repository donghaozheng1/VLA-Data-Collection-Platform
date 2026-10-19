from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

TASKS = [
    {"id": 1, "dataset": "H4展厅导览数据集", "rule": "时间对齐 + 去重", "status": "已完成", "before": 320, "after": 312, "time": "2026-09-20 10:20"},
    {"id": 2, "dataset": "灵巧手抓取数据集", "rule": "丢帧剔除 + 格式标准化", "status": "运行中", "before": 580, "after": "-", "time": "2026-09-21 09:00"},
    {"id": 3, "dataset": "按电梯按钮数据集", "rule": "坏帧剔除 + 时间同步", "status": "排队中", "before": 210, "after": "-", "time": "-"},
]

@router.get("")
async def preprocess_page(request: Request):
    return templates.TemplateResponse(request, "preprocess.html", {
        "request": request, "tasks": TASKS
    })