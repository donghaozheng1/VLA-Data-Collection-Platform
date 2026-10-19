from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

RULES = [
    {"id": 1, "name": "帧率检测", "target": "≥ 25fps", "status": "启用"},
    {"id": 2, "name": "Topic 连续性", "target": "无中断", "status": "启用"},
    {"id": 3, "name": "时间同步", "target": "偏差 < 20ms", "status": "启用"},
    {"id": 4, "name": "丢帧检测", "target": "丢帧率 < 1%", "status": "启用"},
    {"id": 5, "name": "动作连续性", "target": "无异常跳变", "status": "启用"},
]

REPORTS = [
    {"id": 1, "dataset": "H4展厅导览数据集", "passed": 312, "failed": 8, "result": "通过", "time": "2026-09-20 11:00"},
    {"id": 2, "dataset": "灵巧手抓取数据集", "passed": 570, "failed": 10, "result": "通过", "time": "2026-09-21 09:30"},
    {"id": 3, "dataset": "按电梯按钮数据集", "passed": 190, "failed": 20, "result": "打回重采", "time": "2026-09-21 10:10"},
]

@router.get("")
async def qc_page(request: Request):
    return templates.TemplateResponse(request, "qc.html", {
        "request": request, "rules": RULES, "reports": REPORTS
    })