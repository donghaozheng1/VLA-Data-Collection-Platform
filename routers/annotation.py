from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

TASKS = [
    {"id": 1, "dataset": "H4展厅导览数据集", "type": "语言指令 + 动作段落", "annotator": "user_001", "status": "已完成", "progress": 100},
    {"id": 2, "dataset": "灵巧手抓取数据集", "type": "抓取目标 + 动作段落", "annotator": "user_002", "status": "标注中", "progress": 65},
    {"id": 3, "dataset": "按电梯按钮数据集", "type": "按钮状态 + 动作段落", "annotator": "user_003", "status": "待审核", "progress": 100},
    {"id": 4, "dataset": "触摸屏操作数据集", "type": "点击目标 + 动作段落", "annotator": "-", "status": "待分配", "progress": 0},
]

@router.get("")
async def annotation_page(request: Request):
    return templates.TemplateResponse(request, "annotation.html", {
        "request": request, "tasks": TASKS
    })