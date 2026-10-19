from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("")
async def models_page(request: Request):
    """模型管理：真实权重清单与真实推理测试（数据由 /api/inference/* 提供）。"""
    return templates.TemplateResponse(request, "models.html", {"request": request})
