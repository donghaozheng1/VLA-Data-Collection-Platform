from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 统一对外接口：按机器人与场景动态路由到对应模型
ENDPOINTS = [
    {
        "id": 1,
        "path": "/api/v1/detect",
        "method": "POST",
        "description": "目标检测推理",
        "robot": "人形-A1",
        "model": "YOLOv8m",
        "status": "已启用",
    },
    {
        "id": 2,
        "path": "/api/v1/segment",
        "method": "POST",
        "description": "实例分割推理",
        "robot": "人形-A2",
        "model": "YOLOv8x",
        "status": "已启用",
    },
    {
        "id": 3,
        "path": "/api/v1/grasp/pose",
        "method": "POST",
        "description": "抓取位姿估计",
        "robot": "人形-A2",
        "model": "GraspNet",
        "status": "已启用",
    },
    {
        "id": 4,
        "path": "/api/v1/scene/config",
        "method": "GET",
        "description": "获取场景与机器人配置",
        "robot": "全部",
        "model": "-",
        "status": "已启用",
    },
    {
        "id": 5,
        "path": "/api/v1/vla/act",
        "method": "POST",
        "description": "VLA 动作推理（语言指令 → 动作序列）",
        "robot": "AMR-巡检者",
        "model": "VLA-Base",
        "status": "测试中",
    },
]


@router.get("")
async def api_page(request: Request):
    return templates.TemplateResponse(request, "api.html", {
        "request": request, "endpoints": ENDPOINTS
    })
