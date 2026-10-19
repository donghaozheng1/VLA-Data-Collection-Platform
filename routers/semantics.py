from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 统一语义层：8 个核心字段
FIELDS = [
    {
        "id": 1,
        "name": "任务描述",
        "desc": "自然语言描述任务目标",
        "example": "“将红色连接器插入左侧插槽”",
    },
    {
        "id": 2,
        "name": "图像 / 视频",
        "desc": "场景视觉观测",
        "example": "RGB帧序列、深度图",
    },
    {
        "id": 3,
        "name": "末端3D轨迹",
        "desc": "末端执行器在笛卡尔空间的6-DoF位姿序列。跨本体迁移的核心",
        "example": "位置(x,y,z) + 姿态(四元数)",
    },
    {
        "id": 4,
        "name": "动作阶段",
        "desc": "任务的关键阶段切分",
        "example": "接近 → 抓取 → 移动 → 插入",
    },
    {
        "id": 5,
        "name": "成功 / 失败",
        "desc": "任务最终结果标注",
        "example": "成功 / 失败 / 待确认",
    },
    {
        "id": 6,
        "name": "时间戳",
        "desc": "统一时间基准",
        "example": "各模态对齐后的时间戳",
    },
    {
        "id": 7,
        "name": "坐标系定义",
        "desc": "明确坐标原点与方向",
        "example": "机器人基座坐标系 / 世界坐标系",
    },
    {
        "id": 8,
        "name": "物体信息",
        "desc": "操作对象的类别、位置、状态。可选，可从图像中提取，也可单独标注；如果标注了更精确",
        "example": "“连接器，位于工作台左上角”",
    },
]

# Episode 文件结构
EPISODE_FILES = [
    {"name": "metadata.json", "desc": "任务描述、成功/失败、动作阶段、坐标系定义"},
    {"name": "timestamps.parquet", "desc": "统一时间戳（对齐后）"},
    {"name": "trajectory.parquet", "desc": "末端3D轨迹（6-DoF位姿序列）"},
    {"name": "camera_front.mp4", "desc": "前端相机视频流"},
    {"name": "camera_wrist.mp4", "desc": "腕部相机视频流"},
    {"name": "annotations.json", "desc": "物体信息、阶段切分、标注细节"},
]

@router.get("")
async def semantics_page(request: Request):
    return templates.TemplateResponse(request, "semantics.html", {
        "request": request,
        "fields": FIELDS,
        "episode_files": EPISODE_FILES,
    })