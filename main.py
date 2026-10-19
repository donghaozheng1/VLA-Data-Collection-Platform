from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core import db, scheduler
from core import pipeline  # noqa: F401  导入即注册 ingest/qc/preprocess/export 任务处理器
from routers import sim_api
from routers import (
    auth,
    dashboard,
    semantics,
    robots,
    scenes,
    collect,
    data,
    preprocess,
    qc,
    viewer,
    annotation,
    export,
    models,
    training,
    log,
    api,
    system,
    reviews,
    collect_api,
    training_api,
    models_api,
    splits_api,
    platform,
)

app = FastAPI(title="VLA 数采平台")

# 建库建表（幂等）；数据落在 storage/ 下
db.init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(sim_api.router)
app.include_router(auth.router, prefix="/auth", tags=["认证"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["工作台"])
app.include_router(semantics.router, prefix="/semantics", tags=["数据定义"])
app.include_router(robots.router, prefix="/robots", tags=["机器人管理"])
app.include_router(scenes.router, prefix="/scenes", tags=["情景管理"])
app.include_router(collect.router, prefix="/collect", tags=["采集接入"])
app.include_router(data.router, prefix="/data", tags=["数据管理"])
app.include_router(preprocess.router, prefix="/preprocess", tags=["数据预处理"])
app.include_router(qc.router, prefix="/qc", tags=["数据质检"])
app.include_router(viewer.router, prefix="/viewer", tags=["数据浏览"])
app.include_router(viewer.api_router, prefix="/api", tags=["数据浏览接口"])
app.include_router(annotation.router, prefix="/annotation", tags=["数据标注"])
app.include_router(export.router, prefix="/export", tags=["数据集导出"])
app.include_router(models.router, prefix="/models", tags=["模型管理"])
app.include_router(training.router, prefix="/training", tags=["模型训练"])
# 日志页 + /log/api/logs 筛选接口
app.include_router(log.router, prefix="/log", tags=["系统日志"])
# 对外推理接口登记页
app.include_router(api.router, prefix="/interfaces", tags=["接口管理"])
# 平台自身的 JSON 接口（前端页面调用）
app.include_router(platform.router, prefix="/api", tags=["平台接口"])
# 标注质量抽检（页面 + /api/reviews/*）
app.include_router(reviews.router, prefix="/reviews", tags=["标注抽检"])
app.include_router(reviews.api_router, prefix="/api", tags=["标注抽检接口"])
# 实时采集会话接口（客户端上报）
app.include_router(collect_api.api_router, prefix="/api", tags=["采集会话接口"])
# 训练与离线评估接口
app.include_router(training_api.api_router, prefix="/api", tags=["训练与评估"])
# 真实推理代理（推理服务跑在 yolov8n 环境）
app.include_router(models_api.api_router, prefix="/api", tags=["推理代理"])
# 场景/任务与本体/设备接口
app.include_router(scenes.api_router, prefix="/api", tags=["场景与任务"])
app.include_router(robots.api_router, prefix="/api", tags=["本体与设备"])
# 数据划分接口
app.include_router(splits_api.api_router, prefix="/api", tags=["数据划分"])
# 存储清理接口
app.include_router(system.api_router, prefix="/api", tags=["存储"])
app.include_router(system.router, prefix="/system", tags=["系统管理"])


@app.middleware("http")
async def load_current_user(request: Request, call_next):
    """把当前登录用户挂到 request.state.user，供模板显示（Demo 不做强制登录拦截）"""
    request.state.user = auth.current_user(request)
    return await call_next(request)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "login.html", {"request": request})


@app.on_event("startup")
def _start_storage_scheduler() -> None:
    """启动存储策略检查线程（策略默认关闭，所以默认什么都不会删）。"""
    scheduler.start()
