"""系统日志：展示平台**真实发生过的操作**（来自 core.audit 的 audit_log）。

原实现是随机生成的 mock 日志，已替换为真实审计记录；接口形状保持不变
（`/log/api/logs?level=&module=&limit=` -> `{code, data, total, modules, stats}`），
既有前端与测试无需改动。
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from core import audit

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("")
async def log_page(request: Request):
    return templates.TemplateResponse(request, "log.html", {
        "request": request,
        "logs": audit.query(limit=50),
        "stats": audit.stats(),
        "modules": audit.modules(),
    })


@router.get("/api/logs")
async def get_logs_api(request: Request, level: str = None, module: str = None, limit: int = 50):
    limit = max(0, min(int(limit or 50), 500))          # 顺手修掉 limit=-1 会返回 99 条的旧问题
    rows = audit.query(level, module, limit)
    return {"code": 0, "data": rows, "total": len(rows),
            "modules": audit.modules(), "stats": audit.stats()}
