"""本体与设备页：真实本体聚合 + 真实设备在环探测。"""
from fastapi import APIRouter, Body, Request
from fastapi.templating import Jinja2Templates

from core import audit, db, devices

router = APIRouter()
api_router = APIRouter()
templates = Jinja2Templates(directory="templates")


def overview() -> dict:
    """按 robot_type 聚合真实数据 + 采集会话统计。"""
    robots = db.query("""
        SELECT COALESCE(d.robot_type, '(未标注)') AS robot_type,
               COUNT(DISTINCT d.id) AS datasets,
               COALESCE(SUM(d.total_episodes), 0) AS episodes,
               COALESCE(SUM(d.total_frames), 0) AS frames,
               COALESCE(SUM(d.total_videos), 0) AS videos,
               MAX(d.refreshed_at) AS last_seen,
               GROUP_CONCAT(DISTINCT d.name) AS dataset_names
        FROM datasets d
        GROUP BY COALESCE(d.robot_type, '(未标注)')
        ORDER BY episodes DESC
    """)
    for row in robots:
        row["dataset_list"] = sorted(str(row["dataset_names"] or "").split(",")) if row["dataset_names"] else []
        row["sessions"] = db.scalar(
            "SELECT COUNT(*) AS c FROM collect_sessions s JOIN datasets d ON d.name = s.dataset_name"
            " WHERE COALESCE(d.robot_type, '(未标注)') = ?", (row["robot_type"],))
        row["session_frames"] = db.scalar(
            "SELECT COALESCE(SUM(s.frames), 0) AS c FROM collect_sessions s"
            " JOIN datasets d ON d.name = s.dataset_name"
            " WHERE COALESCE(d.robot_type, '(未标注)') = ? AND s.status='committed'",
            (row["robot_type"],)) or 0
    return {
        "robots": robots,
        "sessions": db.query("SELECT * FROM collect_sessions ORDER BY id DESC LIMIT 10"),
        "scans": devices.history(5),
        "session_totals": {
            "total": db.scalar("SELECT COUNT(*) AS c FROM collect_sessions"),
            "committed": db.scalar("SELECT COUNT(*) AS c FROM collect_sessions WHERE status='committed'"),
            "frames": db.scalar("SELECT COALESCE(SUM(frames), 0) AS c FROM collect_sessions"
                                " WHERE status='committed'"),
        },
    }


@router.get("")
async def robots_page(request: Request):
    return templates.TemplateResponse(request, "robots.html",
                                      {"request": request, **overview()})


@api_router.get("/robots/overview")
def robots_overview():
    return overview()


@api_router.get("/devices/scans")
def device_scans(limit: int = 10):
    return devices.history(limit)


@api_router.post("/devices/probe")
def device_probe(payload: dict = Body(default={})):
    """真的去问一遍本机有没有相机/串口/RealSense（采集前用）。"""
    result = devices.scan(operator=payload.get("operator") or "admin",
                          note=payload.get("note"))
    audit.log("INFO" if result["ready"] else "WARNING", "系统",
              f"设备探测：{result['summary']}", detail={"ready": result["ready"]})
    return result
