"""仿真直播的页面与接口。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from core import sim

router = APIRouter()
templates = Jinja2Templates(directory="templates")

BOUNDARY = "frame"


def _mjpeg():
    """把**当前会话**的最新帧按 multipart 边界推给浏览器。

    关键：每轮都重新取 session()，这样 /api/sim/start 换了会话之后这条流会自动跟上
    （之前只在开头取一次，换了会话就一直读旧会话的空帧，画面就"不动"了）。
    """
    import time

    sim.VIEWERS["count"] += 1
    last = None
    try:
        while True:
            sess = sim.session()
            payload = sess.frame_jpeg()
            if payload and payload is not last:
                last = payload
                yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
                       + payload + b"\r\n")
            elif payload:
                yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
                       + payload + b"\r\n")
            time.sleep(1.0 / 30)
    finally:
        sim.VIEWERS["count"] = max(0, sim.VIEWERS["count"] - 1)


_PLACEHOLDER: bytes | None = None


def _placeholder_png() -> bytes:
    """没画面时返回一张占位图。

    之前这里返回 404，而页面每 80ms 轮询一次 -> 浏览器一直重试、画面闪烁、日志刷屏。
    改成返回 200 + 一张"等待中"图片：不再报错，页面上也能看懂状态。
    """
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        import cv2
        import numpy as np

        img = np.full((256, 256, 3), 26, np.uint8)
        cv2.putText(img, "no frame yet", (48, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(img, "click START", (58, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        ok, buf = cv2.imencode(".png", img)
        _PLACEHOLDER = buf.tobytes() if ok else b""
    return _PLACEHOLDER


@router.get("/api/sim/frame")
def sim_frame():
    """单帧接口（轮询用）：拿到当前会话的最新一帧 JPEG；没有则给占位图（始终 200）。"""
    from fastapi import Response

    payload = sim.session().frame_jpeg()
    if not payload:
        return Response(content=_placeholder_png(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})
    return Response(content=payload, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                             "Pragma": "no-cache"})


@router.get("/simulation")
def simulation_page(request: Request):
    return templates.TemplateResponse(request, "simulation.html", {"active": "simulation"})


@router.get("/api/sim/stream")
def sim_stream():
    return StreamingResponse(_mjpeg(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


@router.get("/api/sim/models")
def sim_models():
    """列出可用于仿真直播的策略模型（lerobot backend）。

    LIBERO 仿真的观测是「2 路相机 + 8 维 state」，所以额外算一个 compatible 标记：
    与 LIBERO 观测不匹配的模型（例如 14 维 state 的 zhiyan）会被标出来，避免选了跑不动。
    """
    import json
    import pathlib

    from core import db, lerobot

    out = []
    for m in db.query("SELECT * FROM models ORDER BY id DESC"):
        if m["backend"] != "lerobot":
            continue
        path = pathlib.Path(m["path"])
        policy_dir = path.parent if path.parent.name == "pretrained_model" else path
        if not (policy_dir / "config.json").is_file():
            continue

        dataset_name, cameras, state_dim, action_dim = None, [], None, None
        if m.get("dataset_id"):
            row = db.query_one("SELECT name, path FROM datasets WHERE id=?", (m["dataset_id"],))
            if row:
                dataset_name = row["name"]
                try:
                    info = lerobot.load_meta(row["path"])["info"]
                    feats = info.get("features") or {}
                    cameras = [k for k, v in feats.items() if (v or {}).get("dtype") in ("video", "image")]
                    st = feats.get("observation.state") or {}
                    ac = feats.get("action") or {}
                    state_dim = (st.get("shape") or [None])[0]
                    action_dim = (ac.get("shape") or [None])[0]
                except Exception:  # noqa: BLE001
                    pass

        steps = None
        try:
            steps = (json.loads(m.get("metrics_json") or "{}") or {}).get("steps")
        except Exception:  # noqa: BLE001
            pass

        out.append({
            "id": m["id"], "name": m["name"], "kind": m["kind"],
            "policy_path": str(policy_dir), "dataset": dataset_name,
            "cameras": cameras, "state_dim": state_dim, "action_dim": action_dim,
            "steps": steps,
            "compatible": len(cameras) == 2 and state_dim == 8,
        })
    return {"models": out}


@router.get("/api/sim/status")
def sim_status():
    return sim.session().status()


@router.post("/api/sim/start")
def sim_start(payload: dict | None = None):
    payload = payload or {}
    sess = sim.new_session(steps=int(payload.get("steps") or 600),
                           size=int(payload.get("size") or 256),
                           fps=int(payload.get("fps") or 20),
                           seed=int(payload.get("seed") or 0),
                           policy_path=payload.get("policy_path"),
                           mode=str(payload.get("mode") or "libero"),
                           episode=int(payload.get("episode") or 0))
    return sess.start()


@router.post("/api/sim/pause")
def sim_pause():
    sim.session().pause()
    return sim.session().status()


@router.post("/api/sim/resume")
def sim_resume():
    sim.session().resume()
    return sim.session().status()


@router.post("/api/sim/stop")
def sim_stop():
    sess = sim.session()
    sess.stop()
    return sess.status()


@router.get("/api/sim/video")
def sim_video():
    sess = sim.session()
    if not sess.video:
        from fastapi import HTTPException
        raise HTTPException(404, "还没有录像（先跑一次仿真）")
    from pathlib import Path
    path = Path(sess.video)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        from fastapi import HTTPException
        raise HTTPException(404, f"录像文件不在了：{path}")
    return FileResponse(str(path), filename=path.name, media_type="video/mp4")
