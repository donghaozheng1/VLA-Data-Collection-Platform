"""模型管理与真实推理代理（/api/inference/*）。

平台跑在 base 环境，而**本机 base 的 torch 是坏的**（`torch_cuda.dll` 加载失败）；
真正的推理由 `envs\\yolov8n` 里的 `tools/infer_service.py` 提供（纯标准库 HTTP 服务）。
所以这里只做代理，不把 torch 塞进 Web 进程。

服务地址可用环境变量 `HKC_INFER_URL` 覆盖，默认 http://127.0.0.1:8100
"""
from __future__ import annotations

import os
import time

import httpx
from fastapi import APIRouter, Body, HTTPException

from core import audit

api_router = APIRouter()

INFER_URL = os.environ.get("HKC_INFER_URL", "http://127.0.0.1:8100")
START_COMMAND = (
    r'& "C:\Users\ll\miniconda3\envs\yolov8n\python.exe" tools\infer_service.py '
    r'--model "C:\Users\ll\yolo26n.pt" --port 8100 '
    r'--scan-dirs "C:\Users\ll;C:\Users\ll\Desktop\dhz\model"'
)


def _down(exc: Exception) -> dict:
    return {
        "running": False,
        "service_url": INFER_URL,
        "error": f"{type(exc).__name__}: {exc}",
        "start_command": START_COMMAND,
        "hint": "推理服务未启动。它在另一个 Python 环境里（torch 可用的那个），用上面的命令在项目根目录启动。",
    }


# 推理调用审计要做节流：30fps 推理会把日志刷爆，这里聚合成"每 N 分钟一条摘要"
_USAGE = {"count": 0, "models": {}, "since": time.time()}
USAGE_WINDOW_S = 300


def _record_usage(model: str) -> None:
    _USAGE["count"] += 1
    _USAGE["models"][model] = _USAGE["models"].get(model, 0) + 1
    elapsed = time.time() - _USAGE["since"]
    if elapsed >= USAGE_WINDOW_S and _USAGE["count"]:
        audit.log("INFO", "推理服务",
                  f"近 {int(elapsed // 60)} 分钟推理 {_USAGE['count']} 次："
                  + "、".join(f"{k}×{v}" for k, v in _USAGE["models"].items()),
                  detail={"calls": _USAGE["count"], "models": dict(_USAGE["models"]),
                          "window_s": int(elapsed)})
        _USAGE.update({"count": 0, "models": {}, "since": time.time()})


@api_router.get("/inference/health")
def inference_health():
    try:
        data = httpx.get(f"{INFER_URL}/health", timeout=8).json()
        return {**data, "running": True, "service_url": INFER_URL}
    except Exception as exc:  # noqa: BLE001
        return _down(exc)


@api_router.get("/inference/models")
def inference_models(probe: bool = True):
    try:
        suffix = "?probe=1" if probe else ""
        data = httpx.get(f"{INFER_URL}/models{suffix}", timeout=300).json()
        return {**data, "running": True, "service_url": INFER_URL}
    except Exception as exc:  # noqa: BLE001
        return {**_down(exc), "models": [], "default": None}


@api_router.post("/inference/predict")
def inference_predict(payload: dict = Body(default={})):
    if not payload.get("image_b64"):
        raise HTTPException(400, "缺少 image_b64")
    try:
        resp = httpx.post(f"{INFER_URL}/predict", json=payload, timeout=300)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"推理服务不可达：{type(exc).__name__}: {exc}。"
                                 f"启动命令见 /api/inference/health") from exc
    if resp.status_code == 200:
        try:
            _record_usage(resp.json().get("model") or "unknown")
        except Exception:  # noqa: BLE001
            pass
    if resp.status_code >= 400:
        detail = resp.json().get("error") if resp.headers.get("content-type", "").startswith("application/json") \
            else resp.text[:200]
        raise HTTPException(resp.status_code, f"推理失败：{detail}")
    return resp.json()


@api_router.get("/inference/config")
def inference_config():
    return {"service_url": INFER_URL, "start_command": START_COMMAND}
