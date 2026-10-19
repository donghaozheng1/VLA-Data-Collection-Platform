#!/usr/bin/env python3
"""真实模型推理服务（纯标准库 + ultralytics）。

为什么是独立进程：本机 **base 环境的 torch 是坏的**（`torch_cuda.dll` 加载失败），
而 `envs\\yolov8n` 里 torch 2.11.0+cu128 + ultralytics 8.4.75 + RTX 5070 都正常，且该环境
没有 fastapi/httpx。所以这里用标准库 `http.server` 起一个只监听 127.0.0.1 的小服务，
平台侧（base 环境）通过 HTTP 代理调用——**零新增依赖**。

启动（必须用 yolov8n 环境的解释器）：

    C:\\Users\\ll\\miniconda3\\envs\\yolov8n\\python.exe tools\\infer_service.py \\
        --model C:\\Users\\ll\\yolo26n.pt --port 8100 \\
        --scan-dirs "C:\\Users\\ll;C:\\Users\\ll\\Desktop\\dhz\\model"

接口：
    GET  /health    -> 设备、torch/ultralytics 版本、已加载模型
    GET  /models    -> 扫描到的 .pt 清单（可选 probe=1 真实加载以读出类别）
    POST /predict   -> {image_b64, model?, conf?, render?} -> 真实检测框 + 真实耗时（+标注图）
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

MAX_BODY = 32 * 1024 * 1024          # 32MB，避免超大请求
SCAN_DIRS: list[Path] = []
MODELS: dict = {}                     # path -> YOLO 实例（懒加载缓存）
MODEL_META: dict = {}                 # path -> {classes, task, error}
DEFAULT_MODEL: str = ""
DEVICE_INFO: dict = {}


def allowed(path: str | Path) -> bool:
    """只允许加载扫描目录下的权重（默认模型除外）。"""
    try:
        target = Path(path).resolve()
    except OSError:
        return False
    if DEFAULT_MODEL and str(target) == str(Path(DEFAULT_MODEL).resolve()):
        return True
    for directory in SCAN_DIRS:
        try:
            target.relative_to(Path(directory).resolve())
            return True
        except ValueError:
            continue
    return False


def discover() -> list[dict]:
    """扫描 .pt 权重（不加载，只列文件）。"""
    found: list[dict] = []
    seen = set()
    if DEFAULT_MODEL and Path(DEFAULT_MODEL).is_file():
        found.append(_entry(Path(DEFAULT_MODEL)))
        seen.add(str(Path(DEFAULT_MODEL).resolve()))
    for directory in SCAN_DIRS:
        base = Path(directory)
        if not base.is_dir():
            continue
        for file in sorted(base.glob("*.pt")):
            key = str(file.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(_entry(file))
    return found


def _entry(file: Path) -> dict:
    meta = MODEL_META.get(str(file), {})
    return {"name": file.name, "path": str(file), "size_mb": round(file.stat().st_size / 1024 / 1024, 1),
            "loaded": str(file) in MODELS, "classes": meta.get("classes"),
            "num_classes": meta.get("num_classes"), "task": meta.get("task"),
            "loadable": meta.get("loadable"), "error": meta.get("error")}


def resolve_model(name_or_path: str | None) -> str:
    """把模型名（如 best_0.pt）或路径解析成真实路径。"""
    if not name_or_path:
        return DEFAULT_MODEL
    direct = Path(name_or_path)
    if direct.is_file():
        return str(direct)
    stem = direct.stem
    for item in discover():
        if item["name"] == name_or_path or Path(item["path"]).stem == stem:
            return item["path"]
    raise ValueError(f"找不到模型：{name_or_path}（可用：{[m['name'] for m in discover()]}")


def load_model(path: str):
    """加载并缓存模型；失败时把原因记在 MODEL_META 里（供界面显示）。"""
    key = str(Path(path))
    if key in MODELS:
        return MODELS[key]
    if not allowed(key):
        raise ValueError(f"不允许加载该路径（不在扫描目录内）：{path}")
    from ultralytics import YOLO

    try:
        model = YOLO(key)
        info = {"classes": list(model.names.values()), "num_classes": len(model.names),
                "task": model.task, "loadable": True, "error": None}
        MODELS[key] = model
        MODEL_META[key] = info
        return model
    except Exception as exc:  # noqa: BLE001
        MODEL_META[key] = {"classes": None, "num_classes": None, "task": None,
                           "loadable": False, "error": str(exc).splitlines()[0][:200]}
        raise


def predict(payload: dict) -> dict:
    model_path = resolve_model(payload.get("model"))
    conf = float(payload.get("conf") or 0.25)
    render = bool(payload.get("render", True))
    raw = base64.b64decode((payload.get("image_b64") or "").split(",")[-1])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图像解码失败（需要 PNG/JPG 的 base64）")

    model = load_model(model_path)
    t0 = time.perf_counter()
    results = model.predict(image, conf=conf, verbose=False)
    wall_ms = (time.perf_counter() - t0) * 1000
    result = results[0]
    speed = {k: round(float(v), 2) for k, v in (getattr(result, "speed", {}) or {}).items()}

    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            detections.append({
                "class": result.names.get(class_id, str(class_id)),
                "class_id": class_id,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

    out = {
        "model": Path(model_path).name,
        "model_path": str(model_path),
        "detections": detections,
        "count": len(detections),
        "conf": conf,
        "inference_ms": round(wall_ms, 1),
        "speed_ms": speed,
        "device": DEVICE_INFO.get("device"),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
    }
    if render:
        annotated = result.plot()          # 真实画框（ultralytics 自己画）
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            out["rendered_b64"] = base64.b64encode(buf.tobytes()).decode()
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "hkc-infer/1.0"

    def log_message(self, fmt, *args):        # 静音访问日志
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                 # noqa: N802
        path, _, query = self.path.partition("?")
        if path == "/health":
            self._send(200, {"status": "ok", "device": DEVICE_INFO.get("device"),
                             "torch": DEVICE_INFO.get("torch"),
                             "ultralytics": DEVICE_INFO.get("ultralytics"),
                             "cuda": DEVICE_INFO.get("cuda"),
                             "default_model": Path(DEFAULT_MODEL).name,
                             "loaded_models": [Path(p).name for p in MODELS],
                             "scan_dirs": [str(d) for d in SCAN_DIRS]})
            return
        if path == "/models":
            probe = "probe=1" in query
            if probe:
                for item in discover():
                    if item["loaded"] or item.get("loadable") is not None:
                        continue
                    try:
                        load_model(item["path"])
                    except Exception:  # noqa: BLE001
                        pass
            self._send(200, {"models": discover(), "default": Path(DEFAULT_MODEL).name})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:                # noqa: N802
        if self.path.partition("?")[0] != "/predict":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"error": f"请求体大小异常：{length}"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send(200, predict(payload))
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"{type(exc).__name__}: {exc}",
                             "trace": traceback.format_exc()[-800:]})


def main() -> int:
    global SCAN_DIRS, DEFAULT_MODEL, DEVICE_INFO
    parser = argparse.ArgumentParser(description="HKC 真实模型推理服务（纯标准库）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default=r"C:\Users\ll\yolo26n.pt")
    parser.add_argument("--scan-dirs", default=r"C:\Users\ll;C:\Users\ll\Desktop\dhz\model")
    args = parser.parse_args()

    SCAN_DIRS = [Path(p.strip()) for p in args.scan_dirs.split(";") if p.strip()]
    DEFAULT_MODEL = args.model

    import torch
    import ultralytics

    DEVICE_INFO = {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }
    print(f"[infer] python={__import__('sys').version.split()[0]} torch={DEVICE_INFO['torch']} "
          f"cuda={DEVICE_INFO['cuda']} device={DEVICE_INFO['device']}")
    print(f"[infer] 默认模型={DEFAULT_MODEL}")
    print(f"[infer] 扫描目录={[str(d) for d in SCAN_DIRS]}")
    print(f"[infer] 监听 http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
