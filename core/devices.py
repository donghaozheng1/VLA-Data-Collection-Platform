"""设备在环探测：真的去问一遍本机有没有相机 / 串口 / RealSense。

用途很直接：**采集前一眼看出设备认没认到**。三种探测都是真探测，不是查配置：

1. 相机 —— 用 cv2 依次尝试打开 0..N 号设备并真的读一帧（读得到才算认到）。
2. 串口 —— 读注册表 `HKLM\\HARDWARE\\DEVICEMAP\\SERIALCOMM`（标准库 winreg，无副作用，不会
   像 open("COMx") 那样去抢 DTR）；拿不到时退化用 PowerShell 查 PnP。
3. RealSense —— 尝试 import pyrealsense2 并枚举设备；SDK 不在就如实报告缺失。
"""
from __future__ import annotations

import json

from . import db


def probe_cameras(max_index: int = 4) -> list[dict]:
    """尝试打开 UVC 相机并读一帧。"""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"缺少 opencv：{exc}"}]
    found: list[dict] = []
    backends = [("any", cv2.CAP_ANY), ("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]
    for index in range(max_index):
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            try:
                if not cap.isOpened():
                    continue
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 1)
                ok, frame = cap.read()          # 真的读一帧，读得到才算认到
                found.append({"index": index, "backend": name, "width": width, "height": height,
                              "fps": fps, "read_ok": bool(ok),
                              "frame_shape": list(frame.shape) if ok and frame is not None else None})
                break
            finally:
                cap.release()
    return found


def probe_ports() -> dict:
    """枚举串口（机械臂控制器的典型形态）。"""
    ports: list[str] = []
    method = "registry"
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM") as key:
            index = 0
            while True:
                try:
                    _name, value, _type = winreg.EnumValue(key, index)
                except OSError:
                    break
                ports.append(str(value))
                index += 1
    except Exception:  # noqa: BLE001
        method = "powershell"
        try:
            import subprocess

            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-PnpDevice -PresentOnly -Class Ports -ErrorAction SilentlyContinue | "
                 "Where-Object { $_.FriendlyName -match 'COM\\d+' } | "
                 "Select-Object -ExpandProperty FriendlyName"],
                capture_output=True, text=True, timeout=20)
            ports = [line.strip() for line in (out.stdout or "").splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            return {"ports": [], "method": "failed", "error": str(exc)[:160]}
    return {"ports": sorted(set(ports)), "method": method}


def probe_realsense() -> dict:
    """RealSense：SDK 是否可用 + 枚举到的设备。"""
    try:
        import pyrealsense2 as rs
    except Exception as exc:  # noqa: BLE001
        return {"sdk": False, "devices": [], "error": f"{type(exc).__name__}: {exc}"[:160]}
    try:
        devices = []
        for device in rs.context().query_devices():
            devices.append({
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            })
        return {"sdk": True, "devices": devices}
    except Exception as exc:  # noqa: BLE001
        return {"sdk": True, "devices": [], "error": f"{type(exc).__name__}: {exc}"[:160]}


def scan(operator: str = "admin", note: str | None = None, persist: bool = True) -> dict:
    """完整探测一次并（默认）落库，供 /robots 展示历史。"""
    cameras = probe_cameras()
    ports = probe_ports()
    realsense = probe_realsense()
    summary = (f"相机 {len([c for c in cameras if c.get('read_ok')])} 个 / "
               f"串口 {len(ports.get('ports', []))} 个 / "
               f"RealSense {'SDK 可用' if realsense.get('sdk') else 'SDK 缺失'}"
               f"（设备 {len(realsense.get('devices', []))} 个）")
    result = {"ts": db.now(), "cameras": cameras, "ports": ports, "realsense": realsense,
              "summary": summary, "note": note, "operator": operator,
              "ready": bool([c for c in cameras if c.get("read_ok")])
                       or bool(ports.get("ports")) or bool(realsense.get("devices"))}
    if persist:
        scan_id = db.execute(
            "INSERT INTO device_scans (ts, cameras_json, ports_json, realsense_json, summary, operator)"
            " VALUES (?,?,?,?,?,?)",
            (result["ts"], json.dumps(cameras, ensure_ascii=False),
             json.dumps(ports, ensure_ascii=False), json.dumps(realsense, ensure_ascii=False),
             summary, operator))
        result["scan_id"] = scan_id
    return result


def history(limit: int = 10) -> list[dict]:
    rows = db.query("SELECT * FROM device_scans ORDER BY id DESC LIMIT ?", (limit,))
    for row in rows:
        for key in ("cameras", "ports", "realsense"):
            try:
                row[key] = json.loads(row.get(f"{key}_json") or "null")
            except Exception:  # noqa: BLE001
                row[key] = None
    return rows
