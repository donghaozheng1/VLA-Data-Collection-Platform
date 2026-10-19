"""轻量调度：定期检查存储策略。

平台不是常驻服务架构（uvicorn 单进程 + 线程任务），所以这里用一个 **daemon 线程**做定时检查，
不引入 celery/apscheduler 这类外部依赖。

原则：**默认关闭**（`storage_policy.enabled=0`）。无人值守自动删文件风险太大，
必须由人显式打开、显式设定阈值与类别；每次检查与每次执行都进审计日志。
"""
from __future__ import annotations

import threading
import time

from . import audit, storage

CHECK_INTERVAL_S = 300
_thread: threading.Thread | None = None
_stop = threading.Event()


def tick() -> dict:
    """跑一次检查（供调度线程与手动接口共用）。"""
    try:
        return storage.auto_check()
    except Exception as exc:  # noqa: BLE001 - 调度线程绝不能因为异常挂掉
        audit.log("ERROR", "存储", f"自动清理检查异常：{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}


def _loop() -> None:
    while not _stop.is_set():
        policy = storage.policy()
        tick()
        interval = max(30, int(policy.get("interval_s") or CHECK_INTERVAL_S))
        _stop.wait(interval)


def start() -> dict:
    """启动后台检查线程（幂等）。"""
    global _thread
    if _thread and _thread.is_alive():
        return {"started": False, "reason": "已在运行"}
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="storage-scheduler", daemon=True)
    _thread.start()
    policy = storage.policy()
    audit.log("INFO", "存储",
              f"存储策略调度已启动（间隔 {policy.get('interval_s')}s，"
              f"自动清理 {'开启' if policy.get('enabled') else '关闭'}）")
    return {"started": True, "interval_s": policy.get("interval_s"),
            "enabled": bool(policy.get("enabled"))}


def status() -> dict:
    return {"thread_alive": bool(_thread and _thread.is_alive()),
            "interval_s": storage.policy().get("interval_s")}
