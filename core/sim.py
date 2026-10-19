"""LIBERO 仿真直播会话（父进程侧）。

架构（重要）：**仿真不在服务进程里跑**。平台服务用的是 base miniconda 的 python，而它的
torch 是坏的（`torch_cuda.dll` 加载失败），一旦在服务进程里 import 到 torch 就会整体崩。
所以这里改为：

    服务进程 ──启动子进程──> yolov8n python（torch 正常）+ tools/libero_stream.py
       ▲                                   │
       └──── 读 latest.jpg / status.json ───┘   （帧通过文件传递，不用命名管道——沙箱禁止）

父进程只做三件事：拉起/停止子进程、读最新一帧给 MJPEG 流、读状态给前端。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "storage" / "sim_runs"
PYTHON = pathlib.Path(r"C:\Users\ll\miniconda3\envs\yolov8n\python.exe")
STREAM_SCRIPT = ROOT / "tools" / "libero_stream.py"
TASK = "put both the alphabet soup and the tomato sauce in the basket"
VIEWERS = {"count": 0}          # 全局观看者计数（换会话也不丢）


class SimSession:
    mode = "libero"
    episode = 0
    def __init__(self, steps: int = 600, size: int = 256, fps: int = 20, seed: int = 0,
                 policy_path: str | None = None):
        self.steps = steps
        self.size = size
        self.fps = fps
        self.seed = seed
        self.policy_path = policy_path
        self.run_dir = RUNS / time.strftime("%Y%m%d_%H%M%S")
        self.process: subprocess.Popen | None = None
        self.started_at: float | None = None
        self._frame: bytes | None = None
        self._frame_stamp: float = 0.0
        self._subscribers = 0

    # ---------- 内部 ----------
    def _status_file(self) -> dict:
        path = self.run_dir / "status.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    # ---------- 对外 ----------
    def status(self) -> dict:
        data = self._status_file()
        state = data.get("state") or ("running" if self._alive() else "idle")
        if self._alive() and state in ("finished", "failed"):
            state = state                       # 子进程已写终态
        elif not self._alive() and state not in ("finished", "failed") and self.started_at:
            state = "failed" if (data.get("error") or self.process.returncode) else "finished"
        return {"state": state, "step": data.get("step", 0), "steps": self.steps,
                "reward": data.get("reward", 0.0), "task": data.get("task") or TASK,
                "error": data.get("error"), "seconds": data.get("seconds", 0.0),
                "task": data.get("task"),
                "fps": self.fps, "size": self.size, "video": data.get("video"),
                "policy": self.policy_path, "viewers": VIEWERS["count"],
                "mode": self.mode, "episode": self.episode,
                "run_dir": str(self.run_dir.relative_to(ROOT))}

    def frame_jpeg(self) -> bytes | None:
        """取最新一帧：扫描 frames/ 里编号最大的文件。

        子进程不再写 latest.jpg（改用编号文件，避免 Windows 上改名撞文件锁），
        所以父进程必须来这里取；找不到时回退到上一次缓存的帧。
        """
        frames_dir = self.run_dir / "frames"
        if not frames_dir.is_dir():
            return self._frame
        newest = None
        newest_key = self._frame_stamp
        for item in frames_dir.glob("f*.jpg"):
            try:
                key = int(item.stem[1:])
            except ValueError:
                continue
            if key >= newest_key:
                newest_key, newest = key, item
        if newest is None:
            return self._frame
        try:
            self._frame = newest.read_bytes()
            self._frame_stamp = newest_key
        except Exception:  # noqa: BLE001
            pass
        return self._frame

    def start(self) -> dict:
        if self._alive():
            return self.status()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log = (self.run_dir / "child.log").open("w", encoding="utf-8")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MUJOCO_GL": "wgl",
               "LIBERO_CONFIG_PATH": str(ROOT / "storage" / "libero" / "config")}
        env.pop("PYTHONPATH", None)          # 子进程自己按顺序插路径
        if self.mode in ("zhiyan_replay", "zhiyan_policy"):
            # zhiyan_replay = 真机轨迹回放；zhiyan_policy = 我的模型驱动仿真（需 --policy）
            script = "zhiyan_policy.py" if self.mode == "zhiyan_policy" else "zhiyan_replay.py"
            argv = [str(PYTHON), str(ROOT / "tools" / script),
                    "--run", str(self.run_dir), "--episode", str(self.episode),
                    "--steps", str(self.steps), "--size", str(self.size), "--fps", str(self.fps)]
            if self.mode == "zhiyan_policy" and self.policy_path:
                argv += ["--policy", self.policy_path]
        else:
            argv = [str(PYTHON), str(STREAM_SCRIPT), "--run", str(self.run_dir),
                    "--steps", str(self.steps), "--size", str(self.size),
                    "--fps", str(self.fps), "--seed", str(self.seed)] \
                   + (["--policy", self.policy_path] if self.policy_path else [])
        self.process = subprocess.Popen(
            argv,
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        self.started_at = time.time()
        return self.status()

    def _control(self, value: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "control.txt").write_text(value, encoding="utf-8")

    def pause(self) -> None:
        if self._alive():
            self._control("pause")

    def resume(self) -> None:
        if self._alive():
            self._control("run")

    def stop(self) -> None:
        self._control("stop")
        for _ in range(20):                  # 给它 2 秒自己收尾（写好录像与状态）
            if not self._alive():
                return
            time.sleep(0.1)
        if self._alive():
            self.process.terminate()


SESSION: SimSession | None = None


def session() -> SimSession:
    global SESSION
    if SESSION is None:
        SESSION = SimSession()
    return SESSION


def new_session(**kwargs) -> SimSession:
    global SESSION
    if SESSION is not None:
        SESSION.stop()
    mode = str(kwargs.pop("mode", "libero"))
    episode = int(kwargs.pop("episode", 0))
    SESSION = SimSession(**kwargs)
    SESSION.mode = mode
    SESSION.episode = episode
    return SESSION
