"""网页里发起训练、登记产出的模型。

两条后端：

1. `lerobot` —— **真训练**：调你们的 LeRobot 训练脚本（`src/lerobot/scripts/train.py`），
   流式读日志、解析 `step/loss` 更新进度，训练完把 checkpoint 登记成模型。
   **前提是有一个装了 lerobot 的 Python 环境**；当前机器上没有（见 `detect_envs()` 的实测结果），
   所以这条路的按钮会明确告诉你缺什么，而不是假装开始跑。
2. `builtin` —— **平台自带的最小行为克隆**：只用平台已有的依赖（numpy + pyarrow），
   从选定的**划分**里读 state→action 学一个线性策略，在 val 划分上报 MSE，产出
   `.npz` + `model.json` 并登记到模型库。

`builtin` 的意义是把「网页 → 训练 → 产出模型 → 下载」这条路走通并可验证；
它**不用图像**（只用 state→action），所以不是 VLA 模型 —— 这一点写在模型元数据里，不冒充。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from . import audit, db, jobs, lerobot, training as training_mod

LEROBOT_TRAIN = Path(r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\src\lerobot\scripts\train.py")
LEROBOT_SRC = LEROBOT_TRAIN.parents[2]          # .../lerobot_project/src
# lerobot 的依赖装在**工作区私有目录**里（`pip install --target`），不碰 yolov8n；
# 训练时用 PYTHONPATH 叠加：私有依赖 + lerobot 源码 + yolov8n 自己的 torch。
LEROBOT_DEPS = Path(__file__).resolve().parent.parent / ".deps_lerobot"
MODEL_DIR = db.STORAGE / "models"


def lerobot_python() -> Path | None:
    """可以用来的训练解释器：需要 torch（复用 yolov8n）。"""
    for candidate in (Path(r"C:\Users\ll\miniconda3\envs\yolov8n\python.exe"),):
        if candidate.is_file():
            return candidate
    return None


def lerobot_env() -> dict:
    """训练子进程的环境变量：把私有依赖与 lerobot 源码放到 PYTHONPATH 前面。"""
    import os

    parts = [str(LEROBOT_DEPS), str(LEROBOT_SRC)]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    # HF 的 datasets/hub 默认缓存到 C:\Users\<用户>\.cache\huggingface —— 在工作区之外，
    # 会被沙箱拒绝（WinError 5）。全部指到工作区内的 storage/hf_cache。
    hf_home = db.STORAGE / "hf_cache"
    (hf_home / "datasets").mkdir(parents=True, exist_ok=True)
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(parts),
            "HF_HOME": str(hf_home),
            "HF_DATASETS_CACHE": str(hf_home / "datasets"),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "TORCH_HOME": str(hf_home / "torch"),
            "XDG_CACHE_HOME": str(hf_home),
            "HF_HUB_DISABLE_TELEMETRY": "1"}


# --------------------------------------------------------------------------- 环境探测

def detect_envs() -> list[dict]:
    """探测本机哪些 Python 能跑训练（torch / lerobot）。结果不缓存，UI 点了就实测。"""
    probe = ("import importlib,sys\n"
             "r={}\n"
             "for m in ('torch','lerobot'):\n"
             "    try:\n"
             "        mod=importlib.import_module(m); r[m]=getattr(mod,'__version__','?')\n"
             "    except Exception:\n"
             "        r[m]=None\n"
             "try:\n"
             "    import torch; r['cuda']=bool(torch.cuda.is_available())\n"
             "except Exception:\n"
             "    r['cuda']=None\n"
             "print(__import__('json').dumps(r))\n")
    candidates = [Path(r"C:\Users\ll\miniconda3\python.exe")]
    envs = Path(r"C:\Users\ll\miniconda3\envs")
    if envs.is_dir():
        candidates += [d / "python.exe" for d in envs.iterdir() if d.is_dir()]
    out = []
    yolo = lerobot_python()
    if yolo and yolo not in candidates:
        candidates.append(yolo)
    for path in candidates:
        if not path.is_file():
            continue
        item = {"python": str(path), "torch": None, "lerobot": None, "cuda": None,
                "usable_for": [], "error": None}
        try:
            # 对「装了私有依赖的那个解释器」要用叠加后的 PYTHONPATH 探测，否则看到的 lerobot 是没装
            env = lerobot_env() if (yolo and path == yolo and LEROBOT_DEPS.is_dir()) else None
            res = subprocess.run([str(path), "-c", probe], capture_output=True, text=True,
                                 timeout=180, env=env)
            data = json.loads((res.stdout or "{}").strip().splitlines()[-1]) if res.stdout.strip() else {}
            item.update({k: data.get(k) for k in ("torch", "lerobot", "cuda")})
            item["private_deps"] = str(LEROBOT_DEPS) if env else None
            item["deps_present"] = LEROBOT_DEPS.is_dir()
            if item["lerobot"]:
                item["usable_for"].append("lerobot")
            if item["torch"]:
                item["usable_for"].append("builtin")
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"[:160]
        out.append(item)
    return out


def pick_env(backend: str) -> str | None:
    envs = detect_envs()
    if backend in ("lerobot", "builtin"):
        yolo = lerobot_python()
        if yolo:
            for item in envs:
                if item["python"] == str(yolo) and backend in item["usable_for"]:
                    return item["python"]
    for item in envs:
        if backend in item["usable_for"]:
            return item["python"]
    return None


# --------------------------------------------------------------------------- 划分取数

def _split_frames(dataset_id: int, split: str) -> tuple[list, list, dict, list[int]]:
    """读某个划分里所有帧的 (state, action)。划分没做就报错，不默认全量。"""
    from . import splits as splits_mod

    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError(f"数据集不存在：{dataset_id}")
    overview = splits_mod.overview(dataset_id)
    if not overview.get("assigned"):
        raise ValueError(f"数据集 #{dataset_id} 还没做数据划分 —— 先去导出页做一次（否则评估集不隔离）")
    if split not in overview["counts"]:
        raise ValueError(f"数据集 #{dataset_id} 没有 '{split}' 划分（现有：{list(overview['counts'])}）")

    import pyarrow.parquet as pq

    indices = overview["splits"][split]
    meta = lerobot.load_meta(row["path"])
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for index in indices:
        paths = lerobot.episode_paths(meta["info"], row["path"], index)
        table = pq.read_table(paths["parquet"])
        data = table.to_pydict()
        states.extend([[float(x) for x in row_] for row_ in data["observation.state"]])
        actions.extend([[float(x) for x in row_] for row_ in data["action"]])
    return states, actions, {"dataset": row["name"], "version": row["version"],
                             "content_hash": row["content_hash"], "salt": overview["salt"],
                             "split": split, "episodes": indices}, indices


# --------------------------------------------------------------------------- builtin 训练

@jobs.handler("train")
def train_job(job: jobs.JobContext, payload: dict) -> dict:
    backend = payload.get("backend") or "builtin"
    if backend == "lerobot":
        return _train_lerobot(job, payload)
    return _train_builtin(job, payload)


def _train_builtin(job: jobs.JobContext, payload: dict) -> dict:
    """最小行为克隆：线性 ridge 拟合 state → action，在 val 划分上报 MSE。"""
    import numpy as np

    dataset_id = int(payload["dataset_id"])
    train_split = payload.get("split") or "train"
    val_split = payload.get("val_split") or "val"
    lam = float(payload.get("lambda") or 1e-2)

    job.progress(1, 5, f"读取划分 {train_split}")
    x_train, y_train, info, train_idx = _split_frames(dataset_id, train_split)
    if len(x_train) < 2:
        raise ValueError(f"'{train_split}' 划分只有 {len(x_train)} 帧，样本太少"
                         f"（把 train 比例调大，或换个更大的数据集）")
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    job.progress(2, 5, "拟合线性策略")
    ones = np.ones((x_train.shape[0], 1))
    design = np.hstack([x_train, ones])
    gram = design.T @ design + lam * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ y_train)

    job.progress(3, 5, "在训练集上评估")
    train_pred = design @ weights
    train_mse = float(np.mean((train_pred - y_train) ** 2))

    val_mse = None
    val_frames = 0
    try:
        x_val, y_val, _vi, val_idx = _split_frames(dataset_id, val_split)
        if x_val:
            x_val = np.asarray(x_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
            val_design = np.hstack([x_val, np.ones((x_val.shape[0], 1))])
            val_mse = float(np.mean((val_design @ weights - y_val) ** 2))
            val_frames = len(x_val)
    except ValueError as exc:
        job.log(f"（跳过 val 评估：{exc}）")

    job.progress(4, 5, "保存模型")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = db.now().replace("-", "").replace(":", "").replace(" ", "_")
    name = payload.get("name") or f"{info['dataset']}_bc_{stamp}"
    model_path = MODEL_DIR / f"{name}.npz"
    np.savez(model_path, weights=weights, state_dim=x_train.shape[1], action_dim=y_train.shape[1])
    metrics = {"train_mse": train_mse, "val_mse": val_mse, "train_frames": len(x_train),
               "val_frames": val_frames, "lambda": lam}
    meta_path = MODEL_DIR / f"{name}.json"
    meta_path.write_text(json.dumps({
        "name": name, "backend": "builtin", "kind": "linear-behavior-cloning",
        "uses_vision": False,
        "note": "平台自带的最小行为克隆（state→action 线性 ridge），用于把"
                "「网页→训练→产出模型→下载」这条路走通；不是 VLA 模型，也不吃图像。",
        "dataset_id": dataset_id, "dataset": info["dataset"], "dataset_version": info["version"],
        "content_hash": info["content_hash"], "train_split": train_split, "val_split": val_split,
        "split_salt": info["salt"], "train_episodes": train_idx, "metrics": metrics,
        "created_at": db.now(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    job.progress(5, 5, "登记模型")
    model_id = db.execute(
        "INSERT INTO models (name, backend, kind, dataset_id, split, val_split, split_salt,"
        " uses_vision, path, meta_path, bytes, metrics_json, job_id, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, "builtin", "linear-behavior-cloning", dataset_id, train_split, val_split,
         info["salt"], 0, str(model_path), str(meta_path), model_path.stat().st_size,
         json.dumps(metrics, ensure_ascii=False), job.id, db.now()))
    audit.log("INFO", "模型训练",
              f"网页训练完成（builtin）：模型 #{model_id} {name}，"
              f"train MSE {train_mse:.6f}"
              + (f" / val MSE {val_mse:.6f}" if val_mse is not None else "（无 val 划分）"),
              ref_type="model", ref_id=model_id, detail=metrics)
    return {"model_id": model_id, "name": name, "backend": "builtin", "path": str(model_path),
            "metrics": metrics, "dataset": info["dataset"], "train_split": train_split,
            "val_split": val_split, "train_episodes": train_idx}


# --------------------------------------------------------------------------- lerobot 训练

def _train_lerobot(job: jobs.JobContext, payload: dict) -> dict:
    """调 LeRobot 真训练脚本。

    几个必须遵守的约束（都来自 lerobot 源码，踩过才写下来）：
    - `dataset.repo_id` 必须是 `命名空间/名字` 形式（`configs/default.py: DatasetConfig.repo_id`）；
    - `output_dir` **必须不存在**，否则 `configs/train.py` 直接 raise 拒绝启动（resume=False 时）；
    - 只训**划分内**的 episode（`--dataset.episodes`），否则训练/评估会重叠；
    - 失败时把**进程输出的最后 40 行**带回错误信息，否则只看到"退出码 1"没法排查。
    """
    import collections

    if not LEROBOT_TRAIN.is_file():
        raise ValueError(f"找不到 LeRobot 训练脚本：{LEROBOT_TRAIN}")
    python = payload.get("python") or pick_env("lerobot")
    if not python:
        envs = detect_envs()
        detail = "；".join(
            f"{Path(e['python']).parent.parent.name}: torch={e['torch'] or '无'} "
            f"lerobot={e['lerobot'] or '无'}" for e in envs)
        raise ValueError(
            "没有可用的训练环境：需要一个同时装了 torch 与 lerobot 的 Python。"
            f"当前探测结果 —— {detail}。")

    dataset_id = int(payload["dataset_id"])
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError(f"数据集不存在：{dataset_id}")

    # repo_id：LeRobot 按 HF 仓库 id 解析，必须是 "<namespace>/<name>"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(row["name"])).strip("-").lower() or "dataset"
    repo_id = f"local/{safe}"

    # 只训划分内的 episode
    from . import splits as splits_mod

    overview = splits_mod.overview(dataset_id)
    split_name = payload.get("split") or "train"
    episodes = None
    if overview.get("assigned"):
        if split_name not in overview["counts"]:
            raise ValueError(f"数据集 #{dataset_id} 没有 '{split_name}' 划分"
                             f"（现有：{list(overview['counts'])}）")
        episodes = sorted(overview["splits"][split_name])
        if not episodes:
            raise ValueError(f"划分 '{split_name}' 是空的")

    steps = int(payload.get("steps") or 200)
    policy = payload.get("policy") or "act"
    batch_size_vla = 1 if (payload.get("policy") or "act") in ("smolvla", "pi0", "pi0fast") else None
    # VLA 策略（带语言）不能用 --policy.type 从零建，必须从官方预训练权重微调：
    VLA_POLICIES = {"smolvla": "lerobot/smolvla_base",
                    "pi0": "lerobot/pi0_base",
                    "pi0fast": "lerobot/pi0fast_base"}
    batch_size = int(payload.get("batch_size") or (batch_size_vla or 4))
    stamp = db.now().replace("-", "").replace(":", "").replace(" ", "_")
    runs_root = db.STORAGE / "train_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    out_dir = runs_root / f"{safe}_{policy}_{stamp}_{steps}steps"
    if out_dir.exists():                       # 绝不预建；万一撞名就换一个
        out_dir = runs_root / f"{safe}_{policy}_{stamp}_{steps}steps_b"

    command = [python, str(LEROBOT_TRAIN),
               f"--dataset.repo_id={repo_id}",
               f"--dataset.root={row['path']}",
               # VLA（smolvla/pi0/pi0fast）必须从官方预训练权重微调，不能用 type 从零建
               *([f"--policy.path={VLA_POLICIES[policy]}", "--policy.device=cuda"]
                 if policy in VLA_POLICIES else [f"--policy.type={policy}"]),
               f"--steps={steps}",
               f"--batch_size={batch_size}",
               f"--output_dir={out_dir}",
               f"--save_freq={max(steps, 1)}",
               "--wandb.enable=false",
               f"--policy.repo_id=local/{safe}-{policy}",   # lerobot 要求必填
               "--policy.push_to_hub=false",
               # torch DataLoader 多进程要用命名管道，本机沙箱禁止 → 必须单进程
               "--num_workers=0"]
    if episodes:
        command.append("--dataset.episodes=" + json.dumps(episodes, separators=(",", ":")))
    command.extend(payload.get("extra_args") or [])
    job.log("$ " + " ".join(command))
    job.log(f"训练集：{len(episodes) if episodes else '全部'} 条 episode（划分 {split_name}）")

    started = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, encoding="utf-8", errors="replace",
                               env=lerobot_env(), cwd=str(db.STORAGE))
    tail: collections.deque = collections.deque(maxlen=40)
    last = 0
    for line in process.stdout:            # 流式读日志
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        match = training_mod.METRIC_RE.search(line)
        if match:
            step = training_mod._num(match.group("step")) or 0
            if step != last:
                last = step
                loss = float(match.group("loss"))
                job.progress(min(99, int(step / max(steps, 1) * 100)), 100,
                             f"step {int(step)}/{steps} loss {loss:.4f}")
        elif any(k in line for k in ("Error", "Traceback", "error:", "Error:")):
            job.log(line[:300])
    code = process.wait()
    salvaged = ""
    if code != 0:
        # LeRobot 的 train.py 在**存完 checkpoint 之后**还会调 update_last_checkpoint()，
        # 那里用了 Python 3.12 才有的 Path.is_junction()，在 3.10 上会 AttributeError。
        # 只要 checkpoint 已经落盘，就不该把整次训练判为失败 —— 抢救并如实记录。
        salvaged = next((line for line in reversed(tail) if "Checkpoint policy after step" in line), "")
        has_ckpt = any(q.name == "model.safetensors" for q in out_dir.rglob("model.safetensors"))
        if not (salvaged and has_ckpt):
            raise ValueError(f"训练进程退出码 {code}。最后 {len(tail)} 行输出：\n"
                             + "\n".join(list(tail)[-40:]))
        job.log(f"⚠️ 训练进程以 {code} 退出，但 checkpoint 已保存（{salvaged.strip()}）—— "
                "按成功处理；原因是 lerobot 收尾调用了 Python 3.12 才有的 Path.is_junction()")

    checkpoints = sorted(out_dir.rglob("model.safetensors"), key=lambda q: q.stat().st_mtime)
    if not checkpoints:
        checkpoints = sorted(out_dir.rglob("*.safetensors"), key=lambda q: q.stat().st_mtime)
    if not checkpoints:
        raise ValueError(f"训练结束但没找到 checkpoint：{out_dir}\n最后输出：\n"
                         + "\n".join(list(tail)[-20:]))
    latest = checkpoints[-1]
    size = sum(f.stat().st_size for f in latest.parent.rglob("*") if f.is_file())
    metrics = {"steps": steps, "seconds": round(time.time() - started, 1), "policy": policy,
               "note": (salvaged.strip() if salvaged else None),
               "last_lines": [l for l in list(tail)[-6:] if "loss:" in l] or None,
               "batch_size": batch_size, "exit_code": code, "repo_id": repo_id,
               "train_episodes": episodes, "split": split_name,
               "train_frames": (db.scalar(
                   "SELECT COALESCE(SUM(data_rows), 0) AS c FROM episodes WHERE dataset_id=? AND"
                   " episode_index IN (" + ",".join("?" * len(episodes)) + ")",
                   tuple([dataset_id] + episodes)) if episodes else None)}
    model_id = db.execute(
        "INSERT INTO models (name, backend, kind, dataset_id, split, split_salt, uses_vision,"
        " path, meta_path, bytes, metrics_json, job_id, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (out_dir.name, "lerobot", policy, dataset_id, split_name,
         overview.get("salt") if overview.get("assigned") else None,
         1 if lerobot.video_keys(lerobot.load_meta(row["path"])["info"]) else 1,
         str(latest), None, size, json.dumps(metrics, ensure_ascii=False), job.id, db.now()))
    audit.log("INFO", "模型训练",
              f"网页训练完成（lerobot/{policy}）：模型 #{model_id} {out_dir.name}，"
              f"{steps} 步 / {metrics['seconds']}s / {size / 1024 / 1024:.1f}MB",
              ref_type="model", ref_id=model_id, detail=metrics)
    return {"model_id": model_id, "name": out_dir.name, "backend": "lerobot", "policy": policy,
            "checkpoint": str(latest), "output_dir": str(out_dir), "metrics": metrics,
            "tail": list(tail)[-10:]}


def models(limit: int = 50) -> list[dict]:
    rows = db.query("SELECT * FROM models ORDER BY id DESC LIMIT ?", (limit,))
    for row in rows:
        row["metrics"] = json.loads(row.pop("metrics_json") or "{}")
        ds = db.query_one("SELECT name FROM datasets WHERE id=?", (row["dataset_id"],)) \
            if row["dataset_id"] else None
        row["dataset"] = ds["name"] if ds else None
    return rows
