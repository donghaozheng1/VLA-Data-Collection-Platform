"""训练与离线评估：解析真实训练产物、登记实验与指标、收集 bad case 并回灌。

对接的是 LeRobot 训练产物（本机实测格式）：

    outputs/
      pusht_train_20k/
        checkpoints/<step>/pretrained_model/train_config.json   <- 干净 JSON：policy/dataset/optimizer/steps/seed...
        eval/videos_step_XXXXX/*.mp4                            <- 各步评估视频
      pusht_train_20k_log.txt                                   <- 训练日志：step/loss/grdn/lr/updt_s 序列
      pusht_eval/eval_info.json                                 <- 离线评估：per_episode + aggregated(pc_success...)
      eval/<日期>/<时刻>_<policy>/eval_info.json

设计原则：**只解析真实文件，不编造指标**。没有产物的实验就是"已登记但无指标"。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import audit, db, lerobot

# 实测日志行：step:200 smpl:2K ep:13 epch:0.06 loss:0.471 grdn:9.549 lr:2.0e-05 updt_s:0.192
METRIC_RE = re.compile(
    r"step:(?P<step>\S+)\s+smpl:(?P<samples>\S+)\s+ep:(?P<episodes>\d+)\s+epch:(?P<epoch>[\d.]+)"
    r"\s+loss:(?P<loss>[-\d.eE+]+)\s+grdn:(?P<grad_norm>[-\d.eE+]+)\s+lr:(?P<lr>[-\d.eE+]+)"
    r"\s+updt_s:(?P<updt_s>[-\d.eE+]+)")
CHECKPOINT_GLOB = "checkpoints/*/pretrained_model/train_config.json"


def _num(text) -> float | None:
    """把 2K / 1.5M / 200 解析成数字。"""
    if text is None:
        return None
    token = str(text).strip().rstrip(",")
    if not token:
        return None
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(token[-1:].upper())
    try:
        return float(token[:-1]) * mult if mult else float(token)
    except ValueError:
        return None


# --------------------------------------------------------------------------- 解析

def parse_train_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_train_log(path: str | Path, limit: int | None = None) -> dict:
    """解析训练日志，返回 {series, first_line, last_line, lines}。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    series: list[dict] = []
    for match in METRIC_RE.finditer(text):
        row = match.groupdict()
        point = {
            "step": _num(row["step"]),
            "samples": _num(row["samples"]),
            "episodes": int(row["episodes"]),
            "epoch": float(row["epoch"]),
            "loss": float(row["loss"]),
            "grad_norm": float(row["grad_norm"]),
            "lr": float(row["lr"]),
            "updt_s": float(row["updt_s"]),
        }
        series.append(point)
    if limit and len(series) > limit:
        # 均匀抽稀，保留首尾与趋势
        stride = max(1, len(series) // limit)
        series = series[::stride]
    return {"series": series, "points": len(series),
            "last": series[-1] if series else None,
            "first": series[0] if series else None}


def parse_eval_info(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    aggregated = data.get("aggregated") or {}
    return {
        "per_episode": data.get("per_episode") or [],
        "video_paths": data.get("video_paths") or [],
        "num_episodes": len(data.get("per_episode") or []),
        "pc_success": aggregated.get("pc_success"),
        "avg_sum_reward": aggregated.get("avg_sum_reward"),
        "avg_max_reward": aggregated.get("avg_max_reward"),
        "eval_s": aggregated.get("eval_s"),
        "eval_ep_s": aggregated.get("eval_ep_s"),
    }


def _walk_files(root: Path) -> list[Path]:
    """健壮遍历：跳过坏软链与无权限目录。

    LeRobot 的 `checkpoints/last` 是指向最新步数的**符号链接**，在 Windows 上常是断链，
    `Path.rglob()` 遇到它会直接抛 FileNotFoundError（3.10 的 rglob 没有 onerror 参数）。
    """
    import os

    found: list[Path] = []
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    found.append(Path(entry.path))
            except OSError:
                continue
    return sorted(found)


def _is_checkpoint_config(path: Path) -> bool:
    """<run>/checkpoints/<step>/pretrained_model/train_config.json"""
    return (path.name == "train_config.json" and path.parent.name == "pretrained_model"
            and path.parent.parent.parent.name == "checkpoints")


def find_runs(root: str | Path) -> list[dict]:
    """在 root 下找训练产物：含 checkpoint 配置的目录 + 训练日志 + 评估结果。"""
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    runs: dict[str, dict] = {}

    def entry_for(key: str) -> dict:
        return runs.setdefault(key, {"run_dir": key, "checkpoints": [], "logs": [], "evals": []})

    for path in _walk_files(root_path):
        if _is_checkpoint_config(path):
            run_dir = path.parent.parent.parent.parent
            entry_for(str(run_dir))["checkpoints"].append(
                {"step": path.parent.parent.name, "config": str(path)})
        elif path.name == "eval_info.json":
            entry_for("__evals__")["evals"].append(str(path))
        elif path.name.endswith("_log.txt") and path.parent == root_path:
            base = path.stem.replace("_resume", "").replace("_log", "")
            entry_for(str(root_path / base))["logs"].append(str(path))

    ordered = [v for k, v in runs.items() if k != "__evals__"]
    if "__evals__" in runs:
        ordered.append(runs["__evals__"])
    return ordered


def scan(root: str | Path) -> dict:
    """列出产物概览（不写库），供界面预览。"""
    out = []
    for run in find_runs(root):
        item = {"run_dir": run["run_dir"], "checkpoints": len(run["checkpoints"]),
                "logs": [Path(p).name for p in run["logs"]], "evals": len(run["evals"])}
        configs = sorted(run["checkpoints"], key=lambda c: _num(c["step"]) or 0)
        if configs:
            config = parse_train_config(configs[-1]["config"])
            item.update({
                "policy": (config.get("policy") or {}).get("type"),
                "dataset_repo_id": (config.get("dataset") or {}).get("repo_id"),
                "steps": config.get("steps"), "batch_size": config.get("batch_size"),
                "lr": (config.get("optimizer") or {}).get("lr"),
                "seed": config.get("seed"),
            })
        if run["logs"]:
            parsed = parse_train_log(run["logs"][-1])
            item["log_points"] = parsed["points"]
            item["last_metric"] = parsed["last"]
        out.append(item)
    return {"root": str(root), "runs": out}


# --------------------------------------------------------------------------- 登记

def _match_dataset(repo_id: str | None, dataset_id: int | None) -> dict | None:
    if dataset_id:
        return db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not repo_id:
        return None
    key = str(repo_id).split("/")[-1].lower()
    rows = db.query("SELECT * FROM datasets ORDER BY id")
    for row in rows:
        if key and (key in str(row["name"]).lower() or key in str(row["path"]).lower()):
            return row
    return None


def register_experiment(*, run_dir: str | None = None, log_path: str | None = None,
                        dataset_id: int | None = None, name: str | None = None,
                        note: str | None = None, split: str | None = None) -> dict:
    """登记一个训练实验：解析 checkpoint 配置 + 训练日志指标，并绑定数据集版本与哈希。"""
    run_path = Path(run_dir) if run_dir else None
    if run_path is None and log_path:
        run_path = Path(log_path).parent / Path(log_path).stem.replace("_log", "").replace("_resume", "")
    if run_path is None or not run_path.exists():
        raise ValueError(f"训练产物目录不存在：{run_path}")

    configs = [p for p in _walk_files(run_path) if _is_checkpoint_config(p)]
    configs.sort(key=lambda p: _num(p.parent.parent.name) or 0)
    config = parse_train_config(configs[-1]) if configs else {}
    logs = []
    if run_path.parent.is_dir():
        try:
            logs = sorted(p for p in run_path.parent.iterdir()
                          if p.is_file() and p.name.startswith(run_path.name)
                          and p.name.endswith("_log.txt"))
        except OSError:
            logs = []
    if not logs and log_path:
        logs = [Path(log_path)]
    log_file = str(logs[-1]) if logs else None

    policy = (config.get("policy") or {}).get("type")
    repo_id = (config.get("dataset") or {}).get("repo_id")
    steps = config.get("steps")
    experiment_name = name or f"{run_path.name}" + (f" ({policy})" if policy else "")

    dataset = _match_dataset(repo_id, dataset_id)
    # 绑定训练所用的数据划分（若该数据集做过划分）
    split_info = None
    if dataset:
        from . import splits as splits_mod

        overview = splits_mod.overview(dataset["id"])
        if overview.get("assigned"):
            chosen = split or "train"
            if chosen not in overview["counts"]:
                raise ValueError(f"数据集 #{dataset['id']} 没有 '{chosen}' 划分"
                                 f"（现有：{list(overview['counts'])}）")
            split_info = {"split": chosen, "salt": overview["salt"], "mode": overview["mode"],
                          "counts": overview["counts"],
                          "episodes": overview["splits"].get(chosen, [])}

    reproduce = None
    if config:
        reproduce = (
            f"python -m lerobot.scripts.train --policy={policy} "
            f"--dataset.repo_id={repo_id} --steps={steps} "
            f"--batch_size={config.get('batch_size')} "
            f"--optimizer.lr={(config.get('optimizer') or {}).get('lr')} "
            f"--seed={config.get('seed')} --output_dir={run_path}")

    if split_info:
        episodes = split_info["episodes"]
        split_key = split_info["split"]
        listing = (json.dumps(episodes) if len(episodes) <= 32
                   else f'"<见 meta/splits.json 的 splits.{split_key}，{len(episodes)} 条>"')
        reproduce = (f"{reproduce}\n# 训练集：dataset #{dataset['id']} 的 {split_key} 划分"
                     f"（{len(episodes)}/{sum(split_info['counts'].values())} 条，"
                     f"salt={split_info['salt']}, mode={split_info['mode']}）\n"
                     f"# --dataset.episodes={listing}")
    experiment_id = db.execute(
        """INSERT INTO experiments (name, run_dir, output_dir, policy, dataset_repo_id, dataset_id,
                dataset_version, dataset_hash, steps, batch_size, learning_rate, seed, save_freq,
                eval_freq, config_json, log_path, status, reproduce_cmd, split, split_salt,
                note, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'registered', ?,?,?,?,?)
           ON CONFLICT(run_dir, steps) DO UPDATE SET
                name=excluded.name, policy=excluded.policy, dataset_repo_id=excluded.dataset_repo_id,
                dataset_id=excluded.dataset_id, dataset_version=excluded.dataset_version,
                dataset_hash=excluded.dataset_hash, batch_size=excluded.batch_size,
                learning_rate=excluded.learning_rate, seed=excluded.seed,
                config_json=excluded.config_json, log_path=excluded.log_path,
                reproduce_cmd=excluded.reproduce_cmd, split=excluded.split,
                split_salt=excluded.split_salt, note=excluded.note""",
        (experiment_name, str(run_path), config.get("output_dir"), policy, repo_id,
         dataset["id"] if dataset else None, dataset["version"] if dataset else None,
         dataset["content_hash"] if dataset else None, steps, config.get("batch_size"),
         (config.get("optimizer") or {}).get("lr"), config.get("seed"), config.get("save_freq"),
         config.get("eval_freq"), json.dumps(config, ensure_ascii=False) if config else None,
         log_file, reproduce,
         split_info["split"] if split_info else None,
         split_info["salt"] if split_info else None, note, db.now()))
    row = db.query_one("SELECT id FROM experiments WHERE run_dir=? AND steps IS ?",
                       (str(run_path), steps))
    experiment_id = row["id"]

    metrics = 0
    if log_file:
        parsed = parse_train_log(log_file)
        db.execute("DELETE FROM experiment_metrics WHERE experiment_id=?", (experiment_id,))
        db.execute_many(
            "INSERT INTO experiment_metrics (experiment_id, step, loss, grad_norm, lr, epoch,"
            " episodes, samples, updt_s) VALUES (?,?,?,?,?,?,?,?,?)",
            [(experiment_id, p["step"], p["loss"], p["grad_norm"], p["lr"], p["epoch"],
              p["episodes"], p["samples"], p["updt_s"]) for p in parsed["series"]])
        metrics = len(parsed["series"])
        db.execute("UPDATE experiments SET metrics_count=? WHERE id=?", (metrics, experiment_id))

    evaluations = []
    search_roots = {run_path, run_path.parent}
    for base in search_roots:
        for info_file in [p for p in _walk_files(base) if p.name == "eval_info.json"]:
            evaluations.append(register_evaluation(info_file, experiment_id=experiment_id))

    audit.log("INFO", "模型训练",
              f"登记实验 #{experiment_id} {experiment_name}：policy={policy} steps={steps} "
              f"指标点={metrics} 数据集="
              f"{('#' + str(dataset['id']) + ' v' + str(dataset['version'])) if dataset else '未匹配'}",
              ref_type="experiment", ref_id=experiment_id,
              detail={"policy": policy, "steps": steps, "metrics": metrics,
                      "dataset_id": dataset["id"] if dataset else None})
    return {"experiment_id": experiment_id, "name": experiment_name, "policy": policy,
            "steps": steps, "metrics": metrics,
            "split": split_info["split"] if split_info else None,
            "split_salt": split_info["salt"] if split_info else None,
            "split_episodes": len(split_info["episodes"]) if split_info else None,
            "dataset": {"id": dataset["id"], "name": dataset["name"], "version": dataset["version"],
                        "content_hash": dataset["content_hash"]} if dataset else None,
            "evaluations": evaluations, "reproduce_cmd": reproduce}


def register_evaluation(info_path: str | Path, *, experiment_id: int | None = None) -> dict:
    """登记一份 eval_info.json（离线评估结果 + 逐 episode 结果 + bad case）。"""
    info_file = Path(info_path)
    parsed = parse_eval_info(info_file)
    step = None
    for part in reversed(info_file.parts):
        if part.startswith("videos_step_"):
            step = int(_num(part.replace("videos_step_", "")) or 0)
            break
    name = info_file.parent.name
    eval_dir = str(info_file.parent)
    db.execute(
        """INSERT INTO evaluations (experiment_id, name, eval_dir, step, num_episodes, pc_success,
                avg_sum_reward, avg_max_reward, eval_s, eval_ep_s, videos_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(eval_dir) DO UPDATE SET
                experiment_id=excluded.experiment_id, num_episodes=excluded.num_episodes,
                pc_success=excluded.pc_success, avg_sum_reward=excluded.avg_sum_reward,
                avg_max_reward=excluded.avg_max_reward, videos_json=excluded.videos_json""",
        (experiment_id, name, eval_dir, step, parsed["num_episodes"], parsed["pc_success"],
         parsed["avg_sum_reward"], parsed["avg_max_reward"], parsed["eval_s"], parsed["eval_ep_s"],
         json.dumps(parsed["video_paths"], ensure_ascii=False), db.now()))
    row = db.query_one("SELECT id FROM evaluations WHERE eval_dir=?", (eval_dir,))
    evaluation_id = row["id"]

    db.execute("DELETE FROM eval_cases WHERE evaluation_id=?", (evaluation_id,))
    db.execute_many(
        "INSERT INTO eval_cases (evaluation_id, episode_ix, seed, sum_reward, max_reward, success)"
        " VALUES (?,?,?,?,?,?)",
        [(evaluation_id, c.get("episode_ix"), c.get("seed"), c.get("sum_reward"),
          c.get("max_reward"), 1 if c.get("success") else 0) for c in parsed["per_episode"]])

    # 失败的评估 episode -> bad case（离线评估的证据）
    db.execute("DELETE FROM bad_cases WHERE source='eval' AND evaluation_id=?", (evaluation_id,))
    failed = [c for c in parsed["per_episode"] if not c.get("success")]
    db.execute_many(
        "INSERT INTO bad_cases (source, evaluation_id, eval_episode_ix, reason, detail, status,"
        " created_at) VALUES ('eval', ?,?,?,?, 'open', ?)",
        [(evaluation_id, c.get("episode_ix"), "离线评估未成功",
          f"sum_reward={c.get('sum_reward'):.3f} max_reward={c.get('max_reward'):.3f} seed={c.get('seed')}",
          db.now()) for c in failed])

    return {"evaluation_id": evaluation_id, "name": name, "num_episodes": parsed["num_episodes"],
            "pc_success": parsed["pc_success"], "failed_cases": len(failed), "step": step}


# --------------------------------------------------------------------------- bad case 收集与回灌

def collect_bad_cases(*, dataset_id: int | None = None, include_qc: bool = True,
                      include_annotation: bool = True) -> dict:
    """从质检不通过 / 标注失败 里收集 bad case（评估类的在登记评估时已写入）。"""
    added = {"qc": 0, "annotation": 0}
    datasets = ([db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))]
                if dataset_id else db.query("SELECT * FROM datasets ORDER BY id"))
    for ds in datasets:
        if not ds:
            continue
        if include_qc:
            run = db.query_one("SELECT * FROM qc_runs WHERE dataset_id=? ORDER BY id DESC LIMIT 1",
                               (ds["id"],))
            if run:
                rows = db.query(
                    "SELECT DISTINCT episode_index FROM qc_findings WHERE run_id=? AND severity='error'",
                    (run["id"],))
                for row in rows:
                    exists = db.query_one(
                        "SELECT id FROM bad_cases WHERE source='qc' AND dataset_id=? AND episode_index=?",
                        (ds["id"], row["episode_index"]))
                    if not exists:
                        db.execute(
                            "INSERT INTO bad_cases (source, dataset_id, episode_index, reason, detail,"
                            " status, created_at) VALUES ('qc', ?,?,?,?, 'open', ?)",
                            (ds["id"], row["episode_index"], "质检不通过",
                             f"来自质检 run{run['id']}", db.now()))
                        added["qc"] += 1
        if include_annotation:
            rows = db.query("SELECT episode_index FROM annotations WHERE dataset_id=? AND success=0",
                            (ds["id"],))
            for row in rows:
                exists = db.query_one(
                    "SELECT id FROM bad_cases WHERE source='annotation' AND dataset_id=?"
                    " AND episode_index=?", (ds["id"], row["episode_index"]))
                if not exists:
                    db.execute(
                        "INSERT INTO bad_cases (source, dataset_id, episode_index, reason, detail,"
                        " status, created_at) VALUES ('annotation', ?,?,?,?, 'open', ?)",
                        (ds["id"], row["episode_index"], "标注为失败样本", "来自主标注 success=0",
                         db.now()))
                    added["annotation"] += 1
    return {**added, "total_open": db.scalar("SELECT COUNT(*) AS c FROM bad_cases WHERE status='open'")}


def bad_cases(status: str | None = None, dataset_id: int | None = None, limit: int = 200) -> list[dict]:
    sql = "SELECT * FROM bad_cases WHERE 1=1"
    params: list = []
    if status:
        sql += " AND status=?"
        params.append(status)
    if dataset_id:
        sql += " AND dataset_id=?"
        params.append(dataset_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = db.query(sql, tuple(params))
    for row in rows:
        if row.get("dataset_id"):
            ds = db.query_one("SELECT name FROM datasets WHERE id=?", (row["dataset_id"],))
            row["dataset_name"] = ds["name"] if ds else None
    return rows


def return_to_hard_set(*, dataset_id: int, name: str | None = None,
                       sources: list[str] | None = None) -> dict:
    """把该数据集里未处理的 bad case 汇成一个「困难集」数据集（真实导出 + 打标回灌）。"""
    import shutil

    from . import pipeline

    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError(f"数据集不存在：{dataset_id}")
    sql = "SELECT * FROM bad_cases WHERE dataset_id=? AND status='open'"
    params: list = [dataset_id]
    if sources:
        sql += f" AND source IN ({','.join('?' * len(sources))})"
        params.extend(sources)
    cases = db.query(sql, tuple(params))
    episodes = sorted({c["episode_index"] for c in cases if c["episode_index"] is not None})
    if not episodes:
        raise ValueError("没有可回灌的 bad case（该数据集下没有未处理的失败 episode）")

    hard_name = name or f"{row['name']}_hardcases"
    out_dir = db.EXPORT_DIR / hard_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    meta = lerobot.load_meta(row["path"])
    annotations = {a["episode_index"]: a for a in
                   db.query("SELECT * FROM annotations WHERE dataset_id=?", (dataset_id,))}
    result = lerobot.export_subset(row["path"], meta, episodes, out_dir, hard_name,
                                  annotations=annotations)
    zip_path = pipeline._zip_dataset(out_dir)
    artifact_id = db.execute(
        "INSERT INTO artifacts (job_id, dataset_id, kind, path, bytes, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (None, dataset_id, "hard_set", str(zip_path), zip_path.stat().st_size, db.now()))

    current = db.query_one("SELECT version FROM datasets WHERE id=?", (dataset_id,)) or {}
    pipeline.ingest(_NullJob(), {
        "path": str(out_dir), "name": hard_name,
        "with_video": bool(lerobot.video_keys(meta["info"])),
        "origin": {"kind": "hard_set", "derived_by": "badcase:return",
                   "parent_dataset_id": dataset_id, "parent_version": current.get("version"),
                   "params": {"episodes": len(episodes),
                              "sources": sorted({c["source"] for c in cases})}},
    })
    derived = db.query_one("SELECT id, version, content_hash FROM datasets WHERE path=?",
                           (str(out_dir),))

    db.execute("UPDATE bad_cases SET status='returned', handled_at=?"
               " WHERE dataset_id=? AND status='open'", (db.now(), dataset_id))
    audit.log("INFO", "模型训练",
              f"bad case 回灌：数据集 #{dataset_id} -> 困难集 #{derived['id'] if derived else '?'} "
              f"（{len(episodes)} 条 episode，来源 {'+'.join(sorted({c['source'] for c in cases}))}）",
              ref_type="dataset", ref_id=dataset_id,
              detail={"episodes": episodes, "sources": sorted({c["source"] for c in cases})})
    return {"hard_set_dataset_id": derived["id"] if derived else None,
            "hard_set_version": derived["version"] if derived else None,
            "episodes": episodes, "episodes_count": len(episodes),
            "cases": len(cases), "sources": sorted({c["source"] for c in cases}),
            "path": str(out_dir), "zip": str(zip_path), "artifact_id": artifact_id,
            "frames": result["frames"]}


class _NullJob:
    """给 pipeline.ingest 用的空任务句柄（回灌不是后台任务）。"""

    id = None

    def progress(self, *args, **kwargs) -> None:  # noqa: D401
        return None

    def log(self, *args, **kwargs) -> None:
        return None


def compare(dataset_id: int) -> dict:
    """同一数据集上的实验能不能直接比指标？

    只有当**数据集内容哈希相同**且**训练划分（split + salt）相同**时，指标才可以直接比较；
    否则必须先说明差异。这里把这个判断显式做出来，而不是让人凭印象比数字。
    """
    rows = db.query("SELECT * FROM experiments WHERE dataset_id=? ORDER BY id", (dataset_id,))
    ds = db.query_one("SELECT name, version, content_hash FROM datasets WHERE id=?", (dataset_id,))
    groups: dict[tuple, list] = {}
    for row in rows:
        key = (row.get("dataset_hash"), row.get("split"), row.get("split_salt"))
        groups.setdefault(key, []).append({
            "experiment_id": row["id"], "name": row["name"], "steps": row["steps"],
            "policy": row["policy"], "dataset_version": row["dataset_version"],
            "split": row.get("split"), "split_salt": row.get("split_salt"),
            "metrics_count": row["metrics_count"],
            "evaluations": db.query(
                "SELECT id, name, num_episodes, pc_success FROM evaluations WHERE experiment_id=?",
                (row["id"],)),
        })
    comparable = [{"dataset_hash": k[0], "split": k[1], "split_salt": k[2], "experiments": v}
                  for k, v in groups.items() if len(v) > 1]
    return {
        "dataset_id": dataset_id, "dataset": ds, "experiments": len(rows),
        "comparable_groups": comparable,
        "verdict": ("只有 1 个实验，暂无可比性判断" if len(rows) < 2
                    else (f"有 {len(comparable)} 组实验可直接比指标"
                          if comparable else
                          "这些实验的数据集哈希或划分不同，指标不可直接比较")),
        "note": "离线评估（eval_info.json）是**环境 rollout**（内容是 seed/reward，不是数据集 episode），"
                "所以数据集划分与它不存在重叠问题；划分的作用是训练/验证/测试隔离与结果可复现。",
    }
