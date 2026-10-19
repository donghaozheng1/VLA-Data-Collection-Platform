"""训练与离线评估接口（/api/training/*、/api/badcases/*）。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from core import db, jobs, trainer, training

api_router = APIRouter()

DEFAULT_OUTPUT_ROOT = r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\outputs"


@api_router.get("/training/default-root")
def default_root():
    return {"root": DEFAULT_OUTPUT_ROOT, "exists": Path(DEFAULT_OUTPUT_ROOT).is_dir()}


@api_router.post("/training/scan")
def do_scan(payload: dict = Body(default={})):
    root = payload.get("root") or DEFAULT_OUTPUT_ROOT
    try:
        return training.scan(root)
    except OSError as exc:
        raise HTTPException(400, f"扫描失败：{exc}") from exc


@api_router.post("/training/register")
def do_register(payload: dict = Body(default={})):
    run_dir = payload.get("run_dir")
    log_path = payload.get("log_path")
    if run_dir == "__evals__":
        raise HTTPException(400, "该条目只是评估结果集合，请直接登记具体训练目录")
    if not run_dir and not log_path:
        raise HTTPException(400, "需要 run_dir 或 log_path")
    try:
        return training.register_experiment(run_dir=run_dir, log_path=log_path,
                                           dataset_id=payload.get("dataset_id"),
                                           name=payload.get("name"), note=payload.get("note"),
                                           split=payload.get("split"))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@api_router.get("/training/experiments")
def list_experiments(limit: int = 50):
    rows = db.query("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,))
    for row in rows:
        row.pop("config_json", None)
        row["evals"] = db.query(
            "SELECT id, name, num_episodes, pc_success, avg_max_reward FROM evaluations"
            " WHERE experiment_id=? ORDER BY id", (row["id"],))
    return rows


@api_router.get("/training/experiments/{experiment_id}")
def experiment_detail(experiment_id: int):
    row = db.query_one("SELECT * FROM experiments WHERE id=?", (experiment_id,))
    if not row:
        raise HTTPException(404, "实验不存在")
    row["config"] = json.loads(row.pop("config_json") or "{}")
    row["evaluations"] = db.query("SELECT * FROM evaluations WHERE experiment_id=? ORDER BY id",
                                  (experiment_id,))
    row["metrics_summary"] = {
        "points": db.scalar("SELECT COUNT(*) AS c FROM experiment_metrics WHERE experiment_id=?",
                            (experiment_id,)),
        "first": db.query_one("SELECT * FROM experiment_metrics WHERE experiment_id=? ORDER BY step LIMIT 1",
                              (experiment_id,)),
        "last": db.query_one("SELECT * FROM experiment_metrics WHERE experiment_id=? ORDER BY step DESC LIMIT 1",
                             (experiment_id,)),
        "best_loss": db.scalar("SELECT MIN(loss) AS c FROM experiment_metrics WHERE experiment_id=?",
                               (experiment_id,)),
    }
    if row.get("dataset_id"):
        ds = db.query_one("SELECT name, version, content_hash FROM datasets WHERE id=?",
                          (row["dataset_id"],))
        row["dataset"] = ds
    return row


@api_router.get("/training/experiments/{experiment_id}/metrics")
def experiment_metrics(experiment_id: int, limit: int = 120):
    rows = db.query("SELECT * FROM experiment_metrics WHERE experiment_id=? ORDER BY step",
                    (experiment_id,))
    if limit and len(rows) > limit:
        stride = max(1, len(rows) // limit)
        rows = rows[::stride]
    return rows


@api_router.get("/training/compare/{dataset_id}")
def compare_experiments(dataset_id: int):
    return training.compare(dataset_id)


@api_router.get("/training/evaluations")
def list_evaluations(limit: int = 50):
    rows = db.query("SELECT * FROM evaluations ORDER BY id DESC LIMIT ?", (limit,))
    for row in rows:
        row["videos"] = len(json.loads(row.get("videos_json") or "[]"))
        row.pop("videos_json", None)
    return rows


@api_router.get("/training/evaluations/{evaluation_id}")
def evaluation_detail(evaluation_id: int):
    row = db.query_one("SELECT * FROM evaluations WHERE id=?", (evaluation_id,))
    if not row:
        raise HTTPException(404, "评估不存在")
    row["video_paths"] = json.loads(row.pop("videos_json") or "[]")
    row["cases"] = db.query("SELECT * FROM eval_cases WHERE evaluation_id=? ORDER BY episode_ix",
                            (evaluation_id,))
    return row


@api_router.get("/badcases")
def list_bad_cases(status: str | None = None, dataset_id: int | None = None, limit: int = 300):
    return {"cases": training.bad_cases(status, dataset_id, limit),
            "by_status": db.query("SELECT status, COUNT(*) AS c FROM bad_cases GROUP BY status"),
            "by_source": db.query("SELECT source, COUNT(*) AS c FROM bad_cases GROUP BY source")}


@api_router.post("/badcases/collect")
def do_collect(payload: dict = Body(default={})):
    return training.collect_bad_cases(dataset_id=payload.get("dataset_id"),
                                      include_qc=payload.get("include_qc", True),
                                      include_annotation=payload.get("include_annotation", True))


@api_router.post("/badcases/return")
def do_return(payload: dict = Body(default={})):
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(400, "缺少 dataset_id")
    try:
        return training.return_to_hard_set(dataset_id=int(dataset_id),
                                          name=payload.get("name"),
                                          sources=payload.get("sources"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api_router.get("/training/envs")
def training_envs():
    """本机哪些 Python 能跑训练（实测 torch / lerobot）。"""
    envs = trainer.detect_envs()
    return {"envs": envs,
            "builtin_available": any("builtin" in e["usable_for"] for e in envs),
            "lerobot_available": any("lerobot" in e["usable_for"] for e in envs),
            "lerobot_script": str(trainer.LEROBOT_TRAIN),
            "private_deps": str(trainer.LEROBOT_DEPS),
            "private_deps_present": trainer.LEROBOT_DEPS.is_dir(),
            "train_python": str(trainer.lerobot_python() or ""),
            "lerobot_script_exists": trainer.LEROBOT_TRAIN.is_file(),
            "note": "lerobot 训练需要同时装了 torch 与 lerobot 的 Python；平台不会替你装包。"}


@api_router.post("/training/run")
def training_run(payload: dict = Body(default={})):
    """在网页里发起训练（builtin = 平台自带最小行为克隆；lerobot = 真训练）。"""
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(400, "缺少 dataset_id")
    split = payload.get("split") or "train"
    backend = payload.get("backend") or "builtin"
    if backend == "builtin":
        from core import splits as splits_mod

        overview = splits_mod.overview(int(dataset_id))
        if not overview.get("assigned"):
            raise HTTPException(400, "该数据集还没做数据划分 —— 先去「数据集导出」页做一次划分"
                                     "（训练/评估必须隔离）")
        if split not in overview["counts"]:
            raise HTTPException(400, f"没有 '{split}' 划分（现有：{list(overview['counts'])}）")
    job_id = jobs.submit("train", {**payload, "backend": backend},
                         target_type="dataset", target_id=int(dataset_id))
    return {"job_id": job_id, "backend": backend, "split": split}


@api_router.get("/models")
def list_models(limit: int = 50):
    return trainer.models(limit)
