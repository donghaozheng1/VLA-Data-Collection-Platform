"""四个真实业务环节的任务处理器：接入 / 预处理 / 质检 / 导出。

原则：每一步的产出都是**从真实数据算出来的**，不是写死的；
外部数据集只读，所有产物落在 storage/ 下。
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from . import (audit, db, dedup, jobs, lerobot, qc_checks, remote_qc,  # noqa: F401
               spatial, trainer)  # noqa: F401  trainer 注册 train 处理器；remote_qc 注册 qc_remote

# 默认扫描根（采集页面用）
DEFAULT_SCAN_ROOTS = [
    r"C:\Users\ll\Desktop\VLA",
    r"C:\Users\ll\Desktop\HKC",
    r"C:\Users\ll\Desktop\dhz",
    # 平台自己转换出来的数据集也在这里，扫描时应显示为「已接入」
    str(db.STORAGE / "datasets"),
]


# --------------------------------------------------------------------------- 通用

def _dataset_row(dataset_id: int) -> dict:
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError(f"数据集不存在：id={dataset_id}")
    if not Path(row["path"]).is_dir():
        raise FileNotFoundError(f"数据集路径已不存在：{row['path']}")
    return row


def _dir_bytes(path: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def _annotations(dataset_id: int) -> dict:
    """取该数据集的全部标注：{episode_index: {task_text, success, ...}}，供导出时覆盖源数据。"""
    return {a["episode_index"]: a for a in
            db.query("SELECT * FROM annotations WHERE dataset_id=?", (dataset_id,))}


def _register_version(dataset_id: int, kind: str, path, *, episodes=None, frames=None,
                      zip_path=None, params=None, parent_dataset_id=None, parent_version=None,
                      note=None) -> dict:
    """登记一个数据集版本。

    - 内容指纹（meta/ + data/ 按内容，videos/ 按路径+大小）与上一版相同则**不新增版本**（幂等）
    - 只有内容真的变了才 version+1，这样"版本"才有意义
    """
    fingerprint = lerobot.dataset_fingerprint(path)
    latest = db.query_one(
        "SELECT id, version, content_hash FROM dataset_versions WHERE dataset_id=?"
        " ORDER BY version DESC LIMIT 1", (dataset_id,))

    # 幂等：**同一操作(kind) + 同样内容(hash)** 已登记过就不再新增版本。
    # 不能只和"最新一版"比：接入(源目录哈希) 与 导出(快照哈希) 会交替出现，
    # 只比最新版会每轮凭空多出一个假版本。
    same = db.query_one(
        "SELECT id, version FROM dataset_versions WHERE dataset_id=? AND kind=? AND content_hash=?",
        (dataset_id, kind, fingerprint["sha256"]))
    if same:
        db.execute("UPDATE datasets SET version=?, content_hash=? WHERE id=?",
                   (latest["version"], latest["content_hash"], dataset_id))
        return {"version": latest["version"], "version_id": latest["id"],
                "content_hash": latest["content_hash"], "deduplicated": True,
                "bytes": fingerprint["bytes"]}

    version = (latest["version"] + 1) if latest else 1
    version_id = db.execute(
        "INSERT INTO dataset_versions (dataset_id, version, kind, content_hash, episodes, frames,"
        " bytes, path, zip_path, params_json, parent_dataset_id, parent_version, note, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (dataset_id, version, kind, fingerprint["sha256"], episodes, frames, fingerprint["bytes"],
         str(path), zip_path, json.dumps(params or {}, ensure_ascii=False, default=str),
         parent_dataset_id, parent_version, note, db.now()))
    db.execute("UPDATE datasets SET version=?, content_hash=? WHERE id=?",
               (version, fingerprint["sha256"], dataset_id))
    audit.log("INFO", "版本管理",
              f"数据集 #{dataset_id} 登记 v{version}（{kind}）"
              f"{'，父版本 v' + str(parent_version) if parent_version else ''}",
              ref_type="dataset", ref_id=dataset_id,
              detail={"version": version, "kind": kind, "hash": fingerprint["sha256"][:16],
                      "episodes": episodes, "frames": frames})
    return {"version": version, "version_id": version_id, "content_hash": fingerprint["sha256"],
            "deduplicated": False, "bytes": fingerprint["bytes"]}


def discover(roots: list[str] | None = None, max_depth: int = 6) -> list[dict]:
    """扫描磁盘，找出所有 LeRobot 数据集（带概览）。"""
    out: list[dict] = []
    for root in (roots or DEFAULT_SCAN_ROOTS):
        for path in lerobot.find_datasets(root, max_depth=max_depth):
            try:
                meta = lerobot.load_meta(path)
                info = meta["info"]
            except Exception as exc:  # noqa: BLE001
                out.append({"path": path, "name": Path(path).name, "error": str(exc)})
                continue
            registered = db.query_one("SELECT id FROM datasets WHERE path=?", (path,))
            out.append({
                "path": path,
                "name": Path(path).name,
                "registered_id": registered["id"] if registered else None,
                "codebase_version": info.get("codebase_version"),
                "robot_type": info.get("robot_type"),
                "fps": info.get("fps"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "tasks": list(meta["tasks"].values()),
                "video_keys": lerobot.video_keys(info),
            })
    return out


# --------------------------------------------------------------------------- ① 采集接入

@jobs.handler("ingest")
def ingest(job: jobs.JobContext, payload: dict) -> dict:
    """把一个真实 LeRobot 数据集接入平台：解析元信息 + 逐条探测 + 落库。"""
    path = str(payload["path"])
    name = payload.get("name") or Path(path).name
    with_video = bool(payload.get("with_video", True))

    meta = lerobot.load_meta(path)
    info = meta["info"]
    features = info.get("features") or {}
    codec = None
    image_shape = None
    for key in lerobot.video_keys(info):
        vinfo = (features.get(key) or {}).get("video_info") or {}
        codec = vinfo.get("video.codec") or codec
        shape = (features.get(key) or {}).get("shape")
        image_shape = "x".join(str(s) for s in shape) if shape else image_shape
    size = _dir_bytes(path)
    now = db.now()

    job.progress(1, 10, f"已读取 {Path(path).name} 元信息")

    # 按 path 去重：同一个目录重复点「接入」只刷新，不再新建一条
    # （否则换个名字再接入一次就会多出一份重复数据集）
    existing = (db.query_one("SELECT id, name FROM datasets WHERE path=?", (path,))
                or db.query_one("SELECT id, name FROM datasets WHERE name=?", (name,)))
    values = (path, "lerobot", info.get("codebase_version"), info.get("robot_type"),
              info.get("fps"), info.get("total_episodes"), info.get("total_frames"),
              info.get("total_tasks"), info.get("total_videos"), info.get("chunks_size"), codec,
              image_shape, json.dumps(features, ensure_ascii=False), size)
    if existing:
        # 已登记过：只刷新指标，保留原有名字（重名与否都不再改），避免用户看到名字被改掉
        dataset_id = existing["id"]
        name = existing["name"]
        db.execute(
            """UPDATE datasets SET name=?, path=?, format=?, codebase_version=?, robot_type=?, fps=?,
                   total_episodes=?, total_frames=?, total_tasks=?, total_videos=?, chunks_size=?,
                   codec=?, image_shape=?, features_json=?, size_bytes=?, status='ready',
                   refreshed_at=? WHERE id=?""",
            (name,) + values + (now, dataset_id),
        )
    else:
        dataset_id = db.execute(
            """INSERT INTO datasets (name, path, format, codebase_version, robot_type, fps,
                   total_episodes, total_frames, total_tasks, total_videos, chunks_size, codec,
                   image_shape, features_json, size_bytes, status, created_at, refreshed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'ready', ?, ?)""",
            (name,) + values + (now, now),
        )

    # 增量模式：只探测库里还没有的 episode（采集每加一条就全量重扫是 O(n) 的浪费）
    all_indices = sorted(meta["episodes"]) or list(range(int(info.get("total_episodes") or 0)))
    incremental = bool(payload.get("incremental"))
    explicit = payload.get("episode_indices")
    if explicit is not None:
        wanted = set(explicit)
        indices = [i for i in all_indices if i in wanted]
        mode = "explicit"
    elif incremental:
        existing = {r["episode_index"] for r in db.query(
            "SELECT episode_index FROM episodes WHERE dataset_id=?", (dataset_id,))}
        indices = [i for i in all_indices if i not in existing]
        mode = "incremental"
    else:
        db.execute("DELETE FROM episodes WHERE dataset_id=?", (dataset_id,))
        indices = all_indices
        mode = "full"
    total = len(indices)
    probed_frames = 0
    missing = 0
    for done, index in enumerate(indices, start=1):
        record = lerobot.probe_episode(path, meta, index, with_video=with_video)
        probed_frames += record["data_rows"] or 0
        if not record["data_exists"] or not all(v.get("exists") for v in record["videos"].values()):
            missing += 1
        videos = record["videos"] or {}
        first_video = next(iter(videos.values()), {})
        fps = float(info.get("fps") or 0)
        db.execute(
            """INSERT INTO episodes (dataset_id, episode_index, length, task_index, task_text,
                    duration_s, data_rows, video_frames, action_dim, state_dim, success_count,
                    done_count, nan_values, data_bytes, video_bytes, parquet_path, video_path,
                    probe_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(dataset_id, episode_index) DO UPDATE SET
                    length=excluded.length, task_index=excluded.task_index, task_text=excluded.task_text,
                    duration_s=excluded.duration_s, data_rows=excluded.data_rows,
                    video_frames=excluded.video_frames, action_dim=excluded.action_dim,
                    state_dim=excluded.state_dim, success_count=excluded.success_count,
                    done_count=excluded.done_count, nan_values=excluded.nan_values,
                    data_bytes=excluded.data_bytes, video_bytes=excluded.video_bytes,
                    parquet_path=excluded.parquet_path, video_path=excluded.video_path,
                    probe_json=excluded.probe_json""",
            (dataset_id, index, record["declared_length"], record["task_index"], record["task_text"],
             round((record["data_rows"] or 0) / fps, 3) if fps else None,
             record["data_rows"], record["video_frames"], record["action_dim"],
             record.get("observation_state_dim"), record["success_count"], record["done_count"],
             record["nan_values"], record["data_bytes"], record["video_bytes"],
             record["parquet_path"], first_video.get("path"),
             json.dumps(record, ensure_ascii=False, default=str), now),
        )
        if done % 5 == 0 or done == total:
            job.progress(2 + int(done / max(total, 1) * 8), 10,
                         f"接入 episode {index}（{done}/{total}）")

    db.execute("UPDATE datasets SET refreshed_at=? WHERE id=?", (db.now(), dataset_id))
    audit.log("INFO", "数据接入",
              f"数据集 #{dataset_id} {name}：本次探测 {total} 条（{mode}），共 "
              f"{info.get('total_episodes')} 条 / {info.get('total_frames')} 帧",
              ref_type="dataset", ref_id=dataset_id,
              detail={"mode": mode, "probed": total, "missing_files": missing})

    # 血缘与版本：origin 表示这份数据是"从哪个数据集、经什么操作"来的
    origin = payload.get("origin") or {}
    if origin.get("parent_dataset_id"):
        db.execute("UPDATE datasets SET parent_dataset_id=?, derived_by=? WHERE id=?",
                   (origin["parent_dataset_id"], origin.get("derived_by") or origin.get("kind"),
                    dataset_id))
    version_info = _register_version(
        dataset_id, origin.get("kind") or "ingest", path,
        episodes=total, frames=probed_frames, params=origin.get("params"),
        parent_dataset_id=origin.get("parent_dataset_id"),
        parent_version=origin.get("parent_version"), note=origin.get("note"))

    return {"dataset_id": dataset_id, "name": name, "episodes": total,
            "frames": probed_frames, "missing_files": missing, "with_video": with_video,
            "bytes": size, "version": version_info["version"],
            "content_hash": version_info["content_hash"],
            "version_deduplicated": version_info["deduplicated"]}


# --------------------------------------------------------------------------- ② 预处理（按规则筛选生成新数据集）

PREPROCESS_RULES = {
    "all": "全部保留",
    "qc_pass": "只保留最近一次质检通过的 episode",
    "success_only": "只保留有成功标记的 episode",
    "drop_constant": "剔除 action 全程不变的 episode",
    # 人工勾选：最灵活也最危险，所以**默认全保留**，只按明确排除的列表剔除。
    # 注意 qc_pass 只丢 severity='error'，而 VLM 判定永不产生 error ——
    # 想把 VLM 标出的坏样本剔掉，只能走这条人工路径。
    "manual": "人工勾选（排除指定 episode）",
}


@jobs.handler("preprocess")
def preprocess(job: jobs.JobContext, payload: dict) -> dict:
    """按规则筛出子集，并真实写出一个新数据集（LeRobot 格式）。"""
    dataset_id = int(payload["dataset_id"])
    rule = payload.get("rule") or "qc_pass"
    row = _dataset_row(dataset_id)
    episodes = db.query("SELECT * FROM episodes WHERE dataset_id=? ORDER BY episode_index", (dataset_id,))
    if not episodes:
        raise ValueError("该数据集还没有 episode，请先执行接入")

    keep: list[int] = []
    # 人工勾选的排除名单（manual 规则用）。显式规范化，防止传进来字符串/浮点。
    def _as_index_set(value) -> set[int]:
        out: set[int] = set()
        for item in (value or []):
            try:
                out.add(int(item))
            except (TypeError, ValueError):
                continue
        return out

    manual_keep = _as_index_set(payload.get("keep")) if payload.get("keep") is not None else None
    manual_exclude = (_as_index_set(payload.get("exclude"))
                      if payload.get("exclude") is not None else None)
    for ep in episodes:
        idx = ep["episode_index"]
        if rule == "all":
            keep.append(idx)
        elif rule == "manual":
            # 语义：**默认保留**。给了 exclude 就按"全部减去 exclude"；
            # 只给 keep 才按白名单。两者都给时以 keep 为准（更明确）。
            if manual_keep is not None:
                if idx in manual_keep:
                    keep.append(idx)
            elif manual_exclude is not None:
                if idx not in manual_exclude:
                    keep.append(idx)
            else:
                raise ValueError("人工勾选规则需要提供 exclude 或 keep 列表")
        elif rule == "success_only":
            if (ep["success_count"] or 0) > 0:
                keep.append(idx)
        elif rule == "drop_constant":
            probe = json.loads(ep["probe_json"] or "{}")
            if probe.get("action_constant") is not True:
                keep.append(idx)
        elif rule == "qc_pass":
            bad = db.scalar(
                "SELECT COUNT(*) AS c FROM qc_findings f JOIN qc_runs r ON r.id=f.run_id"
                " WHERE f.dataset_id=? AND f.episode_index=? AND f.severity='error'"
                " AND r.id=(SELECT MAX(id) FROM qc_runs WHERE dataset_id=?)",
                (dataset_id, idx, dataset_id))
            if not bad:
                keep.append(idx)
        else:
            raise ValueError(f"未知预处理规则：{rule}")

    if not keep:
        raise ValueError(f"规则「{PREPROCESS_RULES.get(rule, rule)}」下没有 episode 被保留")

    job.progress(1, 3, f"筛出 {len(keep)}/{len(episodes)} 条")

    stamp = db.now().replace("-", "").replace(":", "").replace(" ", "_")
    name = payload.get("name") or f"{row['name']}_{rule}_{stamp}"
    out_dir = db.EXPORT_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    meta = lerobot.load_meta(row["path"])
    job.progress(2, 3, "写出 LeRobot 数据集")
    result = lerobot.export_subset(row["path"], meta, keep, out_dir, name,
                                   annotations=_annotations(dataset_id))

    job.progress(3, 3, "打包")
    zip_path = _zip_dataset(out_dir)
    artifact_id = db.execute(
        "INSERT INTO artifacts (job_id, dataset_id, kind, path, bytes, created_at) VALUES (?,?,?,?,?,?)",
        (job.id, dataset_id, "dataset", str(zip_path), zip_path.stat().st_size, db.now()),
    )
    # 产物登记为**新数据集**并建立血缘：之后可以继续质检 / 标注 / 再导出
    # 这里是「规则类预处理」：result 只有 episodes/frames/bytes，
    # 不能写修复模式才有的 episodes_in/frames_dropped（早先补日志时替换命中了本函数）。
    audit.log("INFO", "数据预处理",
              f"数据集 #{dataset_id} {row['name']} 预处理"
              f"（规则 {payload.get('rule') or '全部'}）：产出 "
              f"{result.get('episodes')} 条 / {result.get('frames')} 帧，"
              f"排除 {len(episodes) - len(keep)} 条（源数据集保持原样，未做任何删除）",
              ref_type="dataset", ref_id=dataset_id,
              detail={"rule": payload.get("rule"), "episodes": result.get("episodes"),
                      "frames": result.get("frames"), "kept": len(keep),
                      "dropped": len(episodes) - len(keep),
                      "excluded": sorted(set(e["episode_index"] for e in episodes) - set(keep))})
    current = db.query_one("SELECT version FROM datasets WHERE id=?", (dataset_id,)) or {}
    ingest(job, {
        "path": str(out_dir), "name": name, "with_video": True,
        "origin": {"kind": "preprocess", "derived_by": f"preprocess:{rule}",
                   "parent_dataset_id": dataset_id, "parent_version": current.get("version"),
                   "params": {"rule": rule, "kept": len(keep),
                              "dropped": len(episodes) - len(keep)}},
    })
    derived = db.query_one("SELECT id, version, content_hash FROM datasets WHERE path=?",
                           (str(out_dir),))
    return {**result, "rule": rule, "rule_label": PREPROCESS_RULES.get(rule, rule),
            "kept": len(keep), "dropped": len(episodes) - len(keep),
            "zip": str(zip_path), "artifact_id": artifact_id,
            "derived_dataset_id": derived["id"] if derived else None,
            "derived_version": derived["version"] if derived else None,
            "content_hash": derived["content_hash"] if derived else None}


# --------------------------------------------------------------------------- ③ 质检

@jobs.handler("qc")
def qc(job: jobs.JobContext, payload: dict) -> dict:
    """对数据集逐条 episode 跑规则，落库并产出真实报告文件。

    `payload.vlm=true` 时，在 13 条结构规则之外**追加 VLM 内容质检**（见 core/vlm_qc.py）：
    判断任务是否真的完成、有无失败迹象、画面是否可用、是否有多余人手入镜。

    VLM 的结论写进**同一张 qc_findings**，所以下游「打回重采 / 按质检通过导出 /
    修正类预处理 qc_pass」全都不用改。VLM 判定默认是 **warning（只标记待人工复核）**，
    不直接判 error —— 模型会有误判，打回重采的决定权必须在人手里。
    """
    dataset_id = int(payload["dataset_id"])
    row = _dataset_row(dataset_id)
    meta = lerobot.load_meta(row["path"])
    episodes = db.query("SELECT episode_index FROM episodes WHERE dataset_id=? ORDER BY episode_index",
                        (dataset_id,))
    indices = [e["episode_index"] for e in episodes] or sorted(meta["episodes"])
    total = len(indices)

    # VLM（可选）：调用方说跑才跑；配置里只能给默认值，不能替调用方决定
    use_vlm = bool(payload.get("vlm"))
    vlm_cfg = None
    vlm_qc = None
    if use_vlm:
        from . import vlm_qc as vlm_qc_module

        vlm_qc = vlm_qc_module
        vlm_cfg = vlm_qc.load_config({
            "base_url": payload.get("vlm_base_url"),
            "model": payload.get("vlm_model"),
            "mode": payload.get("vlm_mode"),
            "frames": payload.get("vlm_frames"),
            "tile_width": payload.get("vlm_tile_width"),
        })
        if payload.get("vlm_use_cache") is not None:
            vlm_cfg["use_cache"] = bool(payload["vlm_use_cache"])

    # 扩展分类质检（12 项维度）：阈值全部来自 storage/qc_config.json，不写死在代码里
    qc_cfg = qc_checks.load_config({
        "enabled_categories": payload.get("qc_categories"),
    })
    if payload.get("qc_config") and isinstance(payload["qc_config"], dict):
        qc_cfg = qc_checks.load_config(payload["qc_config"])
    qc_fingerprint = qc_checks.config_fingerprint(qc_cfg)


    run_id = db.execute(
        "INSERT INTO qc_runs (dataset_id, job_id, total, created_at) VALUES (?,?,?,?)",
        (dataset_id, job.id, total, db.now()),
    )
    db.execute("DELETE FROM qc_findings WHERE run_id=?", (run_id,))

    per_episode: list[dict] = []      # 先收集，等数据集级检查（跨 episode 离群）跑完再统一落库
    vlm_stats = {"enabled": use_vlm, "checked": 0, "cached": 0, "flagged": 0,
                 "unavailable": 0, "latency_ms": 0}
    for done, index in enumerate(indices, start=1):
        record = lerobot.probe_episode(row["path"], meta, index, with_video=True)
        findings = lerobot.check_episode(record, meta["info"])

        # 扩展分类检查：元数据/时序/动作质量/视频/统计分布/可移植性/运动学
        ext = qc_checks.check_episode_extended(row["path"], meta, index, record, qc_cfg)
        findings = findings + (ext.get("findings") or [])

        vlm_result = None
        if use_vlm:
            vlm_result = vlm_qc.run_episode_safe(row["path"], meta, index, vlm_cfg, dataset_id)
            vlm_findings = vlm_result.get("findings") or []
            findings = findings + vlm_findings
            vlm_stats["checked"] += 1
            vlm_stats["cached"] += 1 if vlm_result.get("cached") else 0
            vlm_stats["latency_ms"] += int(vlm_result.get("latency_ms") or 0)
            # 「服务不可用」要单独数：它不是"VLM 觉得这条可疑"，而是"VLM 根本没看"。
            # 混进 flagged 会让界面显示成"抓出 N 条问题"，严重误导。
            unavailable = any(f["rule"] == "VLM·服务不可用" for f in vlm_findings)
            if unavailable:
                vlm_stats["unavailable"] += 1
            real_warnings = [f for f in vlm_findings
                             if f["severity"] in ("error", "warning")
                             and f["rule"] != "VLM·服务不可用"]
            if real_warnings:
                vlm_stats["flagged"] += 1

        per_episode.append({
            "episode_index": index, "record": record, "findings": findings,
            "measurements": ext.get("measurements") or {},
            "checks_run": ext.get("checks_run") or [],
            "vlm": vlm_result,
        })

        # 开了 VLM 每条都要几秒到几十秒，逐条报进度，否则前端看起来像卡死
        if use_vlm or done % 5 == 0 or done == total:
            job.progress(done, total,
                         f"质检 episode {index}（{done}/{total}）"
                         + (f"｜VLM 判定 {vlm_result.get('verdict')}" if use_vlm and vlm_result else ""))

    # ---- 数据集级：跨 episode 离群（MAD 自适应，不用硬编码 3σ；样本不足只出 info）----
    dataset_findings = qc_checks.dataset_level_stats(per_episode, qc_cfg)
    for item in per_episode:
        extra = dataset_findings.get(item["episode_index"]) or []
        if extra:
            item["findings"] = item["findings"] + extra

    # ---- 统一落库 ----
    findings_rows = []
    report_episodes = []
    passed = failed = 0
    health_scores: list[dict] = []
    for item in per_episode:
        index = item["episode_index"]
        record = item["record"]
        findings = item["findings"]
        summary_e = lerobot.summarize(findings)
        if summary_e["errors"]:
            failed += 1
        else:
            passed += 1
        health = qc_checks.health_score(findings, qc_cfg)
        health_scores.append(health)
        for f in findings:
            category = (f.get("category")
                        or qc_checks.LEGACY_RULE_CATEGORY.get(f["rule"], "format"))
            findings_rows.append((run_id, dataset_id, index, f["rule"], f["severity"],
                                  f["detail"], f["measured"], f["expected"],
                                  category, f.get("check_version") or ""))
        episode_report = {
            "episode_index": index, "result": summary_e["result"], "errors": summary_e["errors"],
            "warnings": summary_e["warnings"], "data_rows": record["data_rows"],
            "video_frames": record["video_frames"], "duration_s":
                round((record["data_rows"] or 0) / float(meta["info"].get("fps") or 1), 3),
            "findings": findings,
            "health": health,
            "measurements": item["measurements"],
            "checks_run": item["checks_run"],
        }
        vlm_result = item["vlm"]
        if vlm_result is not None:
            # 报告里留全 VLM 结果：逐问回答、描述式输出（可直接当语言指令用）、耗时
            episode_report["vlm"] = {
                "source": vlm_result.get("source"), "tiles": vlm_result.get("tiles"),
                "cameras": vlm_result.get("cameras"), "cached": vlm_result.get("cached"),
                "verdict": vlm_result.get("verdict"), "latency_ms": vlm_result.get("latency_ms"),
                "checklist": vlm_result.get("checklist"), "describe": vlm_result.get("describe"),
                "error": vlm_result.get("error"),
            }
        report_episodes.append(episode_report)

    if findings_rows:
        db.execute_many(
            "INSERT INTO qc_findings (run_id, dataset_id, episode_index, rule, severity, detail,"
            " measured, expected, category, check_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
            findings_rows)

    by_rule: dict = {}
    for f in findings_rows:
        key = f[3]
        entry = by_rule.setdefault(key, {"rule": key, "errors": 0, "warnings": 0, "infos": 0,
                                         "severity": f[4], "category": f[8],
                                         "check_version": f[9]})
        entry["errors" if f[4] == "error" else "warnings" if f[4] == "warning" else "infos"] += 1

    # 「12 项」维度的汇总：页面按这个分组显示
    by_category: dict = {}
    for f in findings_rows:
        category = f[8] or "format"
        entry = by_category.setdefault(category, {
            "category": category, "label": qc_checks.CATEGORY_LABELS.get(category, category),
            "errors": 0, "warnings": 0, "infos": 0})
        entry["errors" if f[4] == "error" else "warnings" if f[4] == "warning" else "infos"] += 1

    if use_vlm:
        vlm_stats["model"] = vlm_cfg.get("model")
        vlm_stats["mode"] = vlm_cfg.get("mode")
        vlm_stats["base_url"] = vlm_cfg.get("base_url")
        vlm_stats["cache_hit_rate"] = (round(vlm_stats["cached"] / vlm_stats["checked"], 3)
                                       if vlm_stats["checked"] else 0)
        # 「生效」= 至少真问到过模型。全程不可用时必须显眼地说出来，
        # 否则任务照样显示"完成"，很容易被当成"VLM 质检跑过了"。
        vlm_stats["effective"] = bool(vlm_stats["checked"]
                                      and vlm_stats["unavailable"] < vlm_stats["checked"])
        if vlm_stats["checked"] and not vlm_stats["effective"]:
            vlm_stats["warning"] = (
                "VLM 后端全程不可用：本次没有产生任何真实内容判定，"
                f"请先启动模型服务（{vlm_cfg.get('base_url')}）再重新质检")
        elif vlm_stats["unavailable"]:
            vlm_stats["warning"] = (f"{vlm_stats['unavailable']} 条因后端不可用没看成，"
                                    f"建议修好后重新质检")

    avg_health = (round(sum(h["score"] for h in health_scores) / len(health_scores), 1)
                  if health_scores else None)
    summary = {
        "dataset": row["name"], "dataset_id": dataset_id, "path": row["path"],
        "fps": meta["info"].get("fps"), "total": total, "passed": passed, "failed": failed,
        "result": "打回重采" if failed else "通过",
        "rules": sorted(by_rule.values(), key=lambda r: (-r["errors"], -r["warnings"])),
        "categories": sorted(by_category.values(),
                             key=lambda c: (-c["errors"], -c["warnings"])),
        "coverage": qc_checks.coverage(per_episode),
        "qc_config": {"version": qc_cfg.get("version"), "fingerprint": qc_fingerprint},
        "health": {
            "avg_score": avg_health,
            "min_score": min((h["score"] for h in health_scores), default=None),
            "worst_grade": (min((h["grade"] for h in health_scores), default=None)
                            if health_scores else None),
            "note": "健康度仅用于分诊排序，不预测训练效果，也不参与自动删除",
        },
        "vlm": vlm_stats,
        "generated_at": db.now(),
    }

    vlm_ineffective = bool(use_vlm and vlm_stats["checked"] and not vlm_stats["effective"])
    if use_vlm:
        vlm_note = (f"；VLM 复核 {vlm_stats['checked']} 条，标出可疑 {vlm_stats['flagged']} 条"
                    if vlm_stats["effective"]
                    else f"；VLM **未生效**：{vlm_stats['unavailable']} 条全部因后端不可用没看成")
    else:
        vlm_note = ""
    audit.log("WARNING" if (failed or vlm_ineffective) else "INFO", "数据质检",
              f"数据集 #{dataset_id} {row['name']} 质检：{passed} 通过 / {failed} 打回（run{run_id}）"
              + vlm_note,
              ref_type="dataset", ref_id=dataset_id,
              detail={"run_id": run_id, "passed": passed, "failed": failed,
                      "result": summary["result"], "vlm": vlm_stats if use_vlm else None})
    report_path = db.REPORT_DIR / f"qc_{row['name']}_{run_id}.json"
    report_path.write_text(json.dumps({"summary": summary, "episodes": report_episodes},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    db.execute(
        "UPDATE qc_runs SET report_path=?, passed=?, failed=?, result=?, summary_json=? WHERE id=?",
        (str(report_path), passed, failed, summary["result"],
         json.dumps(summary, ensure_ascii=False), run_id),
    )
    db.execute("INSERT INTO artifacts (job_id, dataset_id, kind, path, bytes, created_at)"
               " VALUES (?,?,?,?,?,?)",
               (job.id, dataset_id, "report", str(report_path), report_path.stat().st_size, db.now()))
    return {"run_id": run_id, **summary, "report": str(report_path)}


# --------------------------------------------------------------------------- ④ 导出

@jobs.handler("export")
def export(job: jobs.JobContext, payload: dict) -> dict:
    """把数据集（或全库通过的 episode）导出成真实 LeRobot v2.1 目录 + zip。"""
    dataset_id = int(payload["dataset_id"])
    row = _dataset_row(dataset_id)
    meta = lerobot.load_meta(row["path"])

    mode = payload.get("mode") or "all"
    split_name = payload.get("split")
    if mode == "split":
        from . import splits as splits_mod

        if not split_name:
            raise ValueError("按划分导出需要指定 split（train/val/test）")
        indices = splits_mod.episodes_of(dataset_id, split_name)
        if not indices:
            raise ValueError(f"数据集 #{dataset_id} 还没有 '{split_name}' 划分，先去导出页做一次数据划分")
    elif mode == "qc_pass":
        bad = {f["episode_index"] for f in db.query(
            "SELECT DISTINCT f.episode_index FROM qc_findings f JOIN qc_runs r ON r.id=f.run_id"
            " WHERE f.dataset_id=? AND f.severity='error'"
            " AND r.id=(SELECT MAX(id) FROM qc_runs WHERE dataset_id=?)", (dataset_id, dataset_id))}
        indices = [e["episode_index"] for e in
                   db.query("SELECT episode_index FROM episodes WHERE dataset_id=? ORDER BY episode_index",
                            (dataset_id,)) if e["episode_index"] not in bad]
    else:
        indices = sorted(meta["episodes"]) or list(range(int(meta["info"].get("total_episodes") or 0)))

    if not indices:
        raise ValueError("没有可导出的 episode")

    stamp = db.now().replace("-", "").replace(":", "").replace(" ", "_")
    name = payload.get("name") or f"{row['name']}_export_{stamp}"
    out_dir = db.EXPORT_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    job.progress(1, 3, f"写出 {len(indices)} 条 episode")
    result = lerobot.export_subset(row["path"], meta, indices, out_dir, name,
                                   annotations=_annotations(dataset_id))
    # 画布类标注（目标框/关键点/阶段）随导出包一起交付
    spatial_meta = spatial.write_meta(out_dir, dataset_id, indices)
    job.progress(2, 3, "打包 zip")
    zip_path = _zip_dataset(out_dir)
    artifact_id = db.execute(
        "INSERT INTO artifacts (job_id, dataset_id, kind, path, bytes, created_at) VALUES (?,?,?,?,?,?)",
        (job.id, dataset_id, "dataset", str(zip_path), zip_path.stat().st_size, db.now()),
    )
    job.progress(3, 3, "登记版本")
    current = db.query_one("SELECT version FROM datasets WHERE id=?", (dataset_id,)) or {}
    version_info = _register_version(
        dataset_id, "export", out_dir, episodes=result["episodes"], frames=result["frames"],
        zip_path=str(zip_path), params={"mode": mode, "name": name, "split": split_name},
        parent_version=current.get("version"))
    return {**result, "mode": mode, "split": split_name, "zip": str(zip_path), "artifact_id": artifact_id,
            "spatial_annotations": spatial_meta,
            "version": version_info["version"], "content_hash": version_info["content_hash"],
            "version_deduplicated": version_info["deduplicated"]}


def _zip_dataset(out_dir: Path) -> Path:
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in out_dir.rglob("*"):
            if file.is_file():
                zf.write(file, Path(out_dir.name) / file.relative_to(out_dir))
    return zip_path


# --------------------------------------------------------------------------- ⓪ 原始轨迹转换（本体专属数据 → VLA 格式）

DATASET_DIR = db.STORAGE / "datasets"
RAW_JOINT_ROOTS = [r"C:\Users\ll\Desktop\dhz\hkc"]


@jobs.handler("convert")
def convert(job: jobs.JobContext, payload: dict) -> dict:
    """把机器人自己采集的原始关节轨迹 parquet 转成 LeRobot v2.1，并直接接入平台。"""
    from . import raw_hkc

    root = payload.get("root") or RAW_JOINT_ROOTS[0]
    name = payload.get("name") or f"raw-{Path(root).name}"
    task_text = payload.get("task_text") or "抓取放置任务（待标注）"

    files = raw_hkc.find_raw_episodes(root)
    if not files:
        raise ValueError(f"在 {root} 下没找到符合「timestamp + joint_1..joint_7」结构的 parquet")
    job.progress(1, 10, f"找到 {len(files)} 段原始轨迹")

    out_dir = DATASET_DIR / name
    result = raw_hkc.convert(files, out_dir, name, task_text=task_text, fps=payload.get("fps"))
    job.progress(6, 10, f"已写出 {result['episodes']} 条 episode，开始接入")

    # 复用 ingest 处理器完成登记（同一套探测与落库逻辑，避免两套实现）
    ingest(job, {"path": str(out_dir), "name": name, "with_video": False})
    job.progress(10, 10, "完成")
    dataset = db.query_one("SELECT id FROM datasets WHERE name=?", (name,))
    return {**result, "dataset_id": dataset["id"] if dataset else None, "task_text": task_text}


# --------------------------------------------------------------------------- ⑤ 修正类预处理（会改帧）

@jobs.handler("repair")
def repair_job(job: jobs.JobContext, payload: dict) -> dict:
    """对质检发现的坏样本做真实修复，产物登记为新数据集并建立血缘。"""
    from . import repair

    dataset_id = int(payload["dataset_id"])
    row = _dataset_row(dataset_id)
    ops = payload.get("ops") or repair.DEFAULT_OPS
    name = payload.get("name") or f"{row['name']}_fixed"
    out_dir = db.EXPORT_DIR / name
    meta = lerobot.load_meta(row["path"])
    stamp = db.now().replace("-", "").replace(":", "").replace(" ", "_")

    job.progress(1, 4, f"开始修复（{len(ops)} 项操作）")
    result = repair.repair_dataset(
        row["path"], meta, out_dir, name, ops,
        resample_to=payload.get("resample_to"),
        dup_tol=float(payload.get("dup_tol", 1e-6)),
        annotations=_annotations(dataset_id))

    job.progress(2, 4, "写修复报告")
    report_path = db.REPORT_DIR / f"repair_{row['name']}_{stamp}.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    db.execute("INSERT INTO artifacts (job_id, dataset_id, kind, path, bytes, created_at)"
               " VALUES (?,?,?,?,?,?)",
               (job.id, dataset_id, "report", str(report_path), report_path.stat().st_size, db.now()))

    job.progress(3, 4, "打包并登记新数据集")
    zip_path = _zip_dataset(out_dir)
    db.execute("INSERT INTO artifacts (job_id, dataset_id, kind, path, bytes, created_at)"
               " VALUES (?,?,?,?,?,?)",
               (job.id, dataset_id, "dataset", str(zip_path), zip_path.stat().st_size, db.now()))

    audit.log("INFO", "修正处理",
              f"数据集 #{dataset_id} {row['name']} 修复："
              f"{result.get('episodes_in')} → {result.get('episodes_out')} 条，"
              f"删 {result.get('frames_dropped')} 帧，"
              f"重建视频 {result.get('videos_rebuilt')} 个",
              ref_type="dataset", ref_id=dataset_id,
              detail={"ops": ops})

    current = db.query_one("SELECT version FROM datasets WHERE id=?", (dataset_id,)) or {}
    ingest(job, {
        "path": str(out_dir), "name": name,
        "with_video": bool(lerobot.video_keys(meta["info"])),
        "origin": {"kind": "repair", "derived_by": "repair:" + "+".join(ops),
                   "parent_dataset_id": dataset_id, "parent_version": current.get("version"),
                   "params": {"ops": ops, "dropped_episodes": len(result["dropped_episodes"]),
                              "frames_dropped": result["frames_dropped"]}},
    })
    derived = db.query_one("SELECT id, version, content_hash FROM datasets WHERE path=?",
                           (str(out_dir),))
    job.progress(4, 4, "完成")
    return {
        "episodes_in": result["episodes_in"], "episodes_out": result["episodes_out"],
        "frames_out": result["frames_out"], "frames_dropped": result["frames_dropped"],
        "videos_rebuilt": result["videos_rebuilt"], "op_labels": result["op_labels"],
        "dropped_episodes": result["dropped_episodes"],
        "changed_episodes": [r for r in result["per_episode"]
                             if r["dropped_frames"] or r["resampled"]][:50],
        "zip": str(zip_path), "report": str(report_path),
        "derived_dataset_id": derived["id"] if derived else None,
        "derived_version": derived["version"] if derived else None,
        "content_hash": derived["content_hash"] if derived else None,
    }
