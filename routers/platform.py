"""平台自身的 JSON 接口（前端页面调用）：数据集 / episode / 任务 / 质检 / 标注 / 产物。

与 /interfaces 页面的区别：/interfaces 是「对外推理接口」的登记展示页，
这里 /api/* 是平台内部真正在跑的接口。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from core import audit, db, dedup, inline_images, jobs, lerobot, pipeline, spatial

router = APIRouter()


# --------------------------------------------------------------------------- 概览

@router.get("/overview")
def overview():
    return {
        "datasets": db.scalar("SELECT COUNT(*) AS c FROM datasets"),
        "episodes": db.scalar("SELECT COUNT(*) AS c FROM episodes"),
        "frames": db.scalar("SELECT COALESCE(SUM(data_rows),0) AS c FROM episodes"),
        "hours": round((db.scalar("SELECT COALESCE(SUM(data_rows),0) AS c FROM episodes") or 0)
                       / 10 / 3600, 3),
        "qc_runs": db.scalar("SELECT COUNT(*) AS c FROM qc_runs"),
        "qc_bad_episodes": db.scalar(
            "SELECT COUNT(DISTINCT episode_index) AS c FROM qc_findings WHERE severity='error'"),
        "annotated": db.scalar("SELECT COUNT(*) AS c FROM annotations"),
        "artifacts": db.scalar("SELECT COUNT(*) AS c FROM artifacts"),
        "bytes": db.scalar("SELECT COALESCE(SUM(size_bytes),0) AS c FROM datasets"),
        "running_jobs": jobs.running_count(),
    }


# --------------------------------------------------------------------------- 数据集 / episode

@router.get("/datasets")
def list_datasets():
    rows = db.query("SELECT * FROM datasets ORDER BY id DESC")
    for row in rows:
        row["bad_episodes"] = db.scalar(
            "SELECT COUNT(DISTINCT episode_index) AS c FROM qc_findings"
            " WHERE dataset_id=? AND severity='error'", (row["id"],))
        row["annotated"] = db.scalar(
            "SELECT COUNT(*) AS c FROM annotations WHERE dataset_id=?", (row["id"],))
    return rows


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int):
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise HTTPException(404, "数据集不存在")
    return row


@router.get("/datasets/{dataset_id}/episodes")
def list_episodes(dataset_id: int, limit: int = 500, offset: int = 0, only_bad: bool = False):
    sql = ("SELECT e.*, (SELECT COUNT(*) FROM qc_findings f JOIN qc_runs r ON r.id=f.run_id"
           "  WHERE f.dataset_id=e.dataset_id AND f.episode_index=e.episode_index"
           "  AND f.severity='error' AND r.id=(SELECT MAX(id) FROM qc_runs WHERE dataset_id=e.dataset_id)) AS error_count,"
           " a.task_text AS annotated_task, a.success AS annotated_success"
           " FROM episodes e LEFT JOIN annotations a"
           "   ON a.dataset_id=e.dataset_id AND a.episode_index=e.episode_index"
           " WHERE e.dataset_id=?")
    if only_bad:
        sql += " AND error_count > 0"
    sql += " ORDER BY e.episode_index LIMIT ? OFFSET ?"
    rows = db.query(sql, (dataset_id, limit, offset))
    for row in rows:
        probe = json.loads(row.pop("probe_json", None) or "{}")
        row["action_constant"] = probe.get("action_constant")
        row["timestamp_start"] = probe.get("timestamp_start")
        row["timestamp_end"] = probe.get("timestamp_end")
    return {"total": db.scalar("SELECT COUNT(*) AS c FROM episodes WHERE dataset_id=?", (dataset_id,)),
            "episodes": rows}


@router.get("/episodes/{dataset_id}/{episode_index}/frames")
def episode_frames_meta(dataset_id: int, episode_index: int):
    """这条 episode 的视觉源信息：有 mp4 还是内嵌图像、有几帧、几路相机。"""
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise HTTPException(404, f"数据集不存在：{dataset_id}")
    meta = lerobot.load_meta(row["path"])
    return inline_images.episode_meta(row["path"], meta, episode_index)


@router.get("/episodes/{dataset_id}/{episode_index}/frame")
def episode_frame(dataset_id: int, episode_index: int, index: int = 0, max_size: int = 640,
                  key: str | None = None):
    """取该 episode 第 index 帧（JPEG）：先试视频，视频没有再试内嵌图像（LIBERO 那类）。"""
    import cv2
    from fastapi import Response

    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise HTTPException(404, f"数据集不存在：{dataset_id}")
    meta = lerobot.load_meta(row["path"])

    # 1) 视频路线
    try:
        info = inline_images.episode_meta(row["path"], meta, episode_index)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"读取 episode 失败：{exc}") from exc
    errors = []
    if info["has_video"]:
        try:
            # 按 key **精确选中该路相机**的视频。原来走 spatial.photo_of_episode 只能拿到一路，
            # 多相机数据集（例如 zhiyan 有 head / hand_right）就没法切换视角。
            paths = lerobot.episode_paths(meta["info"], row["path"], episode_index)
            videos = paths.get("videos") or {}
            video = videos.get(key) if key else next(iter(videos.values()), None)
            if not video or not Path(video).is_file():
                errors.append(f"找不到该相机的视频（key={key}，可选 {sorted(videos)}）")
            else:
                cap = cv2.VideoCapture(str(video))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
                    ok, frame = cap.read()
                finally:
                    cap.release()
                if ok and frame is not None:
                    if max_size and frame.shape[1] > max_size:
                        scale = max_size / float(frame.shape[1])
                        frame = cv2.resize(frame, (max_size, max(1, int(frame.shape[0] * scale))))
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ok:
                        return Response(content=buf.tobytes(), media_type="image/jpeg",
                                        headers={"X-Frame-Source": "video",
                                                 "X-Frame-Key": str(key or ""),
                                                 "X-Frame-Index": str(index),
                                                 "X-Frame-Size": f"{frame.shape[1]}x{frame.shape[0]}"})
                errors.append("视频里读不到这一帧")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"视频取帧失败：{exc}")

    # 2) 内嵌图像路线（LIBERO 等）
    if info["has_inline_images"]:
        try:
            payload, ctype, used_key = inline_images.frame_bytes(
                row["path"], meta, episode_index, index, key or (info["image_keys"] or [None])[0])
            return Response(content=payload, media_type=ctype,
                            headers={"X-Frame-Source": "inline_image", "X-Frame-Key": used_key,
                                     "X-Frame-Index": str(index)})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"内嵌图像取帧失败：{exc}")

    raise HTTPException(404, "；".join(errors) or "该 episode 没有可用的视觉数据")


@router.get("/episodes/{dataset_id}/{episode_index}/video")
def episode_video(dataset_id: int, episode_index: int, key: str | None = None):
    """流式返回该 episode 的真实视频（只读原始数据集）。"""
    row = db.query_one("SELECT path FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise HTTPException(404, "数据集不存在")
    meta = lerobot.load_meta(row["path"])
    paths = lerobot.episode_paths(meta["info"], row["path"], episode_index)
    video = paths["videos"].get(key) if key else next(iter(paths["videos"].values()), None)
    if not video or not Path(video).is_file():
        raise HTTPException(404, "视频文件不存在")
    return FileResponse(str(video), media_type="video/mp4")


# --------------------------------------------------------------------------- 采集：扫描与接入

@router.post("/collect/scan")
def collect_scan(payload: dict = Body(default={})):
    roots = payload.get("roots") or pipeline.DEFAULT_SCAN_ROOTS
    return {"roots": roots, "datasets": pipeline.discover(roots)}


# --------------------------------------------------------------------------- 任务

@router.get("/jobs")
def list_jobs(limit: int = 30):
    return jobs.recent(limit)


@router.get("/jobs/{job_id}")
def get_job(job_id: int):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.get("result_json"):
        job["result"] = json.loads(job["result_json"])
    return job


@router.post("/jobs/{kind}")
def submit_job(kind: str, payload: dict = Body(default={})):
    if kind not in jobs.HANDLERS:
        raise HTTPException(400, f"未知任务类型：{kind}（可用：{', '.join(jobs.HANDLERS)}）")
    target_id = payload.get("dataset_id")
    job_id = jobs.submit(kind, payload, target_type="dataset", target_id=target_id)
    return {"job_id": job_id, "kind": kind}


# --------------------------------------------------------------------------- VLM 质检

@router.get("/vlm/health")
def vlm_health():
    """VLM 后端探活：服务是否可达、模型是否在列。不抛异常，只返回事实。"""
    from core import vlm_qc

    return vlm_qc.health()


@router.get("/vlm/config")
def vlm_config_get():
    from core import vlm_qc

    cfg = vlm_qc.load_config()
    return {"config": cfg, "path": str(vlm_qc.config_path()),
            "modes": {"checklist": "判别式：预设问题 + yes/no（只能查预设的问题）",
                      "describe": "描述式：先描述再判定（能发现未预设的问题）",
                      "both": "两种都跑（推荐）"}}


@router.put("/vlm/config")
def vlm_config_put(payload: dict = Body(default={})):
    """保存 VLM 配置（问题清单/后端/抽帧参数）。传 checklist:[] 表示恢复默认问题。"""
    from core import vlm_qc

    if "checklist" in payload and not isinstance(payload["checklist"], (list, type(None))):
        raise HTTPException(400, "checklist 必须是数组")
    audit.log("INFO", "数据质检", "更新 VLM 质检配置",
              detail={k: v for k, v in (payload or {}).items() if k != "checklist"})
    return {"config": vlm_qc.save_config(payload)}


@router.get("/vlm/batches")
def vlm_batches(dataset_id: int | None = None):
    """VLM 质检的**批次**列表。

    批次身份 = (model, prompt_hash)。缓存键里含 prompt_hash，所以问题清单 / 模型 /
    抽帧参数一变就是新批次 —— 同一批次的若干 episode 是一次质检一起产出的。
    没有批次概念的话，多次质检的结果会全混在一张表里（实测 #38 混了 4 批）。
    """
    sql = ("SELECT model, prompt_hash, COUNT(*) AS n,"
           " MIN(created_at) AS started_at, MAX(created_at) AS finished_at,"
           " SUM(COALESCE(compliant, 0)) AS compliant_n,"
           " SUM(COALESCE(latency_ms, 0)) AS total_ms,"
           " MIN(episode_index) AS ep_min, MAX(episode_index) AS ep_max"
           " FROM vlm_checks")
    params: list = []
    if dataset_id:
        sql += " WHERE dataset_id=?"
        params.append(dataset_id)
    sql += " GROUP BY model, prompt_hash ORDER BY started_at DESC"
    rows = db.query(sql, tuple(params))
    batches = []
    for i, row in enumerate(rows):
        batches.append({
            "key": row["prompt_hash"], "model": row["model"], "episodes": row["n"],
            "compliant": row["compliant_n"], "flagged": row["n"] - (row["compliant_n"] or 0),
            "started_at": row["started_at"], "finished_at": row["finished_at"],
            "avg_ms": round((row["total_ms"] or 0) / max(1, row["n"])),
            "ep_min": row["ep_min"], "ep_max": row["ep_max"],
            "is_latest": i == 0,
        })
    return {"batches": batches, "latest": batches[0]["key"] if batches else None,
            "count": len(batches)}


@router.get("/vlm/results")
def vlm_results(dataset_id: int | None = None, episode_index: int | None = None,
                limit: int = 200, only_flagged: bool = False,
                batch: str | None = None):
    """已跑过的 VLM 质检结果（含逐问回答与描述式输出）。

    `batch` 的取值：
      - 省略 / `latest` —— **最新一个批次**（默认，最常用）
      - `all`           —— 每个 episode 取最新一份（跨批次汇总）
      - 具体 prompt_hash —— 只看那一批次

    描述式的 `suggested_instruction` 可以直接当 VLA 的语言指令用，
    所以这个接口同时服务于"复核队列"和"标注回填"两个用途。
    """
    sql = "SELECT * FROM vlm_checks WHERE 1=1"
    params: list = []
    if dataset_id:
        sql += " AND dataset_id=?"
        params.append(dataset_id)
    if episode_index is not None:
        sql += " AND episode_index=?"
        params.append(episode_index)
    if only_flagged:
        sql += " AND compliant=0"
    if batch and batch not in ("latest", "all"):
        sql += " AND prompt_hash=?"
        params.append(batch)
    # 多取一些：同一 episode 可能有多份缓存（见下面的去重），
    # 若按 limit 原样取会在去重后不够数。
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, limit) * 12)

    rows = db.query(sql, tuple(params))

    if batch in (None, "") and rows:
        # 默认：只显示最新批次（rows 已按 id DESC，第一条最新）
        newest = rows[0].get("prompt_hash")
        rows = [r for r in rows if r.get("prompt_hash") == newest]

    # **同一 episode 只保留最新一份**。
    # 缓存键里含 prompt_hash，所以改了问题清单 / 模型 / 抽帧参数就会新增一条 ——
    # 这是刻意的（不该复用旧结论），显示时必须去重。
    latest: dict = {}
    for row in rows:
        key = (row.get("dataset_id"), row.get("episode_index"))
        latest.setdefault(key, []).append(row)
    out_rows = []
    for key in sorted(latest, key=lambda k: (k[0] or 0, k[1] or 0)):
        bucket = latest[key]
        head = dict(bucket[0])                       # 已按 id DESC，第一条即最新
        head["cached_versions"] = len(bucket)
        out_rows.append(head)
    rows = out_rows[: max(1, limit)]
    # 把 findings_json 解析成 result 再给页面 —— 前端读的是 row.result。
    # 注意：这里**不能提前 return**（我在加批次过滤时误加过一句 return rows，
    # 导致下面这段变成死代码，页面上所有结论都变成 unknown 和 "-"）。
    for row in rows:
        try:
            row["result"] = json.loads(row.pop("findings_json", None) or "{}")
        except Exception:  # noqa: BLE001
            row["result"] = {}
    return rows


@router.post("/vlm/apply-instruction")
def vlm_apply_instruction(payload: dict = Body(default={})):
    """把描述式质检产出的 `suggested_instruction` 回填成该 episode 的语言指令标注。

    只在原标注为空时写入（除非 overwrite=true）—— 人工写过的指令不该被模型覆盖。
    """
    dataset_id = payload.get("dataset_id")
    if dataset_id is None:
        raise HTTPException(400, "需要 dataset_id")
    dataset_id = int(dataset_id)
    overwrite = bool(payload.get("overwrite"))
    only = payload.get("episode_indices")

    sql = ("SELECT episode_index, findings_json FROM vlm_checks WHERE dataset_id=?"
           " ORDER BY episode_index, id DESC")
    rows = db.query(sql, (dataset_id,))
    existing = {a["episode_index"]: a for a in
                db.query("SELECT * FROM annotations WHERE dataset_id=?", (dataset_id,))}

    applied: list[dict] = []
    skipped: list[dict] = []
    seen: set[int] = set()
    for row in rows:
        index = row["episode_index"]
        if index in seen:                     # 同一 episode 多条缓存，取最新一条
            continue
        seen.add(index)
        if only is not None and index not in set(only):
            continue
        try:
            result = json.loads(row.get("findings_json") or "{}")
        except Exception:  # noqa: BLE001
            continue
        instruction = ((result.get("describe") or {}).get("suggested_instruction") or "").strip()
        if not instruction:
            skipped.append({"episode_index": index, "reason": "该条没有描述式指令"})
            continue
        current = existing.get(index) or {}
        if (current.get("task_text") or "").strip() and not overwrite:
            skipped.append({"episode_index": index, "reason": "已有人工指令，未覆盖"})
            continue
        db.execute(
            """INSERT INTO annotations (dataset_id, episode_index, task_text, success, note,
                    annotator, updated_at) VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(dataset_id, episode_index) DO UPDATE SET
                    task_text=excluded.task_text, note=excluded.note,
                    annotator=excluded.annotator, updated_at=excluded.updated_at""",
            (dataset_id, index, instruction, current.get("success"),
             "由 VLM 描述式质检生成，待人工确认", "vlm", db.now()))
        applied.append({"episode_index": index, "task_text": instruction})

    audit.log("INFO", "数据标注",
              f"数据集 #{dataset_id} 回填 VLM 指令 {len(applied)} 条（跳过 {len(skipped)} 条）",
              ref_type="dataset", ref_id=dataset_id,
              detail={"applied": len(applied), "skipped": len(skipped), "overwrite": overwrite})
    return {"applied": applied, "skipped": skipped}


# --------------------------------------------------------------------------- 质检

@router.get("/qc/categories")
def qc_categories():
    """质检维度（「12 项」）：页面按这个分组展示问题。"""
    from core import qc_checks

    return {"categories": [{"key": key, "label": label, "desc": desc}
                           for key, label, desc in qc_checks.CATEGORIES]}


@router.get("/qc/config")
def qc_config_get():
    """质检阈值配置（全部可调，不写死在代码里）。"""
    from core import qc_checks

    cfg = qc_checks.load_config()
    return {"config": cfg, "path": str(qc_checks.config_path()),
            "fingerprint": qc_checks.config_fingerprint(cfg),
            "check_version": qc_checks.CHECK_VERSION,
            "categories": [{"key": k, "label": l, "desc": d} for k, l, d in qc_checks.CATEGORIES]}


@router.put("/qc/config")
def qc_config_put(payload: dict = Body(default={})):
    """改阈值 / 开关某个维度。传 enabled_categories 可只开需要的几类。"""
    from core import qc_checks

    if "enabled_categories" in payload and not isinstance(
            payload["enabled_categories"], (dict, type(None))):
        raise HTTPException(400, "enabled_categories 必须是对象")
    audit.log("INFO", "数据质检", "更新质检阈值配置",
              detail={k: v for k, v in (payload or {}).items() if k != "kinematics"})
    cfg = qc_checks.save_config(payload)
    return {"config": cfg, "fingerprint": qc_checks.config_fingerprint(cfg)}


@router.get("/qc/coverage")
def qc_coverage(run_id: int | None = None):
    """覆盖率分母：质检里各维度跑在了多少比例的 episode 上。"""
    if run_id:
        row = db.query_one("SELECT summary_json FROM qc_runs WHERE id=?", (run_id,))
    else:
        row = db.query_one("SELECT summary_json FROM qc_runs ORDER BY id DESC LIMIT 1")
    if not row:
        raise HTTPException(404, "没有质检记录")
    try:
        summary = json.loads(row["summary_json"] or "{}")
    except Exception:  # noqa: BLE001
        summary = {}
    return {"coverage": summary.get("coverage") or {},
            "categories": summary.get("categories") or [],
            "health": summary.get("health") or {},
            "qc_config": summary.get("qc_config") or {}}


# --------------------------------------------------------------------------- 服务器质检

@router.get("/qc/remote/config")
def qc_remote_config_get():
    """服务器端质检配置（lerobot-doctor 跑在服务器上，本机只读结果）。"""
    from core import remote_qc

    cfg = remote_qc.load_config()
    safe = {k: v for k, v in cfg.items() if k != "key_file"}
    safe["key_file_exists"] = bool(__import__("pathlib").Path(cfg["key_file"]).is_file())
    return {"config": safe, "path": str(remote_qc.config_path()),
            # 预设与 doctor 自己的默认值都发出去：页面能显示"原版是多少"，
            # 让用户清楚自己改了什么，而不是盲目调数。
            "presets": {"teleop": remote_qc.DOCTOR_PRESET_TELEOP,
                        "doctor_own_defaults": remote_qc.DOCTOR_THRESHOLD_DEFAULTS},
            "checks": [{"key": k, "label": v[1], "category": v[0]}
                       for k, v in remote_qc.CHECK_CATEGORY.items()]}


@router.put("/qc/remote/config")
def qc_remote_config_put(payload: dict = Body(default={})):
    from core import remote_qc

    if "roots" in payload and not isinstance(payload["roots"], (list, type(None))):
        raise HTTPException(400, "roots 必须是数组")
    audit.log("INFO", "数据质检", "更新服务器质检配置",
              detail={k: v for k, v in (payload or {}).items() if k != "key_file"})
    return {"config": {k: v for k, v in remote_qc.save_config(payload).items()
                       if k != "key_file"}}


@router.get("/qc/remote/probe")
def qc_remote_probe():
    """探测：SSH 通不通、远端 lerobot-doctor 在不在。不抛异常。"""
    from core import remote_qc

    return remote_qc.probe()


@router.get("/qc/remote/resolve/{dataset_id}")
def qc_remote_resolve(dataset_id: int):
    """预览：这个数据集会被解析到服务器上的哪个路径（不执行检查）。"""
    from core import remote_qc

    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise HTTPException(404, "数据集不存在")
    return remote_qc.resolve_server_path(remote_qc.load_config(), row)


@router.get("/qc/remote/datasets")
def qc_remote_datasets():
    """哪些本机数据集已经能对应到服务器路径（给页面下拉用）。"""
    from core import remote_qc

    cfg = remote_qc.load_config()
    rows = db.query("SELECT id, name, server_path, total_episodes, codebase_version"
                    " FROM datasets ORDER BY id")
    out = []
    for row in rows:
        item = dict(row)
        item["mapped"] = bool((row.get("server_path") or "").strip())
        out.append(item)
    return {"enabled": bool(cfg.get("enabled")), "host": cfg.get("host"),
            "datasets": out}


@router.put("/datasets/{dataset_id}/server_path")
def set_dataset_server_path(dataset_id: int, payload: dict = Body(default={})):
    """给数据集手工指定服务器路径（空字符串 = 清除，回到自动匹配）。"""
    from core import remote_qc

    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise HTTPException(404, "数据集不存在")
    value = (payload.get("server_path") or "").strip()
    if value:
        safe, why = remote_qc._safe_remote_path(value)
        if not safe:
            raise HTTPException(400, f"路径不合法：{why}")
        if not remote_qc._within_roots(value, remote_qc.load_config().get("roots") or []):
            raise HTTPException(400, "路径不在允许的 roots 内（先在配置里加 root，或用合法子路径）")
    db.execute("UPDATE datasets SET server_path=? WHERE id=?", (value or None, dataset_id))
    audit.log("INFO", "数据质检", f"数据集 #{dataset_id} 指定服务器路径：{value or '（清除）'}",
              ref_type="dataset", ref_id=dataset_id)
    return {"dataset_id": dataset_id, "server_path": value or None}


@router.get("/qc/remote/upload_status/{dataset_id}")
def qc_remote_upload_status(dataset_id: int):
    """上传前"算账"：本机大小、服务器剩余空间、是否已有同名副本、目标路径。不发数据。"""
    from core import remote_qc

    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise HTTPException(404, "数据集不存在")
    cfg = remote_qc.load_config()
    status = remote_qc.upload_status(cfg, row)
    status.pop("marker", None)          # 内部细节，不给页面
    return status


@router.get("/qc/remote/uploaded")
def qc_remote_uploaded():
    """服务器上传目录里已有哪些数据集副本（含指纹与上传时间）。"""
    import shlex

    from core import remote_qc

    cfg = remote_qc.load_config()
    root = str(cfg.get("upload_root") or "").rstrip("/")
    if not root:
        return {"root": "", "items": []}
    script = (
        f"cd {shlex.quote(root)} 2>/dev/null || exit 0\n"
        "for d in */; do\n"
        '  n="${d%/}"\n'
        '  [ -d "$n" ] || continue\n'
        f'  m=$(cat "$n/{remote_qc.MARKER_NAME}" 2>/dev/null | tr -d "\\n")\n'
        '  printf \'%s\\t%s\\n\' "$n" "$m"\n'
        "done\n"
    )
    try:
        _rc, out, _err = remote_qc.ssh_run(cfg, script, timeout_s=40)
    except remote_qc.RemoteError as exc:
        raise HTTPException(502, f"读取服务器上传目录失败：{exc}")
    items = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        name, _, blob = line.partition("\t")
        items.append({"name": name.strip(), "meta": remote_qc._extract_json(blob) or {}})
    return {"root": root, "items": items}


@router.get("/qc/report/{run_id}")
def qc_report(run_id: int):
    """质检报告的界面化数据：汇总 + 按维度分组 + 远端完整结果（含 PASS 明细）。

    页面用它渲染「报告视图」，不用下载 JSON 就能看全。
    """
    run = db.query_one("SELECT * FROM qc_runs WHERE id=?", (run_id,))
    if not run:
        raise HTTPException(404, "质检记录不存在")
    try:
        summary = json.loads(run.get("summary_json") or "{}")
    except Exception:  # noqa: BLE001
        summary = {}

    findings = db.query(
        "SELECT episode_index, rule, category, severity, detail, measured, expected,"
        " check_version, source FROM qc_findings WHERE run_id=?"
        " ORDER BY category, episode_index IS NULL, episode_index, id", (run_id,))

    # 远端原始结果（lerobot-doctor 的完整 JSON，**含 PASS 明细**，报告要看全）
    doctor = None
    doctor_note = ""
    raw_path = str(((summary.get("raw") or {}).get("doctor_json") or ""))
    if raw_path:
        p = Path(raw_path)
        if not p.is_file():
            doctor_note = "原始结果文件已被清理（报告仍可从摘要与问题清单查看）"
        elif p.stat().st_size > 5 * 1024 * 1024:
            doctor_note = f"原始结果过大（{p.stat().st_size / 1024 / 1024:.1f}MB），请用下载按钮"
        else:
            try:
                doctor = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                doctor_note = f"原始结果解析失败：{type(exc).__name__}: {exc}"

    by_cat: dict = {}
    for item in findings:
        by_cat.setdefault(item.get("category") or "other", []).append(item)

    dataset = db.query_one(
        "SELECT name, codebase_version, total_episodes, total_frames, fps, robot_type,"
        " server_path FROM datasets WHERE id=?", (run["dataset_id"],)) or {}

    # 每个维度的状态（服务器运行来自 remote.checks，本机运行来自 categories）
    dims = []
    remote = summary.get("remote") or {}
    if remote.get("checks"):
        for check in remote["checks"]:
            dims.append({"key": check.get("name"), "label": check.get("label") or check.get("name"),
                         "status": check.get("severity"), "messages": check.get("messages")})
    else:
        for cat in summary.get("categories") or []:
            total = (cat.get("errors") or 0) + (cat.get("warnings") or 0) + (cat.get("infos") or 0)
            dims.append({"key": cat.get("category"), "label": cat.get("label"),
                         "status": ("FAIL" if cat.get("errors") else
                                    "WARN" if cat.get("warnings") else "PASS"),
                         "messages": total})
    # 本机扩展分类里没有命中的维度也补上（显示为 PASS，报告才完整）。
    # **只对本机运行补**：服务器运行的维度完全由 remote.checks 决定，
    # 补本机分类会和 doctor 的检查项撞名，出现一堆重名卡片（踩过）。
    if not remote.get("checks"):
        from core import qc_checks
        seen = {d["key"] for d in dims}
        for key, label, _desc in qc_checks.CATEGORIES:
            if key not in seen and key != "format":
                dims.append({"key": key, "label": label, "status": "PASS", "messages": 0})

    # 表头口径：服务器运行的 passed 是「检查项数」，不是 episode 数，别混着显示
    is_server = bool(remote.get("host"))
    head = {
        "mode": "server" if is_server else "local",
        "episodes": (remote.get("total_episodes") if is_server else run.get("total")),
        "passed_episodes": (None if is_server else run.get("passed")),
        "failed_episodes": (None if is_server else run.get("failed")),
        "checks_passed": ((remote.get("counts") or {}).get("PASS") if is_server else None),
        "checks_warn": ((remote.get("counts") or {}).get("WARN") if is_server else None),
        "checks_fail": ((remote.get("counts") or {}).get("FAIL") if is_server else None),
    }

    return {
        "run": {k: run.get(k) for k in ("id", "dataset_id", "total", "passed", "failed",
                                        "result", "created_at", "report_path")},
        "dataset": dataset,
        "summary": summary,
        "head": head,
        "findings": findings,
        "by_category": by_cat,
        "dims": dims,
        "doctor": doctor,
        "doctor_note": doctor_note,
        "raw_path": raw_path,
    }


@router.get("/qc/report/{run_id}/download")
def qc_report_download(run_id: int):
    """下载这次质检的报告 JSON（**按 run_id**，不放任意路径，避免变成任意文件读取）。"""
    run = db.query_one("SELECT * FROM qc_runs WHERE id=?", (run_id,))
    if not run:
        raise HTTPException(404, "质检记录不存在")
    path = run.get("report_path")
    if not path or not Path(path).is_file():
        raise HTTPException(404, "这次运行没有报告文件（可能已被存储清理）")
    return FileResponse(str(path), filename=Path(path).name, media_type="application/json")


@router.get("/qc/report/{run_id}/download-raw")
def qc_report_download_raw(run_id: int):
    """下载服务器端 lerobot-doctor 的原始 JSON（含 PASS 明细）。"""
    run = db.query_one("SELECT * FROM qc_runs WHERE id=?", (run_id,))
    if not run:
        raise HTTPException(404, "质检记录不存在")
    try:
        summary = json.loads(run.get("summary_json") or "{}")
    except Exception:  # noqa: BLE001
        summary = {}
    raw = str(((summary.get("raw") or {}).get("doctor_json") or ""))
    if not raw or not Path(raw).is_file():
        raise HTTPException(404, "这次运行没有保留原始结果")
    return FileResponse(raw, filename=Path(raw).name, media_type="application/json")


@router.get("/datasets/{dataset_id}/episode_problems")
def episode_problems(dataset_id: int, run_id: int | None = None):
    """按 episode 汇总最近一次质检的问题，供「挑出要排除的 episode」用。

    实现已抽到 `core.qc_checks.episode_problem_summary`，与「数据浏览」页共用同一套
    聚合规则（含"普遍性问题归为数据集级"的处理），避免两处逻辑与阈值漂移。
    只做汇总与建议，**不删任何东西** —— 真正的排除由 preprocess(rule=manual) 生成新数据集。
    """
    from core import pipeline as pipeline_mod
    from core import qc_checks

    data = qc_checks.episode_problem_summary(dataset_id, run_id)
    if data.get("error"):
        raise HTTPException(404, data["error"])
    data["rules"] = [{"key": k, "label": v} for k, v in pipeline_mod.PREPROCESS_RULES.items()]
    return data


@router.get("/qc/runs")
def qc_runs(dataset_id: int | None = None, limit: int = 20):
    if dataset_id:
        return db.query("SELECT * FROM qc_runs WHERE dataset_id=? ORDER BY id DESC LIMIT ?",
                        (dataset_id, limit))
    return db.query("SELECT * FROM qc_runs ORDER BY id DESC LIMIT ?", (limit,))


@router.get("/qc/runs/{run_id}")
def qc_run_detail(run_id: int):
    run = db.query_one("SELECT * FROM qc_runs WHERE id=?", (run_id,))
    if not run:
        raise HTTPException(404, "质检记录不存在")
    run["findings"] = db.query(
        "SELECT * FROM qc_findings WHERE run_id=? ORDER BY"
        " CASE severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, episode_index",
        (run_id,))
    run["summary"] = json.loads(run.get("summary_json") or "{}")
    return run


# --------------------------------------------------------------------------- 标注

@router.get("/annotations")
def list_annotations(dataset_id: int | None = None):
    if dataset_id:
        return db.query("SELECT * FROM annotations WHERE dataset_id=? ORDER BY episode_index",
                        (dataset_id,))
    return db.query("SELECT * FROM annotations ORDER BY id DESC")


@router.put("/annotations/{dataset_id}/{episode_index}")
def save_annotation(dataset_id: int, episode_index: int, payload: dict = Body(default={})):
    if not db.query_one("SELECT id FROM episodes WHERE dataset_id=? AND episode_index=?",
                        (dataset_id, episode_index)):
        raise HTTPException(404, "episode 不存在")
    success = payload.get("success")
    db.execute(
        """INSERT INTO annotations (dataset_id, episode_index, task_text, success, note,
                annotator, updated_at) VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(dataset_id, episode_index) DO UPDATE SET
                task_text=excluded.task_text, success=excluded.success, note=excluded.note,
                annotator=excluded.annotator, updated_at=excluded.updated_at""",
        (dataset_id, episode_index, payload.get("task_text"),
         None if success is None else int(bool(success)), payload.get("note"),
         payload.get("annotator") or "admin", db.now()),
    )
    annotator = payload.get("annotator") or "admin"
    from core import annotators as annotators_mod

    annotators_mod.ensure_known(annotator)
    audit.log("INFO", "数据标注",
              f"数据集 #{dataset_id} ep{episode_index} 保存标注：成功位={success} "
              f"指令={(payload.get('task_text') or '')[:40]!r}",
              ref_type="episode", ref_id=episode_index, actor=annotator,
              detail={"dataset_id": dataset_id, "annotator": annotator})
    return db.query_one("SELECT * FROM annotations WHERE dataset_id=? AND episode_index=?",
                        (dataset_id, episode_index))


# --------------------------------------------------------------------------- 产物

@router.get("/artifacts")
def list_artifacts(kind: str | None = None, limit: int = 50):
    if kind:
        return db.query("SELECT * FROM artifacts WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))
    return db.query("SELECT * FROM artifacts ORDER BY id DESC LIMIT ?", (limit,))


@router.get("/repair/ops")
def repair_ops():
    """可用的修正类预处理操作（界面用来渲染勾选项）。"""
    from core import repair

    return {"ops": [{"key": key, "label": label, "default": key in repair.DEFAULT_OPS}
                    for key, label in repair.OPS.items()]}


@router.get("/datasets/{dataset_id}/versions")
def dataset_versions(dataset_id: int):
    """该数据集的版本历史（新到旧），含内容哈希与父版本。"""
    rows = db.query("SELECT * FROM dataset_versions WHERE dataset_id=? ORDER BY version DESC",
                    (dataset_id,))
    for row in rows:
        row["params"] = json.loads(row.get("params_json") or "{}")
    return {"dataset": db.query_one(
                "SELECT id, name, version, content_hash, derived_by, parent_dataset_id"
                " FROM datasets WHERE id=?", (dataset_id,)),
            "versions": rows}


@router.get("/datasets/{dataset_id}/lineage")
def dataset_lineage(dataset_id: int):
    """血缘链：从最上游的数据集到当前数据集。"""
    chain = []
    current = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    seen = set()
    while current and current["id"] not in seen:
        seen.add(current["id"])
        chain.append({"id": current["id"], "name": current["name"], "version": current["version"],
                      "content_hash": current["content_hash"], "derived_by": current["derived_by"],
                      "episodes": current["total_episodes"], "frames": current["total_frames"]})
        parent = current.get("parent_dataset_id")
        current = db.query_one("SELECT * FROM datasets WHERE id=?", (parent,)) if parent else None
    return {"chain": list(reversed(chain))}


@router.get("/datasets/{dataset_id}/derived")
def dataset_derived(dataset_id: int):
    """由该数据集派生出来的数据集（预处理产物等）。"""
    return db.query(
        "SELECT id, name, version, derived_by, total_episodes, total_frames FROM datasets"
        " WHERE parent_dataset_id=? ORDER BY id", (dataset_id,))


@router.get("/versions/{version_id}/download")
def download_version(version_id: int):
    row = db.query_one("SELECT * FROM dataset_versions WHERE id=?", (version_id,))
    if not row:
        raise HTTPException(404, "版本不存在")
    target = Path(row.get("zip_path") or row.get("path") or "")
    if not target.is_file():
        raise HTTPException(404, f"文件已不存在：{target}")
    return FileResponse(str(target), filename=target.name)


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: int):
    row = db.query_one("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
    if not row:
        raise HTTPException(404, "产物不存在")
    path = Path(row["path"])
    if not path.is_file():
        raise HTTPException(404, f"文件已不存在：{path}")
    return FileResponse(str(path), filename=path.name)


@router.get("/spatial/summary/{dataset_id}")
def spatial_summary(dataset_id: int):
    return spatial.summary(dataset_id)


@router.get("/spatial/{dataset_id}")
def spatial_items(dataset_id: int, episode_index: int | None = None, kind: str | None = None):
    return spatial.items(dataset_id, episode_index, kind)


@router.post("/spatial")
def spatial_create(payload: dict = Body(default={})):
    try:
        return spatial.put(int(payload["dataset_id"]), int(payload["episode_index"]),
                           payload.get("kind"), payload.get("label") or "",
                           payload.get("data") or {}, payload.get("frame_index"),
                           payload.get("annotator") or "admin")
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"标注不合法：{exc}") from exc


@router.delete("/spatial/{annotation_id}")
def spatial_delete(annotation_id: int):
    try:
        return spatial.delete(annotation_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/dedup/report")
def dedup_report(dataset_ids: str | None = None):
    """跨数据集重复 episode 报告（只报告，不删任何东西）。"""
    ids = [int(x) for x in dataset_ids.split(",")] if dataset_ids else None
    return dedup.report(ids)


@router.get("/annotators/roster")
def annotator_roster():
    from core import annotators

    return annotators.roster()


@router.post("/annotators")
def annotator_register(payload: dict = Body(default={})):
    from core import annotators

    try:
        return annotators.register(payload.get("name"), payload.get("role") or "标注员",
                                   payload.get("note"), payload.get("active", True))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/annotators/work/start")
def work_start(payload: dict = Body(default={})):
    from core import annotators

    try:
        return annotators.start_work(payload.get("annotator") or "admin",
                                     payload.get("dataset_id"), payload.get("note"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/annotators/work/stop")
def work_stop(payload: dict = Body(default={})):
    from core import annotators

    try:
        return annotators.stop_work(payload.get("annotator") or "admin", payload.get("note"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/annotators/work/current")
def work_current(annotator: str | None = None):
    from core import annotators

    rows = annotators.open_sessions()
    if annotator:
        rows = [r for r in rows if r["annotator"] == annotator]
    return rows


@router.get("/annotators/stats")
def annotator_stats(days: int = 7, annotator: str | None = None):
    from core import annotators

    return annotators.stats(days, annotator)


@router.delete("/models/{model_id}")
def delete_model(model_id: int, files: bool = False):
    """删除一个训练模型。

    三条保护：
      1. 被评估记录引用的模型 **拒绝删除**（避免评估页出现悬空引用）；
      2. 只有当文件位于 `storage/models` 或 `storage/train_runs` 下才允许删磁盘文件；
      3. 默认只删数据库记录（`files=false`），要删磁盘文件必须显式传 `files=true`。
    """
    import shutil

    row = db.query_one("SELECT * FROM models WHERE id=?", (model_id,))
    if not row:
        raise HTTPException(404, f"模型不存在：{model_id}")

    # 引用保护：只有 evaluations 表里确实有 model_id 列时才校验（不同版本 schema 可能不同）
    refs: list = []
    try:
        refs = db.query("SELECT id FROM evaluations WHERE model_id=?", (model_id,))
    except Exception:  # noqa: BLE001
        refs = []
    if refs:
        raise HTTPException(400, f"该模型被 {len(refs)} 条评估记录引用，请先删除相关评估再删模型")

    removed: list[str] = []
    if files:
        path = Path(row["path"])
        allowed = [db.STORAGE / "models", db.STORAGE / "train_runs"]
        inside = None
        for root in allowed:
            try:
                path.relative_to(root)
                inside = root
                break
            except ValueError:
                continue
        if inside is None:
            raise HTTPException(400, f"出于安全考虑，不删除该路径下的文件：{path}")

        # lerobot：path 形如 <run>/checkpoints/020000/pretrained_model/model.safetensors -> 删整个 run 目录
        if path.parent.name == "pretrained_model" and len(path.parents) > 3:
            target = path.parents[3]
        else:
            target = path
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(f"目录 {target}")
            elif target.is_file():
                target.unlink(missing_ok=True)
                removed.append(f"文件 {target}")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"删除文件失败：{type(exc).__name__}: {exc}") from exc

        # 顺带清掉"打包下载"的缓存 zip
        safe_name = re.sub(r"[^\w.-]+", "-", row["name"])   # py3.10 的 f-string 不能含反斜杠
        package = db.STORAGE / "models" / "_packages" / f"{safe_name}.zip"
        if package.is_file():
            package.unlink(missing_ok=True)
            removed.append(f"文件 {package}")

    db.execute("DELETE FROM models WHERE id=?", (model_id,))
    audit.log("WARN", "模型管理",
              f"删除模型 #{model_id} {row['name']}"
              + (f"，并删除 {len(removed)} 项磁盘文件" if removed else "（仅删记录）"),
              ref_type="model", ref_id=model_id,
              detail={"backend": row["backend"], "dataset_id": row["dataset_id"],
                      "files": files, "removed": removed})
    return {"deleted": model_id, "name": row["name"], "files_removed": removed}


@router.get("/models/{model_id}/download")
def download_model(model_id: int):
    """下载训练产出的模型。

    - `builtin`：直接给 `.npz`（一层权重，本身就是完整产物）。
    - `lerobot`：**打包整个 `pretrained_model` 目录**（`config.json` + `model.safetensors`
      + `train_config.json`）。只给权重文件是不够的 —— 没有 config 就无法重建策略。
      打包结果缓存在 `storage/models/_packages/`，避免每次重复压缩 200MB。
    """
    import zipfile

    row = db.query_one("SELECT * FROM models WHERE id=?", (model_id,))
    if not row:
        raise HTTPException(404, "模型不存在")
    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(404, f"模型文件不在了：{path}")

    if row["backend"] != "lerobot":
        return FileResponse(str(path), filename=path.name,
                            media_type="application/octet-stream")

    source_dir = path.parent                       # .../checkpoints/000200/pretrained_model
    package_dir = db.STORAGE / "models" / "_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    archive = package_dir / (re.sub(r"[^\w.-]+", "-", row["name"]) + ".zip")
    if not archive.is_file() or archive.stat().st_mtime < path.stat().st_mtime:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(source_dir.rglob("*")):
                if item.is_file():
                    zf.write(item, item.relative_to(source_dir))
    audit.log("INFO", "模型下载", f"下载模型 #{model_id} {row['name']}"
              f"（{archive.stat().st_size / 1024 / 1024:.1f}MB，含 config + 权重）",
              ref_type="model", ref_id=model_id)
    return FileResponse(str(archive), filename=archive.name,
                        media_type="application/zip")
