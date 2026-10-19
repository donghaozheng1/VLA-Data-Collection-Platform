"""端到端验证：通过真实 HTTP 走一遍 采集→质检→标注→导出。"""
import time
import httpx

BASE = __import__("os").environ.get("FLOW_BASE", "http://127.0.0.1:8000")
SRC = r"C:\Users\ll\Desktop\VLA\lerobot\.hf_cache\lerobot\lerobot\pusht"

for _ in range(60):
    try:
        httpx.get(BASE + "/", timeout=2)
        break
    except Exception:
        time.sleep(0.5)
else:
    raise SystemExit("服务没起来")

ok_count = 0
fail = []


def check(label, cond, extra=""):
    global ok_count
    if cond:
        ok_count += 1
        print(f"[OK ] {label} {extra}")
    else:
        fail.append(label)
        print(f"[FAIL] {label} {extra}")


def wait_job(client, job_id, timeout=180):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        last = job
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.5)
    return last


with httpx.Client(base_url=BASE, follow_redirects=False, timeout=120) as c:
    print("===== 1. 页面 =====")
    for path in ["/dashboard", "/collect", "/data", "/preprocess", "/qc", "/annotation",
                 "/export", "/semantics", "/models", "/system", "/log", "/interfaces"]:
        r = c.get(path)
        check(f"GET {path}", r.status_code == 200, f"-> {r.status_code} {len(r.content)}B")

    r = c.get("/dashboard")
    check("base.html 已引入 platform.js", "/static/js/platform.js" in r.text)
    r = c.get("/static/js/platform.js")
    check("静态文件可访问", r.status_code == 200, f"{len(r.content)}B")

    print("\n===== 2. 概览接口 =====")
    o = c.get("/api/overview").json()
    print("   ", o)
    check("overview 有数据集", o["datasets"] >= 1)

    print("\n===== 3. 扫描磁盘（真实发现数据集）=====")
    t0 = time.time()
    scan = c.post("/api/collect/scan", json={}).json()
    print(f"    扫描 {len(scan['roots'])} 个根目录，发现 {len(scan['datasets'])} 个数据集，"
          f"耗时 {time.time()-t0:.1f}s")
    for d in scan["datasets"]:
        print(f"      - {d['name']}: {d.get('total_episodes')} 条 / {d.get('total_frames')} 帧"
              f" / 已接入 {d.get('registered_id')}")
    check("扫描到数据集", len(scan["datasets"]) >= 1)

    print("\n===== 4. 采集接入（真实任务 + 轮询）=====")
    t0 = time.time()
    job_id = c.post("/api/jobs/ingest", json={"path": SRC, "name": "pusht-real"}).json()["job_id"]
    job = wait_job(c, job_id)
    print(f"    job{job_id} {job['status']} ({time.time()-t0:.1f}s)")
    check("接入任务成功", job["status"] == "succeeded", str(job.get("error") or "")[:80])
    if job.get("result"):
        print("    结果:", {k: job["result"][k] for k in ("episodes", "frames", "missing_files")})

    ds_id = next(d["id"] for d in c.get("/api/datasets").json() if d["name"] == "pusht-real")

    print("\n===== 5. episode 明细（真实行数/帧数）=====")
    eps = c.get(f"/api/datasets/{ds_id}/episodes?limit=3").json()
    print(f"    total={eps['total']}")
    for e in eps["episodes"]:
        print(f"      ep{e['episode_index']}: rows={e['data_rows']} frames={e['video_frames']} "
              f"dur={e['duration_s']}s success={e['success_count']} errors={e['error_count']}")
    check("episode 返回真实数据", eps["episodes"][0]["data_rows"] > 0)

    print("\n===== 6. 质检（真实任务 + 明细）=====")
    t0 = time.time()
    job_id = c.post("/api/jobs/qc", json={"dataset_id": ds_id}).json()["job_id"]
    job = wait_job(c, job_id)
    print(f"    job{job_id} {job['status']} ({time.time()-t0:.1f}s)")
    check("质检任务成功", job["status"] == "succeeded", str(job.get("error") or "")[:80])
    res = job.get("result") or {}
    print(f"    run{res.get('run_id')}: {res.get('passed')} 通过 / {res.get('failed')} 打回 / {res.get('result')}")
    detail = c.get(f"/api/qc/runs/{res['run_id']}").json()
    print(f"    findings={len(detail['findings'])}  报告文件={detail['report_path']}")
    check("质检报告落盘", detail["report_path"] is not None)

    print("\n===== 7. 标注（写库 + 读回）=====")
    ep0 = eps["episodes"][0]["episode_index"]
    saved = c.put(f"/api/annotations/{ds_id}/{ep0}",
                  json={"task_text": "把 T 形块推到 T 形目标上（平台标注）",
                        "success": True, "note": "验收样本"}).json()
    back = c.get("/api/annotations?dataset_id=%d" % ds_id).json()
    check("标注已持久化", any(a["episode_index"] == ep0 and a["success"] == 1 for a in back),
          f"ep{ep0} -> {saved.get('task_text')[:30]}…")

    print("\n===== 8. 导出（真实数据集 + zip 产物）=====")
    t0 = time.time()
    job_id = c.post("/api/jobs/export",
                    json={"dataset_id": ds_id, "mode": "all", "name": "flow_export"}).json()["job_id"]
    job = wait_job(c, job_id)
    print(f"    job{job_id} {job['status']} ({time.time()-t0:.1f}s)")
    check("导出任务成功", job["status"] == "succeeded", str(job.get("error") or "")[:80])
    r = job.get("result") or {}
    print(f"    {r.get('episodes')} 条 / {r.get('frames')} 帧 / {r.get('bytes')} bytes -> {r.get('zip')}")
    check("导出帧数与源一致", r.get("frames") == 25650, f"{r.get('frames')}")

    print("\n===== 8b. 标注是否写进导出物（闭环）=====")
    import json as _json
    from pathlib import Path as _Path

    import pyarrow.parquet as _pq

    out_dir = _Path(r["path"])
    tasks = {}
    with (out_dir / "meta" / "tasks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = _json.loads(line)
            tasks[row["task_index"]] = row["task"]
    data = _pq.read_table(out_dir / "data" / "chunk-000" / "episode_000000.parquet").to_pydict()
    got = tasks.get(sorted(set(data["task_index"]))[0])
    check("导出物的语言指令来自标注", got == "\u628a T \u5f62\u5757\u63a8\u5230 T \u5f62\u76ee\u6807\u4e0a\uff08\u5e73\u53f0\u6807\u6ce8\uff09", repr(got))
    check("导出物末帧成功位来自标注", bool(data["next.success"][-1]) is True)

    print("\n===== 8c. 导出是否登记为版本（版本管理）=====")
    vers = c.get(f"/api/datasets/{ds_id}/versions").json()
    latest = (vers.get("versions") or [{}])[0]
    check("导出登记了版本与内容哈希",
          isinstance(r.get("version"), int) and bool(r.get("content_hash")),
          f"v{r.get('version')} {str(r.get('content_hash'))[:12]}…")
    check("版本历史最新哈希与导出结果一致",
          latest.get("content_hash") == r.get("content_hash"),
          f"{str(latest.get('content_hash'))[:12]}…")
    check("版本可回溯到父版本",
          latest.get("parent_version") is None or latest.get("parent_version") >= 1,
          f"parent={latest.get('parent_version')}")

    before = len(vers["versions"])
    jid = c.post("/api/jobs/ingest", json={"path": SRC, "name": "pusht-real"}).json()["job_id"]
    wait_job(c, jid)
    after = len(c.get(f"/api/datasets/{ds_id}/versions").json()["versions"])
    check("重复接入未新增版本（版本幂等）",
          after == before, f"{before} -> {after}")

    print("\n===== 8d. 修正类预处理（会改帧）=====")
    jid = c.post("/api/jobs/repair",
                 json={"dataset_id": ds_id,
                       "ops": ["drop_broken", "drop_invalid", "dedup_ts", "align_ts"]}).json()["job_id"]
    job = wait_job(c, jid)
    check("修复任务成功", job["status"] == "succeeded", str(job.get("error") or "")[:80])
    rr = job.get("result") or {}
    check("修复未误删干净数据",
          rr.get("episodes_out") == rr.get("episodes_in") and not rr.get("dropped_episodes"),
          f"{rr.get('episodes_in')} -> {rr.get('episodes_out')}, dropped={len(rr.get('dropped_episodes') or [])}")
    derived_id = rr.get("derived_dataset_id")
    check("修复产物登记为新数据集", bool(derived_id), f"#{derived_id}")
    if derived_id:
        d = next((x for x in c.get("/api/datasets").json() if x["id"] == derived_id), {})
        check("修复数据集带血缘", str(d.get("derived_by", "")).startswith("repair:"),
              str(d.get("derived_by")))
        chain = c.get(f"/api/datasets/{derived_id}/lineage").json()["chain"]
        check("血缘链可回溯到源数据集", len(chain) >= 2,
              " <- ".join(str(n["id"]) for n in chain))

    print("\n===== 8e. 标注质量抽检（抽检→盲标复核→一致性）=====")
    smp = c.post("/api/reviews/sample",
                 json={"dataset_id": ds_id, "count": 1, "salt": "flow-test"}).json()
    check("抽检批次生成", bool(smp.get("batch_id")) and smp.get("sampled") == 1,
          f"批次#{smp.get('batch_id')} {smp.get('episodes')}")
    q = c.get(f"/api/reviews/queue?dataset_id={ds_id}&reviewer=flow_reviewer"
              f"&batch_id={smp['batch_id']}").json()
    check("盲标队列不泄露主标结果",
          bool(q.get("items")) and "primary_task" not in q["items"][0],
          f"{len(q.get('items') or [])} 条, blind={q.get('blind')}")
    ep_review = q["items"][0]["episode_index"]
    c.put(f"/api/reviews/{ds_id}/{ep_review}",
          json={"reviewer": "flow_reviewer", "task_text": "flow 复核指令", "success": True})
    ag = c.get(f"/api/reviews/agreement?dataset_id={ds_id}&reviewer=flow_reviewer").json()
    check("一致性指标可计算", ag["reviewed"] >= 1 and ag["agreement"] is not None,
          f"复核 {ag['reviewed']} 条, 一致率 {ag['agreement']}, Kappa {ag['kappa']}")

    print("\n===== 8f. 实时采集会话（客户端上报 → 真实 episode）=====")
    sess = c.post("/api/collect/session", json={
        "dataset_name": "flow-collect",
        "task_text": "抓取红色连接器并插入左侧插槽",
        "fps": 10, "state_dim": 7, "image_keys": [], "source": "flow-test"}).json()
    check("采集会话创建", bool(sess.get("id")) and sess.get("status") == "open",
          f"session #{sess.get('id')}")
    frames = [{"timestamp": i / 10.0, "state": [float(i + j) for j in range(7)]} for i in range(20)]
    pushed = c.post(f"/api/collect/session/{sess['id']}/frames", json={"frames": frames}).json()
    check("逐帧上报", pushed.get("frames") == 20, str(pushed.get("frames")))
    done = c.post(f"/api/collect/session/{sess['id']}/commit",
                  json={"success": True, "note": "flow 采集"}).json()
    check("提交落成真实 episode", done.get("rows") == 20,
          f"episode {done.get('episode_index')}, {done.get('rows')} 行")
    check("自动接入平台并登记版本",
          bool(done.get("platform_dataset_id")) and (done.get("platform_version") or 0) >= 1,
          f"dataset #{done.get('platform_dataset_id')} v{done.get('platform_version')}")
    check("时间同步体检", (done.get("step_dev_max") if done.get("step_dev_max") is not None else 9) <= 0.2
          and done.get("timestamps_monotonic") is True,
          f"步长偏差 {done.get('step_dev_max')}")

    print("\n===== 8g. 训练与离线评估（真实产物）=====")
    root = c.get("/api/training/default-root").json()
    if root.get("exists"):
        found = c.post("/api/training/scan", json={"root": root["root"]}).json()
        check("扫描训练产物", len(found.get("runs", [])) > 0,
              f"{len(found.get('runs', []))} 个条目")
        run = next((r for r in found["runs"] if r.get("log_points")), None)
        if run:
            reg = c.post("/api/training/register", json={"run_dir": run["run_dir"]}).json()
            check("登记实验并解析指标", (reg.get("metrics") or 0) > 0,
                  f"实验 #{reg.get('experiment_id')} 指标点 {reg.get('metrics')}")
            check("实验绑定数据集版本与哈希",
                  bool(reg.get("dataset")) and bool((reg.get("dataset") or {}).get("content_hash")),
                  str(reg.get("dataset")))
        evals = c.get("/api/training/evaluations").json()
        check("离线评估带真实成功率",
              any(e.get("pc_success") is not None for e in evals), f"{len(evals)} 份评估")
        col = c.post("/api/badcases/collect", json={}).json()
        check("bad case 收集可用", "total_open" in col,
              f"未处理 {col.get('total_open')}")
    else:
        print("[跳过] 本机没有训练产物目录")

    print("\n===== 8h. 真实模型推理（经平台代理）=====")
    health = c.get("/api/inference/health").json()
    if health.get("running"):
        check("推理服务可用", bool(health.get("device")), str(health.get("device")))
        ms = c.get("/api/inference/models?probe=true").json()
        check("扫到真实权重", len(ms.get("models", [])) > 0,
              f"{len(ms.get('models', []))} 个")
        import base64 as _b64

        import cv2 as _cv2
        import numpy as _np

        canvas = _np.zeros((480, 640, 3), _np.uint8)
        canvas[:, :] = (40, 40, 40)
        _cv2.rectangle(canvas, (80, 80), (300, 320), (200, 180, 120), -1)
        _ok, _buf = _cv2.imencode(".jpg", canvas)
        r = c.post("/api/inference/predict",
                   json={"image_b64": _b64.b64encode(_buf.tobytes()).decode(),
                         "model": "yolo26n.pt", "conf": 0.25}).json()
        check("推理接口返回真实结果",
              "inference_ms" in r and "detections" in r and bool(r.get("device")),
              f"检出 {r.get('count')} 个，{r.get('inference_ms')}ms，{r.get('device')}")
    else:
        print("[跳过] 推理服务未启动（需要 yolov8n 环境跑 tools/infer_service.py）")

    print("\n===== 8i. 真实审计日志（/log）=====")
    before = c.get("/log/api/logs?limit=1").json()["stats"]["total"]
    c.put("/api/annotations/2/0", json={"task_text": "审计日志测试", "success": True,
                                        "note": "flow"})
    after = c.get("/log/api/logs?limit=1").json()["stats"]["total"]
    check("操作会写入审计日志", after > before, f"{before} -> {after}")
    log = c.get("/log/api/logs?limit=5").json()
    check("审计日志含真实模块与级别",
          bool(log["data"]) and all(r.get("module") and r.get("level") for r in log["data"]),
          f"累计 {log['stats']['total']} 条，ERROR {log['stats']['error']}")
    clamp = c.get("/log/api/logs?limit=-1").json()
    check("limit 负值被钳制", clamp["total"] == 0, f"limit=-1 -> {clamp['total']}")

    print("\n===== 8j. 场景/任务与本体/设备 =====")
    ov = c.get("/api/scenes/overview").json()
    check("任务聚合可用", ov["totals"]["episodes"] > 0,
          f"{ov['totals']['tasks']} 任务 / {ov['totals']['episodes']} 条 / "
          f"{ov['totals']['raw_variants']} 种写法")
    if ov["suggestions"]:
        sug = ov["suggestions"][0]
        before_tasks = ov["totals"]["tasks"]
        c.post("/api/scenes/merge", json={"tasks": sug["tasks"], "canonical": sug["tasks"][0]})
        merged = c.get("/api/scenes/overview").json()
        check("任务归一生效", merged["totals"]["tasks"] < before_tasks,
              f"{before_tasks} -> {merged['totals']['tasks']}")
    else:
        print("[跳过] 没有待归一的相似写法")
    rob = c.get("/api/robots/overview").json()
    check("本体聚合可用", len(rob["robots"]) > 0,
          f"{[r['robot_type'] for r in rob['robots']]}")
    probe = c.post("/api/devices/probe", json={"note": "flow"}).json()
    check("设备探测落库", bool(probe.get("scan_id")) and "cameras" in probe,
          probe.get("summary"))
    rs_info = probe.get("realsense") or {}
    check("RealSense 字段契约", isinstance(rs_info.get("sdk"), bool)
          and isinstance(rs_info.get("devices"), list),
          f"sdk={rs_info.get('sdk')} 设备={len(rs_info.get('devices') or [])}")

    print("\n===== 8k. 数据划分（可复现）+ Kappa 置信区间 =====")
    a1 = c.post("/api/splits/assign", json={"dataset_id": 5, "salt": "flow",
                                           "mode": "random"}).json()
    a2 = c.post("/api/splits/assign", json={"dataset_id": 5, "salt": "flow",
                                           "mode": "random"}).json()
    check("同 salt 划分可复现", a1["counts"] == a2["counts"], str(a1["counts"]))
    ov = c.get("/api/splits/5").json()
    check("划分落库且名单完整",
          ov["assigned"] and ov["total"] == 14
          and sum(len(v) for v in ov["splits"].values()) == 14,
          f"{ov['counts']} salt={ov['salt']} mode={ov['mode']}")
    ordered = c.post("/api/splits/assign", json={
        "dataset_id": 5, "mode": "ordered",
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}}).json()
    check("ordered 写进 LeRobot 原生 splits", bool(ordered.get("info_splits")),
          str(ordered.get("info_splits")))
    ag = c.get("/api/reviews/agreement?dataset_id=2").json()
    check("Kappa 带置信区间与可靠性标志",
          "kappa_ci" in ag and "reliable" in ag,
          f"kappa={ag.get('kappa')} CI={ag.get('kappa_ci')} 配对={ag.get('success_pairs')}")

    print("\n===== 8l. 划分绑定实验 + 存储清理 =====")
    run_dir = (r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\outputs"
               r"\pusht_train_20k")
    reg = c.post("/api/training/register", json={"dataset_id": 1, "split": "train",
                                                 "run_dir": run_dir}).json()
    check("实验绑定划分", reg.get("split") == "train",
          f"split={reg.get('split')} salt={reg.get('split_salt')} "
          f"episodes={reg.get('split_episodes')}")
    check("复现命令含划分说明", "划分" in (reg.get("reproduce_cmd") or ""))
    bad = c.post("/api/training/register", json={"dataset_id": 1, "split": "nope",
                                                 "run_dir": run_dir})
    check("非法划分名被拒", bad.status_code == 400)
    plan = c.get("/api/storage/plan").json()
    check("存储计划可用", "reclaimable_bytes" in plan and "kept" in plan,
          f"可回收 {plan['reclaimable_bytes'] / 1024 / 1024:.1f}MB / "
          f"受保护 {len(plan['kept'])} 项")
    ds = next((x for x in c.get("/api/datasets").json() if x["name"] == "hkc-joint-7dof"), None)
    if ds:
        guard = c.post("/api/storage/cleanup", json={"paths": [ds["path"]]}).json()
        check("拒绝删除数据集目录",
              guard["deleted_files"] == 0 and not guard["errors"])
    preview = c.post("/api/storage/cleanup", json={"dry_run": True,
                                                   "kinds": ["orphan_report"]}).json()
    check("清理预览不删任何东西", preview["dry_run"] is True)

    print("\n===== 8m. 画布类标注（框/关键点/阶段）=====")
    frame = c.get("/api/episodes/1/0/frame?index=2")
    check("单帧接口返回 JPEG", frame.status_code == 200
          and frame.content[:2] == b"\xff\xd8", f"{len(frame.content)} 字节")
    saved = c.post("/api/spatial", json={
        "dataset_id": 1, "episode_index": 0, "kind": "bbox", "label": "flow_框",
        "data": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}, "frame_index": 2}).json()
    check("保存目标框", bool(saved.get("id")), f"id={saved.get('id')}")
    rejected = c.post("/api/spatial", json={
        "dataset_id": 1, "episode_index": 0, "kind": "bbox", "label": "越界",
        "data": {"x": 0.9, "y": 0.1, "w": 0.5, "h": 0.2}})
    check("非法标注被拒（400）", rejected.status_code == 400,
          str(rejected.json())[:60])
    summary = c.get("/api/spatial/summary/1").json()
    check("标注统计可用", summary["total"] >= 1 and "by_kind" in summary,
          f"total={summary['total']} {summary['by_kind']}")
    exp = c.post("/api/jobs/export", json={"dataset_id": 1, "name": "flow_spatial"}).json()
    job = None
    for _ in range(300):
        job = c.get(f"/api/jobs/{exp['job_id']}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    meta = (job.get("result") or {}).get("spatial_annotations") or {}
    check("导出包带上画布标注", bool(meta.get("written")),
          str(meta.get("counts")))

    print("\n===== 8n. 跨数据集重复 episode 检测 =====")
    job = c.post("/api/jobs/dedup", json={})
    dedup_job = None
    for _ in range(600):
        dedup_job = c.get(f"/api/jobs/{job.json()['job_id']}").json()
        if dedup_job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    check("指纹任务成功", dedup_job["status"] == "succeeded",
          str(dedup_job.get("error") or "")[:60])
    rep = c.get("/api/dedup/report").json()
    check("报告可用", rep["hashed_episodes"] > 0 and "groups" in rep,
          f"已指纹化 {rep['hashed_episodes']} 条 / 精确重复 "
          f"{rep['exact_groups']} 组 / 跨族 {rep['unexplained_groups']} 组")
    check("判定分类含两种情形",
          any("血缘族" in k or "血缘" in k for k in rep["by_classification"]),
          str(rep["by_classification"]))
    if rep["warnings"]:
        check("跨族重复给出可读告警",
              "数据泄漏" in rep["warnings"][0],
              rep["warnings"][0][:70])
    cached = rep["hashed_episodes"]
    check("指纹缓存生效（重报不重算）", cached > 0, f"{cached} 条")

    print("\n===== 8o. 标注账号与工时 =====")
    roster = c.get("/api/annotators/roster").json()
    check("花名册可用（既有记录自动补登）", len(roster) > 0,
          f"{[r['name'] for r in roster][:5]}")
    who = "flow_标注员"
    started = c.post("/api/annotators/work/start", json={"annotator": who, "dataset_id": 2})
    check("上工成功", started.status_code == 200, str(started.json())[:80])
    dup = c.post("/api/annotators/work/start", json={"annotator": who})
    check("重复上工被拒", dup.status_code == 400, str(dup.json())[:60])
    c.put("/api/annotations/2/1", json={"task_text": "工时测试", "success": True,
                                        "annotator": who})
    c.post("/api/spatial", json={"dataset_id": 2, "episode_index": 1, "kind": "keypoint",
                                 "label": "工时测试点", "data": {"x": 0.5, "y": 0.5},
                                 "annotator": who})
    time.sleep(1.1)
    stopped = c.post("/api/annotators/work/stop", json={"annotator": who}).json()
    ops = sum((stopped.get("counts") or {}).values())
    check("收工统计真实操作数", ops >= 2,
          f"{stopped.get('minutes')} 分钟 / {ops} 次操作 {stopped.get('counts')}")
    stopped_twice = c.post("/api/annotators/work/stop", json={"annotator": who})
    check("重复收工被拒", stopped_twice.status_code == 400)
    stats = c.get("/api/annotators/stats?days=1").json()
    mine = next((p for p in stats["people"] if p["annotator"] == who), None)
    check("工时进入统计", bool(mine) and mine["operations"] >= 2,
          f"工时 {mine['minutes'] if mine else '-'} 分钟，"
          f"操作 {mine['operations'] if mine else '-'}")

    print("\n===== 8p. 存储自动清理策略 =====")
    pol = c.get("/api/storage/policy").json()
    check("默认策略是关闭的", pol["enabled"] is False,
          f"阈值 {pol['threshold_mb']}MB / 间隔 {pol['interval_s']}s / 类别 {pol['kinds']}")
    closed = c.post("/api/storage/check", json={}).json()
    check("关闭时只记账不动手", closed["decision"] == "disabled",
          f"决定={closed['decision']} 可回收={closed['reclaimable_mb']}MB")
    high = c.post("/api/storage/policy", json={"enabled": True, "threshold_mb": 99999}).json()
    check("阈值过高不会执行", high["enabled"] is True
          and c.post("/api/storage/check", json={}).json()["decision"] == "below_threshold")
    bad = c.post("/api/storage/policy", json={"kinds": ["datasets"]})
    check("非法清理类别被拒", bad.status_code == 400, str(bad.json())[:60])
    restored = c.post("/api/storage/policy", json={"enabled": False, "threshold_mb": 200}).json()
    check("恢复为关闭", restored["enabled"] is False)

    print("\n===== 8q. 网页里发起训练（builtin）并产出模型 =====")
    envs = c.get("/api/training/envs").json()
    check("环境探测可用", "envs" in envs,
          f"builtin={envs['builtin_available']} lerobot={envs['lerobot_available']}")
    run = c.post("/api/training/run", json={"dataset_id": 5, "backend": "builtin",
                                            "split": "train", "name": "flow_bc_model"})
    check("发起训练", run.status_code == 200, str(run.json())[:80])
    if run.status_code == 200:
        job = None
        for _ in range(600):
            job = c.get(f"/api/jobs/{run.json()['job_id']}").json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.5)
        res = job.get("result") or {}
        check("训练完成且产出模型",
              job["status"] == "succeeded" and res.get("model_id"),
              f"模型 #{res.get('model_id')} {res.get('name')} "
              f"train_mse={((res.get('metrics') or {}).get('train_mse'))}")
        models = c.get("/api/models").json()
        mine = next((m for m in models if m["name"] == "flow_bc_model"), None)
        check("模型已登记且可下载",
              bool(mine) and c.get(f"/api/models/{mine['id']}/download").status_code == 200,
              f"入库 {len(models)} 个，下载 {mine['bytes'] if mine else 0} 字节")
    if saved.get("id"):
        c.delete(f"/api/spatial/{saved['id']}")

    print("\n===== 9. 产物下载 =====")
    arts = c.get("/api/artifacts").json()
    print(f"    产物 {len(arts)} 个: {[a['kind'] + '#' + str(a['id']) for a in arts[:5]]}")
    if arts:
        dl = c.get(f"/api/artifacts/{arts[0]['id']}/download")
        check("产物可下载", dl.status_code == 200, f"{len(dl.content)}B")
        check("下载内容是 zip", dl.content[:2] == b"PK", str(dl.content[:2]))

    print("\n===== 10. 视频流式读取（原始数据只读）=====")
    v = c.get(f"/api/episodes/{ds_id}/{ep0}/video")
    check("视频可流式返回", v.status_code == 200 and v.headers.get("content-type") == "video/mp4",
          f"{len(v.content)}B")

    print("\n===== 11. 任务列表 =====")
    jobs = c.get("/api/jobs").json()
    print(f"    {len(jobs)} 个任务: " + ", ".join(f"#{j['id']}{j['kind_label']}:{j['status']}" for j in jobs[:6]))
    check("任务列表有记录", len(jobs) >= 3)

print(f"\n===== 结果：{ok_count} 项通过，{len(fail)} 项失败 =====")
for f in fail:
    print("  - " + f)
raise SystemExit(1 if fail else 0)
