r"""VLM 质检自检：不需要 Qwen 权重、不需要起平台，也不碰真实平台数据。

在项目根目录运行：

    python vlm_selftest.py

它会：

1. 在系统临时目录建一套**独立的** storage（不会写 storage/platform.db）
2. 起一个纯标准库的 mock VLM（OpenAI 兼容，`tools/vlm_mock_server.py`）
3. 造一个真能解码的 LeRobot v2.1 夹具（真 mp4 + 真 parquet）
4. 验证：抽帧拼图 / 两种模式 / 判定与严重级 / 正反问自一致 / 无法解析兜底 /
   服务不可用降级 / 缓存命中与失效 / 非视觉数据集提示

和 smoke_test.py、flow_test.py 一样，它是**活体断言**：任何一条不成立就会打印 FAIL。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# ---- 关键：把持久化路径指到临时目录，必须在导入 vlm_qc 之前完成 ----
from core import db  # noqa: E402

# 默认放在项目 storage 下的临时子目录（不写系统的 %TEMP%，那边可能没有写权限），
# 跑完会整个删掉；用 VLM_SELFTEST_ROOT 可以换位置。
_env_root = os.environ.get("VLM_SELFTEST_ROOT")
TEST_ROOT = (Path(_env_root) if _env_root
             else ROOT / "storage" / "_vlm_selftest")
if TEST_ROOT.exists():
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
TEST_ROOT.mkdir(parents=True, exist_ok=True)
db.STORAGE = TEST_ROOT
db.DB_PATH = TEST_ROOT / "platform.db"
db.REPORT_DIR = TEST_ROOT / "reports"
db.EXPORT_DIR = TEST_ROOT / "exports"

import vlm_mock_server  # noqa: E402
from core import lerobot, vlm_qc  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not condition else ""))


def set_mode(mode: str) -> None:
    """切换 mock 行为。同一个进程里改模块全局变量，下一次请求就生效。"""
    vlm_mock_server.MODE = mode
    time.sleep(0.05)


def with_expects(config: dict) -> None:
    """告诉 mock 每一项的合规答案是 yes 还是 no。

    必须设：`all_pass` 不能假设「正向恒答 yes」—— 清单里出现 `expect: "no"` 的项
    （「有人经过」「开放世界事件」）时，全答 yes 反而意味着这些项全都不合规，
    all_pass 会退化成 all_fail。
    """
    items = config.get("checklist") or vlm_qc.DEFAULT_CHECKLIST
    vlm_mock_server.EXPECTS = [str(it.get("expect") or "yes") for it in items]


def make_dataset(root: Path, frames: int = 24, cameras: int = 1) -> None:
    """造一个**真的能解码**的 LeRobot v2.1 数据集（真 mp4 + 真 parquet）。"""
    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    keys = [f"observation.images.cam{i}" for i in range(cameras)]
    for key in keys:
        (root / "videos" / "chunk-000" / key).mkdir(parents=True, exist_ok=True)
        path = root / "videos" / "chunk-000" / key / "episode_000000.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
        for i in range(frames):
            img = np.full((120, 160, 3), 40, dtype=np.uint8)
            x = int(10 + (130 * i / max(1, frames - 1)))      # 随时间右移的方块
            cv2.rectangle(img, (x, 45), (x + 20, 75), (0, 200, 255), -1)
            cv2.putText(img, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(img)
        writer.release()

    features = {k: {"dtype": "video", "shape": [120, 160, 3],
                    "names": ["height", "width", "channels"],
                    "video_info": {"video.codec": "mp4v", "video.fps": 10.0}} for k in keys}
    features.update({
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["j0", "j1"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["j0", "j1"]},
    })
    info = {
        "codebase_version": "v2.1", "robot_type": "selftest_bot", "fps": 10,
        "total_episodes": 1, "total_frames": frames, "total_tasks": 1,
        "total_videos": cameras, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "把方块推到目标位置"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["把方块推到目标位置"], "length": frames},
                   ensure_ascii=False) + "\n", encoding="utf-8")

    table = pa.table({
        "observation.state": pa.array([[float(i), 0.0] for i in range(frames)],
                                      pa.list_(pa.float32())),
        "action": pa.array([[float(i + 1), 0.0] for i in range(frames)], pa.list_(pa.float32())),
        "timestamp": pa.array([i / 10.0 for i in range(frames)], pa.float32()),
        "frame_index": pa.array(list(range(frames)), pa.int64()),
        "episode_index": pa.array([0] * frames, pa.int64()),
        "index": pa.array(list(range(frames)), pa.int64()),
        "task_index": pa.array([0] * frames, pa.int64()),
        "next.reward": pa.array([0.0] * frames, pa.float32()),
        "next.done": pa.array([False] * (frames - 1) + [True]),
        "next.success": pa.array([False] * (frames - 1) + [True]),
    })
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")


def main() -> int:
    port = 8199
    print(f"临时 storage：{TEST_ROOT}")
    db.init_db()

    ds = TEST_ROOT / "testdata" / "vlm-fixture"
    make_dataset(ds, frames=24, cameras=1)
    meta = lerobot.load_meta(ds)

    dataset_id = db.execute(
        "INSERT INTO datasets (name, path, fps, total_episodes, total_frames, created_at,"
        " refreshed_at) VALUES (?,?,?,?,?,?,?)",
        ("vlm-fixture", str(ds), 10, 1, 24, db.now(), db.now()))
    db.execute("INSERT INTO episodes (dataset_id, episode_index, length, data_rows, video_frames,"
               " created_at) VALUES (?,?,?,?,?,?)", (dataset_id, 0, 24, 24, 24, db.now()))

    vlm_mock_server.MODE = "all_pass"
    server = ThreadingHTTPServer(("127.0.0.1", port), vlm_mock_server.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"
    cfg = vlm_qc.load_config({"base_url": base, "use_cache": False, "timeout_s": 20})
    with_expects(cfg)

    print("\n[1] 抽帧与拼图")
    image_meta = vlm_qc.build_images(ds, meta, 0, cfg)
    check("抽到图", len(image_meta["images"]) == 1)
    check("拼图尺寸合理", image_meta.get("size", "").count("x") == 1, str(image_meta.get("size")))
    indices = image_meta["cameras"][0]["indices"] if image_meta.get("cameras") else []
    check("首帧与末帧都在采样里", bool(indices) and indices[0] == 0 and indices[-1] == 23, str(indices))
    check("内容指纹完整", len(image_meta["sha"]) == 64)
    check("默认抽 6 格", image_meta["tiles"] == 6, f"tiles={image_meta['tiles']}")

    print("\n[2] 多相机（2 路相机 → 2 行）")
    ds2 = TEST_ROOT / "testdata" / "vlm-fixture-2cam"
    make_dataset(ds2, frames=12, cameras=2)
    image_meta2 = vlm_qc.build_images(ds2, lerobot.load_meta(ds2), 0, cfg)
    check("两路相机都被采到", len(image_meta2["cameras"]) == 2,
          str([c["key"] for c in image_meta2["cameras"]]))
    check("每路至少 2 帧", all(len(c["indices"]) >= 2 for c in image_meta2["cameras"]))

    print("\n[3] 后端探活")
    health = vlm_qc.health(cfg)
    check("探活可达", health["ok"], health["detail"])
    check("找到配置里的模型", health.get("model_found") is True, str(health.get("models")))

    print("\n[4] 全合规 → 不该误报")
    set_mode("all_pass")
    findings = vlm_qc.check_episode_vlm(ds, meta, 0, cfg, dataset_id)
    rules = [f["rule"] for f in findings]
    check("没有 error/warning 判定",
          not any(f["severity"] in ("error", "warning") for f in findings), str(findings))
    check("带出场景描述（可回填语言指令）", any("场景描述" in r for r in rules), str(rules))

    print("\n[5] 全不合规 → 问题清单要命中，且是 warning 不是 error")
    set_mode("all_fail")
    findings = vlm_qc.check_episode_vlm(ds, meta, 0, cfg, dataset_id)
    rules = [f["rule"] for f in findings]
    check("任务完成被判不合规", "VLM·任务完成" in rules, str(rules))
    check("动作成败被判不合规", "VLM·动作成败" in rules, str(rules))
    check("有人经过被判不合规", "VLM·旁人经过" in rules, str(rules))
    check("人手介入被判不合规", "VLM·人手介入" in rules, str(rules))
    check("开放世界事件被判不合规", "VLM·开放世界事件" in rules, str(rules))
    check("描述判定为 fail", "VLM·描述判定" in rules, str(rules))
    # 核心保证：打回重采的 error 只能来自结构规则，**不能由模型判定产生**
    check("VLM 判定绝不产生 error（打回权限不交给模型）",
          not any(f["severity"] == "error" for f in findings), str(findings))
    check("不合规判定只落在 warning/info",
          all(f["severity"] in ("warning", "info") for f in findings),
          str([(f["rule"], f["severity"]) for f in findings]))
    check("正反一致时不算「不一致」", not any("正反不一致" in r for r in rules), str(rules))

    print("\n[6] 正反问不一致 → 降级为待人工确认，不误报")
    set_mode("disagree")
    findings = vlm_qc.check_episode_vlm(ds, meta, 0, cfg, dataset_id)
    check("出现正反不一致", any("正反不一致" in f["rule"] for f in findings), str(findings))
    check("不一致判为 info",
          all(f["severity"] == "info" for f in findings if "正反不一致" in f["rule"]),
          str([(f["rule"], f["severity"]) for f in findings]))

    print("\n[7] 模型输出无法解析 → 走「未取到回答」兜底")
    set_mode("unparseable")
    findings = vlm_qc.check_episode_vlm(ds, meta, 0, cfg, dataset_id)
    check("出现未取到回答", any("未取到回答" in f["rule"] for f in findings), str(findings))

    print("\n[8] 后端不可用 → 降级为 warning，绝不抛异常")
    cfg_down = vlm_qc.load_config({"base_url": "http://127.0.0.1:9/v1", "use_cache": False,
                                  "timeout_s": 5})
    try:
        findings = vlm_qc.check_episode_vlm(ds, meta, 0, cfg_down, dataset_id)
        check("没有抛异常", True)
        check("给出服务不可用告警",
              any(f["rule"] == "VLM·服务不可用" and f["severity"] == "warning" for f in findings),
              str(findings))
    except Exception as exc:  # noqa: BLE001
        check("没有抛异常", False, f"{type(exc).__name__}: {exc}")

    print("\n[9] 缓存：同一内容第二次不再调 VLM")
    set_mode("all_fail")
    cfg_cache = vlm_qc.load_config({"base_url": base, "use_cache": True, "timeout_s": 20})
    first = vlm_qc.run_episode(ds, meta, 0, cfg_cache, dataset_id)
    second = vlm_qc.run_episode(ds, meta, 0, cfg_cache, dataset_id)
    check("第一次不是缓存", first.get("cached") is False)
    check("第二次命中缓存", second.get("cached") is True, str(second.get("cached")))
    check("缓存已入库",
          bool(db.query_one("SELECT id FROM vlm_checks WHERE dataset_id=?", (dataset_id,))))

    print("\n[10] 配置改了缓存要失效")
    cfg_changed = vlm_qc.load_config({"base_url": base, "use_cache": True,
                                      "checklist": cfg["checklist"][:2]})
    with_expects(cfg_changed)          # 清单变了，mock 的合规答案也要跟着变
    check("指纹随配置变化", vlm_qc.prompt_hash(cfg_cache) != vlm_qc.prompt_hash(cfg_changed))

    print("\n[11] 两种模式可以单独跑")
    set_mode("all_pass")
    only_describe = vlm_qc.load_config({"base_url": base, "use_cache": False, "mode": "describe",
                                       "timeout_s": 20})
    findings = vlm_qc.check_episode_vlm(ds, meta, 0, only_describe, dataset_id)
    check("describe 模式不产生 checklist 规则",
          not any(f["rule"].startswith("VLM·任务完成") for f in findings), str(findings))
    only_checklist = vlm_qc.load_config({"base_url": base, "use_cache": False, "mode": "checklist",
                                        "emit_description": False, "timeout_s": 20})
    with_expects(only_checklist)
    findings = vlm_qc.check_episode_vlm(ds, meta, 0, only_checklist, dataset_id)
    check("checklist 模式不产生描述判定",
          not any("描述判定" in f["rule"] for f in findings), str(findings))
    with_expects(cfg)                  # 还原成完整清单

    print("\n[12] 非视觉数据集（无 mp4、无内嵌图像）→ info 提示，不报错")
    ds3 = TEST_ROOT / "testdata" / "vlm-fixture-novision"
    make_dataset(ds3, frames=8, cameras=1)
    shutil.rmtree(ds3 / "videos")
    findings = vlm_qc.check_episode_vlm(ds3, lerobot.load_meta(ds3), 0, cfg, None)
    check("给出无可检视觉的 info",
          any("无可检视觉" in f["rule"] and f["severity"] == "info" for f in findings),
          str(findings))

    print("\n[13] 配置读写往返")
    saved = vlm_qc.save_config({"frames": 4})
    check("配置能存下来", saved.get("frames") == 4, str(saved.get("frames")))
    check("问题清单没被改掉", len(saved.get("checklist") or []) == 5)
    vlm_qc.save_config({"frames": 6})

    print("\n[14] 端到端：真实 pipeline.qc 处理器 + VLM（全部落在临时库）")
    from core import jobs, pipeline
    set_mode("all_fail")
    job_id = db.execute(
        "INSERT INTO jobs (kind, status, progress, total, created_at)"
        " VALUES ('qc','running',0,1,?)", (db.now(),))
    result = pipeline.qc(jobs.JobContext(job_id),
                         {"dataset_id": dataset_id, "vlm": True, "vlm_frames": 4,
                          "vlm_mode": "both", "vlm_base_url": base})
    check("质检返回了 run_id", bool(result.get("run_id")), str(result.get("run_id")))
    check("summary 带 VLM 统计", (result.get("vlm") or {}).get("enabled") is True,
          str(result.get("vlm")))
    check("VLM 统计里有模型名", bool((result.get("vlm") or {}).get("model")),
          str(result.get("vlm")))
    vlm_rows = db.query("SELECT * FROM qc_findings WHERE run_id=? AND rule LIKE 'VLM·%'",
                        (result["run_id"],))
    check("VLM 结论已落进 qc_findings", len(vlm_rows) > 0, f"{len(vlm_rows)} 条")
    check("VLM 结论不产生 error（不会误触发打回重采）",
          not any(r["severity"] == "error" for r in vlm_rows),
          str([(r["rule"], r["severity"]) for r in vlm_rows]))
    # 这条专门防"配置里 base_url 被写丢、悄悄回落到默认 ollama 地址"这类问题：
    # 后端没起来时不会报错，只会每条都产出「服务不可用」，很容易被当成"跑通了"。
    check("没有出现服务不可用（说明后端参数确实传到了）",
          not any(r["rule"] == "VLM·服务不可用" for r in vlm_rows),
          str([r["detail"] for r in vlm_rows if r["rule"] == "VLM·服务不可用"]))
    check("质检报告文件已写出", Path(result.get("report", "")).is_file(), result.get("report"))
    if Path(result.get("report", "")).is_file():
        report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
        episode0 = (report.get("episodes") or [{}])[0]
        check("报告里带 VLM 逐问回答与描述", bool(episode0.get("vlm", {}).get("checklist")),
              str(list((episode0.get("vlm") or {}).keys())))
    else:
        check("报告里带 VLM 逐问回答与描述", False, "报告文件不存在")

    print("\n[15] 不带 vlm 的质检应完全不受影响（回归）")
    set_mode("all_fail")
    job_id2 = db.execute(
        "INSERT INTO jobs (kind, status, progress, total, created_at)"
        " VALUES ('qc','running',0,1,?)", (db.now(),))
    plain = pipeline.qc(jobs.JobContext(job_id2), {"dataset_id": dataset_id})
    check("默认不跑 VLM", (plain.get("vlm") or {}).get("enabled") is False,
          str(plain.get("vlm")))
    plain_rows = db.query("SELECT * FROM qc_findings WHERE run_id=? AND rule LIKE 'VLM·%'",
                          (plain["run_id"],))
    check("默认质检不产生 VLM 结论", len(plain_rows) == 0, f"{len(plain_rows)} 条")

    print("\n[16] 后端全程不可用时：任务不能假装成功（防「看着完成、其实没生效」）")
    job_id3 = db.execute(
        "INSERT INTO jobs (kind, status, progress, total, created_at)"
        " VALUES ('qc','running',0,1,?)", (db.now(),))
    dead = pipeline.qc(jobs.JobContext(job_id3),
                       {"dataset_id": dataset_id, "vlm": True, "vlm_frames": 4,
                        "vlm_mode": "both", "vlm_base_url": "http://127.0.0.1:9/v1",
                        # 必须强制不走缓存：前面 [14] 已经问过同一条，
                        # 缓存命中会直接复用当时的结果 —— 那是正确行为（结论确实来自模型），
                        # 但那样就模拟不出"后端全程不可用"的场景了。
                        "vlm_use_cache": False})
    stats = dead.get("vlm") or {}
    check("任务本身仍然完成（结构质检不受影响）", bool(dead.get("run_id")), str(dead.get("run_id")))
    check("标出 VLM 未生效 effective=False", stats.get("effective") is False, str(stats))
    check("不可用条数 = 检查条数", stats.get("unavailable") == stats.get("checked")
          and bool(stats.get("checked")), str(stats))
    check("不能把「服务不可用」算成可疑 flagged=0",
          stats.get("flagged") == 0, f"flagged={stats.get('flagged')}")
    check("给出显式告警文案", bool(stats.get("warning")), str(stats.get("warning")))
    unavailable_rows = db.query(
        "SELECT * FROM qc_findings WHERE run_id=? AND rule='VLM·服务不可用'", (dead["run_id"],))
    check("每条都留下了服务不可用的记录", len(unavailable_rows) == 1,
          f"{len(unavailable_rows)} 条")

    print("\n[17] 任务描述必须进提示词（实测不喂任务，模型会把画面臆测成别的任务）")
    meta0 = meta
    task = vlm_qc.task_text_of(meta0, 0)
    check("能从 episodes.jsonl 取到任务描述", task == "把方块推到目标位置", repr(task))
    im = vlm_qc.build_images(ds, meta0, 0, cfg)
    with_task = vlm_qc.build_prompts(cfg, im, "抓取红色连接器并插入插槽")["checklist"]["text"]
    check("提示词里含任务描述", "抓取红色连接器并插入插槽" in with_task)
    check("提示词里要求以该任务为准", "以这个任务为准" in with_task)
    without = vlm_qc.build_prompts(cfg, im, "")["checklist"]["text"]
    check("没有任务描述时明确说明", "没有提供任务描述" in without)
    desc_with = vlm_qc.build_prompts(cfg, im, "放置物体")["describe"]["text"]
    check("描述式提示词也带上任务", "放置物体" in desc_with)

    print("\n[18] 布尔归一化（实测模型会把 true 返回成字符串 \"true\"）")
    check("_as_bool(\"true\") 为 True", vlm_qc._as_bool("true") is True)
    check("_as_bool(\"false\") 为 False", vlm_qc._as_bool("false") is False)
    check("_as_bool(True) 为 True", vlm_qc._as_bool(True) is True)
    check("_as_bool(1) 为 True", vlm_qc._as_bool(1) is True)
    check("_as_bool(\"不确定\") 为 None", vlm_qc._as_bool("不确定") is None)
    describe, findings = vlm_qc._parse_describe(cfg, {"verdict": "pass", "task_completed": "false",
                                                      "description": "x", "reason": "y"})
    check("字符串 \"false\" 能触发自相矛盾告警",
          any("自相矛盾" in f["rule"] for f in findings), str(findings))
    check("describe 里 task_completed 被归一成 bool", describe["task_completed"] is False,
          repr(describe["task_completed"]))

    print("\n[19] 跨模式矛盾：判别式说没完成、描述式说 pass → 必须标出来")
    set_mode("contradict")
    result = vlm_qc.run_episode(ds, meta0, 0, cfg, None)
    rules = [f["rule"] for f in result["findings"]]
    check("识别出两种模式结论矛盾", "VLM·两种模式结论矛盾" in rules, str(rules))
    confused = [f for f in result["findings"] if f["rule"] == "VLM·两种模式结论矛盾"]
    check("矛盾结论是可追溯的", bool(confused) and "描述式理由" in confused[0]["detail"],
          str(confused[:1]))
    check("矛盾结论带上实测值", bool(confused) and "判别" in (confused[0]["measured"] or ""),
          str(confused[:1]))
    set_mode("all_pass")

    print("\n[20] 默认清单：语义级问题，且 expect=no 的反向问法真的要能判不合规")
    rules = [it.get("rule") for it in vlm_qc.DEFAULT_CHECKLIST]
    check("共 5 项", len(vlm_qc.DEFAULT_CHECKLIST) == 5, str(rules))
    check("覆盖「任务是否执行完」", any("任务完成" in r for r in rules), str(rules))
    check("覆盖「动作成功/失败判定」", any("动作成败" in r for r in rules), str(rules))
    check("覆盖「人手介入操作」", any("人手介入" in r for r in rules), str(rules))
    check("覆盖「旁人经过」", any("旁人经过" in r for r in rules), str(rules))
    check("覆盖「开放世界事件检测」", any("开放世界" in r for r in rules), str(rules))
    # 「人手介入」与「旁人经过」必须是两项：混在一项里问会让模型摇摆（实测正反不一致）
    check("「人手介入」与「旁人经过」是分开的两项",
          any("人手介入" in r for r in rules) and any("旁人经过" in r for r in rules))
    mutual = [it for it in vlm_qc.DEFAULT_CHECKLIST if "人手介入" in (it.get("rule") or "")]
    if mutual:
        q = mutual[0].get("q") or ""
        neg = mutual[0].get("q_neg") or ""
        check("「人手介入」问的是接触操作，不是路过",
              "接触" in q or "接触" in neg, f"q={q}")
        check("「旁人经过」明确排除接触",
              any("没有接触" in (it.get("q") or "")
                  for it in vlm_qc.DEFAULT_CHECKLIST if "旁人经过" in (it.get("rule") or "")))
    check("每项都有正反问法与 expect",
          all(it.get("q") and it.get("q_neg") and it.get("expect") for it in vlm_qc.DEFAULT_CHECKLIST))
    check("两种 expect 都用到（yes 与 no）",
          {it.get("expect") for it in vlm_qc.DEFAULT_CHECKLIST} == {"yes", "no"},
          str({it.get("expect") for it in vlm_qc.DEFAULT_CHECKLIST}))
    check("没有任何一项是 error 级别（VLM 永不触发打回）",
          all((it.get("severity") or "warning") != "error" for it in vlm_qc.DEFAULT_CHECKLIST))

    # expect=no 是生产上从没走过的新路径：模型答 yes = 出问题了 = 必须报
    no_items = [it for it in vlm_qc.DEFAULT_CHECKLIST if it.get("expect") == "no"]
    check("有 expect=no 的项", len(no_items) >= 1, str(len(no_items)))
    if no_items:
        probe = dict(cfg)
        probe["checklist"] = [no_items[0]]
        probe["mode"] = "checklist"
        prompts = vlm_qc.build_prompts(probe, im, task)
        asked = prompts["checklist"]["questions"]
        check("expect=no 的项也生成正反两问", len(asked) == 2, str([q["id"] for q in asked]))
        pos = next((q for q in asked if q["polarity"] == "positive"), None)
        neg = next((q for q in asked if q["polarity"] == "negative"), None)
        check("正向问的是「有没有出问题」这件事",
              pos is not None and pos["q"] == no_items[0]["q"], str(pos))
        # 答 yes（确实有人经过 / 确实有意外事件）→ 应判不合规并产出 finding
        raw = {"answers": [{"id": pos["id"], "answer": "yes", "why": "画面右侧有人走过"},
                           {"id": neg["id"], "answer": "no", "why": "确实有人经过"}]}
        _checks, findings = vlm_qc._parse_checklist(probe, prompts, raw, "{}")
        check("答 yes 被判不合规", len(findings) == 1, str(findings)[:200])
        if findings:
            check("finding 级别 = 该项配置的级别",
                  findings[0]["severity"] == (no_items[0].get("severity") or "warning"),
                  findings[0]["severity"])
            check("expect 记录为 no", findings[0]["expected"] == "no", findings[0]["expected"])
            check("理由被带进 detail", "有人走过" in findings[0]["detail"], findings[0]["detail"])
        # 答 no（没人经过）→ 合规，不该报
        raw2 = {"answers": [{"id": pos["id"], "answer": "no", "why": "全程无人"},
                            {"id": neg["id"], "answer": "yes", "why": "确认无人"}]}
        _c2, f2 = vlm_qc._parse_checklist(probe, prompts, raw2, "{}")
        check("答 no 判合规、不误报", len(f2) == 0, str(f2)[:200])
        # 正反不一致 → 待人工确认，不能蒙混
        raw3 = {"answers": [{"id": pos["id"], "answer": "no", "why": "x"},
                            {"id": neg["id"], "answer": "no", "why": "y"}]}
        _c3, f3 = vlm_qc._parse_checklist(probe, prompts, raw3, "{}")
        check("正反不一致 → 标为待人工确认",
              len(f3) == 1 and "不一致" in f3[0]["rule"], str(f3)[:200])

    print("\n" + "=" * 66)
    total = len(PASSED) + len(FAILED)
    print(f"通过 {len(PASSED)} / {total}")
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print(f"  - {name}")
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
