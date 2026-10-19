#!/usr/bin/env python3
"""无真机时的全流程演练：拿仿真数据集把平台整条链走一遍，并逐步打印真实输入/输出。

前提：先用 `tools/sim_collect.py` 采好（或本脚本自动采）。

    python tools/sim_flow_tour.py

依次执行：采集 → 质检 → 修正类预处理 → 再质检 → 标注 → 画布 → 抽检与一致性 →
划分 → 导出（并校验交付包内容）→ 训练登记 → bad case 回灌 → 重复检测 → 存储清理计划。
每一步都断言关键事实，最后给出通过/失败统计。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"   [{'OK ' if ok else 'FAIL'}] {name} {detail}")


def wait_job(c: httpx.Client, job_id: int, timeout_s: int = 600) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.5)
    return {"status": "timeout", "error": "等待任务超时"}


def main() -> int:
    clean, broken = "sim-realman7-pickplace", "sim-broken"
    with httpx.Client(base_url=BASE, timeout=600) as c:
        if c.get("/api/overview").status_code != 200:
            raise SystemExit("平台没起来")

        # ---------- 0. 采集（若还没有） ----------
        print("\n===== 0. 仿真采集 =====")
        existing = {d["name"] for d in c.get("/api/datasets").json()}
        if clean not in existing:
            subprocess.run([sys.executable, str(ROOT / "tools" / "sim_collect.py"),
                            "--dataset", clean, "--episodes", "4", "--seconds", "6"], check=False)
        if broken not in existing:
            subprocess.run([sys.executable, str(ROOT / "tools" / "sim_collect.py"),
                            "--dataset", broken, "--episodes", "2", "--seconds", "4",
                            "--defect", "backwards@1"], check=False)
        datasets = {d["name"]: d for d in c.get("/api/datasets").json()}
        clean_ds, broken_ds = datasets[clean], datasets[broken]
        step("仿真数据集已接入", clean_ds["total_episodes"] == 4 and clean_ds["total_videos"] == 4,
             f"#{clean_ds['id']} {clean_ds['total_episodes']} 条 / {clean_ds['total_videos']} 视频 "
             f"/ v{clean_ds['version']} / hash {str(clean_ds['content_hash'])[:10]}")

        # ---------- 1. 质检（干净集应通过） ----------
        print("\n===== 1. 质检：干净集 =====")
        job = wait_job(c, c.post("/api/jobs/qc", json={"dataset_id": clean_ds["id"]}).json()["job_id"])
        res = job.get("result") or {}
        step("干净集质检通过", res.get("failed") == 0,
             f"{res.get('passed')} 通过 / {res.get('failed')} 打回 · {res.get('result')}")

        # ---------- 2. 质检（缺陷集应打回，并指出具体规则） ----------
        print("\n===== 2. 质检：注入缺陷的集合 =====")
        job = wait_job(c, c.post("/api/jobs/qc", json={"dataset_id": broken_ds["id"]}).json()["job_id"])
        res = job.get("result") or {}
        detail = c.get(f"/api/qc/runs/{res['run_id']}").json()
        rules = sorted({f["rule"] for f in detail["findings"] if f["severity"] == "error"})
        step("缺陷集被打回并指出规则", res.get("failed", 0) > 0 and rules,
             f"{res.get('passed')} 通过 / {res.get('failed')} 打回，命中规则：{rules}")
        for f in detail["findings"][:3]:
            print(f"        [{f['severity']}] {f['rule']}: {f['detail'][:78]}")

        # ---------- 3. 修正类预处理 ----------
        print("\n===== 3. 修正类预处理（真改帧）=====")
        ops = [o["key"] for o in c.get("/api/repair/ops").json()["ops"]]
        job = wait_job(c, c.post("/api/jobs/repair", json={
            "dataset_id": broken_ds["id"], "name": f"{broken}_fixed",
            "ops": ["dedup_ts", "align_ts"]}).json()["job_id"])
        rep = job.get("result") or {}
        step("修复产出新数据集", job["status"] == "succeeded" and rep.get("episodes_out"),
             f"可用修复操作 {len(ops)} 个 {ops}，{rep.get('episodes_in')} → {rep.get('episodes_out')} 条，"
             f"重建视频 {rep.get('videos_rebuilt')} 个")
        fixed = next((d for d in c.get("/api/datasets").json()
                      if d["name"] == f"{broken}_fixed"), None)
        step("修复结果已接入且带血缘", bool(fixed) and fixed.get("derived_by"),
             f"#{fixed['id'] if fixed else '-'} derived_by={fixed.get('derived_by') if fixed else '-'}")
        if fixed:
            job = wait_job(c, c.post("/api/jobs/qc", json={"dataset_id": fixed["id"]}).json()["job_id"])
            res = job.get("result") or {}
            step("修复后再质检通过", res.get("failed") == 0,
                 f"{res.get('passed')} 通过 / {res.get('failed')} 打回")

        # ---------- 4. 标注（语言指令 + 成功位） ----------
        print("\n===== 4. 标注：语言指令 + 成功位 =====")
        episodes = c.get(f"/api/datasets/{clean_ds['id']}/episodes?limit=20").json()["episodes"]
        for index, ep in enumerate(episodes):
            text = ("[仿真] 抓取蓝色方块并放入右侧插槽" if index < 3
                    else "[仿真] 抓取蓝色方块（本条为失败样本：方块中途掉落）")
            c.put(f"/api/annotations/{clean_ds['id']}/{ep['episode_index']}",
                  json={"task_text": text, "success": index < 3, "note": "仿真标注",
                        "annotator": "张工"})
        ann = c.get(f"/api/annotations?dataset_id={clean_ds['id']}").json()
        annotations = ann if isinstance(ann, list) else ann.get("annotations", [])
        ok_count = sum(1 for a in annotations if a["success"] == 1)
        step("标注写入平台库", len(annotations) == 4,
             f"{len(annotations)} 条标注，其中成功 {ok_count} 条 / 失败 {len(annotations) - ok_count} 条")

        # ---------- 5. 画布标注 ----------
        print("\n===== 5. 画布标注（框 / 关键点 / 阶段）=====")
        existing_spatial = c.get(f"/api/spatial/{clean_ds['id']}?episode_index=0").json()
        for old_item in existing_spatial:            # 重复跑演练时先清掉上次的，保持数字干净
            c.delete(f"/api/spatial/{old_item['id']}")
        saved = []
        for payload in ({"kind": "bbox", "label": "蓝色方块",
                         "data": {"x": 0.18, "y": 0.55, "w": 0.15, "h": 0.20}, "frame_index": 12},
                        {"kind": "keypoint", "label": "插槽中心", "data": {"x": 0.70, "y": 0.62},
                         "frame_index": 12},
                        {"kind": "phase", "label": "接近", "data": {"start": 0, "end": 20}}):
            r = c.post("/api/spatial", json={"dataset_id": clean_ds["id"], "episode_index": 0,
                                             "annotator": "张工", **payload})
            saved.append(r.status_code == 200)
        bad = c.post("/api/spatial", json={"dataset_id": clean_ds["id"], "episode_index": 0,
                                          "kind": "bbox", "label": "越界",
                                          "data": {"x": 0.9, "y": 0.1, "w": 0.5, "h": 0.2}})
        summary = c.get(f"/api/spatial/summary/{clean_ds['id']}").json()
        step("三类画布标注入库 + 非法被拒", all(saved) and bad.status_code == 400,
             f"入库 {summary['total']} 条 {summary['by_kind']}，越界框被拒 400")

        # ---------- 6. 抽检与一致性 ----------
        print("\n===== 6. 标注抽检：双人复核 + Kappa =====")
        for ep, reviewer, success in ((0, "李工", True), (3, "李工", False),
                                      (1, "王工", True), (3, "王工", True)):
            c.put(f"/api/reviews/{clean_ds['id']}/{ep}",
                  json={"reviewer": reviewer, "task_text": "[仿真] 抓取蓝色方块并放入右侧插槽",
                        "success": success})
        agreement = c.get(f"/api/reviews/agreement?dataset_id={clean_ds['id']}").json()
        step("一致性指标可用（含置信区间）",
             agreement["success_pairs"] > 0 and "kappa_ci" in agreement,
             f"配对 {agreement['success_pairs']} 条，一致率 {agreement['agreement']}，"
             f"Kappa {agreement['kappa']} CI {agreement['kappa_ci']}，"
             f"可靠={agreement['reliable']}，争议 {agreement['dispute_count']} 条")

        # ---------- 7. 数据划分 ----------
        print("\n===== 7. 可复现数据划分 =====")
        a1 = c.post("/api/splits/assign", json={"dataset_id": clean_ds["id"], "salt": "sim-v1",
                                                "mode": "ordered",
                                                "ratios": {"train": 0.5, "val": 0.25,
                                                           "test": 0.25}}).json()
        step("划分完成（ordered 写原生 splits）", bool(a1.get("info_splits")),
             f"计数 {a1['counts']}，info.json splits={a1.get('info_splits')}")

        # ---------- 8. 导出并校验交付包 ----------
        print("\n===== 8. 导出 train 划分并校验包内容 =====")
        exp = c.post("/api/splits/export", json={"dataset_id": clean_ds["id"], "split": "train"}).json()
        job = wait_job(c, exp["job_id"])
        result = job.get("result") or {}
        out_dir = Path(result.get("path") or "")
        step("按划分导出成功", job["status"] == "succeeded" and out_dir.is_dir(),
             f"{result.get('episodes')} 条 / {result.get('frames')} 帧 → {result.get('zip')}")
        if out_dir.is_dir():
            import pyarrow.parquet as pq

            parquet = sorted((out_dir / "data" / "chunk-000").glob("*.parquet"))
            tasks = (out_dir / "meta" / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()
            table = pq.read_table(parquet[0]).to_pydict()
            last_success = bool(table["next.success"][-1])
            step("交付包含 parquet + 任务文本 + 成功位",
                 bool(parquet) and bool(tasks) and "next.success" in table,
                 f"{len(parquet)} 个 parquet，tasks.jsonl {len(tasks)} 行"
                 f"（{json.loads(tasks[0])['task'][:34]}…），末帧 next.success={last_success}")
            ann_file = out_dir / "meta" / "annotations.json"
            step("交付包带画布标注", ann_file.is_file(),
                 f"annotations.json counts={json.loads(ann_file.read_text(encoding='utf-8'))['counts']}"
                 if ann_file.is_file() else "缺 annotations.json")

        # ---------- 9. 训练登记 ----------
        print("\n===== 9. 训练实验登记（绑定数据集版本与划分）=====")
        reg = c.post("/api/training/register", json={
            "dataset_id": clean_ds["id"], "split": "train",
            "run_dir": r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\outputs\pusht_train_20k",
            "name": "sim-pickplace-diffusion"}).json()
        step("实验登记并绑定划分", reg.get("split") == "train" and reg.get("metrics") is not None,
             f"实验 #{reg.get('experiment_id')} 指标点 {reg.get('metrics')} "
             f"split={reg.get('split')}/{reg.get('split_salt')} episodes={reg.get('split_episodes')}")

        # ---------- 10. bad case 回灌 ----------
        print("\n===== 10. bad case 收集与回灌 =====")
        col = c.post("/api/badcases/collect", json={"dataset_id": clean_ds["id"]}).json()
        all_cases = c.get(f"/api/badcases?dataset_id={clean_ds['id']}&limit=100").json()["cases"]
        open_cases = [x for x in all_cases if x["status"] == "open"]
        returned_cases = [x for x in all_cases if x["status"] == "returned"]
        if not open_cases and returned_cases:
            # 演练可以重复跑：上一次已经回灌过，这次没有新增可回灌的（平台行为正确）
            existing_hard = next((d for d in c.get("/api/datasets").json()
                                  if d["name"] == f"{clean}_hardcases"), None)
            step("bad case 收集与回灌（本轮无新增，历史已回灌）", bool(existing_hard),
                 f"该数据集 {len(returned_cases)} 条已回灌过 → 困难集 "
                 f"#{existing_hard['id'] if existing_hard else '?'} "
                 f"（{existing_hard['total_episodes'] if existing_hard else '?'} 条）；"
                 f"本次新增可收集 {col.get('annotation', 0) + col.get('qc', 0)} 条")
        else:
            step("bad case 收集（失败标注进入队列）", len(open_cases) > 0,
                 f"新增 QC {col.get('qc')} / 标注 {col.get('annotation')}，"
                 f"该数据集未处理 {len(open_cases)} 条")
            hard = c.post("/api/badcases/return", json={"dataset_id": clean_ds["id"]}).json()
            step("回灌产出困难集（带血缘）", bool(hard.get("hard_set_dataset_id")),
                 f"困难集 #{hard.get('hard_set_dataset_id')} v{hard.get('hard_set_version')} "
                 f"{hard.get('episodes_count')} 条 episode {hard.get('episodes')}，"
                 f"来源 {hard.get('sources')}")

        # ---------- 11. 重复检测 ----------
        print("\n===== 11. 重复检测（确认仿真集内部不重复） =====")
        job = wait_job(c, c.post("/api/jobs/dedup", json={}).json()["job_id"])
        report = c.get("/api/dedup/report").json()
        clean_groups = [g for g in report["groups"]
                        if any(m["dataset_id"] == clean_ds["id"] for m in g["members"])]
        step("仿真集 4 条彼此不同（未误报重复）", not clean_groups,
             f"指纹 {report['hashed_episodes']} 条，涉及仿真集的重复组 {len(clean_groups)} 个")

        # ---------- 12. 存储清理计划 ----------
        print("\n===== 12. 存储清理计划 =====")
        plan = c.get("/api/storage/plan").json()
        step("清理计划可用", "reclaimable_bytes" in plan,
             f"可回收 {plan['reclaimable_bytes'] / 1024 / 1024:.1f}MB / "
             f"{plan['reclaimable_files']} 文件，受保护 {len(plan['kept'])} 项")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n===== 演练结果：{passed}/{len(RESULTS)} 步通过 =====")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"   ❌ {name} {detail}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
