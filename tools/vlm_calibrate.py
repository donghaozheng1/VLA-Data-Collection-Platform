"""VLM 质检的**校准**工具：拿人工标好的样本，量化 VLM 判得准不准。

为什么必须有这一步：VLM 说"这条不合规"听起来很有说服力，但在没有校准集的情况下，
你**无法回答**两个必然会被问到的问题 ——

  1. 它报出来的问题里，有多少是真的？（precision / 误报率）
  2. 真的坏样本里，它漏了多少？（recall / 漏报率）

没有这两个数，VLM 质检的结论就只能"参考"，不能用于打回重采。
这个脚本就是用来把这两个数算出来，并给出**多条判定口径**供你选一个上线。

两种用法：

    # ① 先给数据集生成一份空的标注模板（人工填 ok: true/false）
    python tools/vlm_calibrate.py --dataset 1 --dump-template labels.json

    # ② 拿填好的标签跑校准，输出混淆矩阵与多条口径的 P/R/F1
    python tools/vlm_calibrate.py --dataset 1 --labels labels.json

标签文件格式（两种都支持）：

    {"dataset_id": 1, "labels": [{"episode_index": 0, "ok": true}, ...]}
    {"episode_index,ok": ...}      # 或 CSV：episode_index,ok
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # 让 core.* 可导入（tools/ 在子目录下）

from core import db, lerobot, vlm_qc  # noqa: E402


# --------------------------------------------------------------------------- 标签读写


def dump_template(dataset_id: int, out_path: Path) -> dict:
    """为数据集里每条 episode 生成待标注模板（ok 留空由人填）。"""
    dataset = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not dataset:
        raise SystemExit(f"数据集不存在：id={dataset_id}")
    episodes = db.query(
        "SELECT episode_index, length AS frames, task_text, success_count FROM episodes"
        " WHERE dataset_id=? ORDER BY episode_index", (dataset_id,))
    if not episodes:
        raise SystemExit(f"数据集 #{dataset_id} 还没有 episode，请先执行接入")
    payload = {
        "dataset_id": dataset_id,
        "dataset": dataset["name"],
        "_说明": "逐条把 ok 填成 true（这条数据合格）或 false（这条不合格，例如任务没做成）。"
                 "至少标 50 条，好坏样本都要有，否则算出来的指标没有意义。",
        "labels": [{"episode_index": e["episode_index"], "ok": None,
                    "frames": e["frames"], "task": e["task_text"],
                    "success_count": e["success_count"]} for e in episodes],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(out_path), "episodes": len(episodes)}


def load_labels(path: Path) -> tuple[int | None, dict[int, bool]]:
    """读标签文件；支持 JSON 与 CSV。返回 (dataset_id, {episode_index: ok})。"""
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        labels: dict[int, bool] = {}
        for row in csv.DictReader(text.splitlines()):
            index = int(row["episode_index"])
            labels[index] = str(row["ok"]).strip().lower() in ("1", "true", "yes", "y", "是")
        return None, labels

    payload = json.loads(text)
    raw = payload.get("labels") if isinstance(payload, dict) else payload
    labels = {}
    for item in raw or []:
        ok = item.get("ok")
        if ok is None:
            continue                       # 没填的跳过，不当作"合格"
        labels[int(item["episode_index"])] = bool(ok)
    return (payload.get("dataset_id") if isinstance(payload, dict) else None), labels


# --------------------------------------------------------------------------- 指标


def confusion(y_true: list[bool], y_pred: list[bool]) -> dict:
    """y_true/y_pred 里 True 表示「不合规」。返回混淆矩阵与常用指标。"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y_true),
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "accuracy": round((tp + tn) / len(y_true), 4) if y_true else None,
        "false_alarm_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "miss_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
    }


def fmt(value) -> str:
    return "—" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


# --------------------------------------------------------------------------- 主流程


def calibrate(dataset_id: int, labels: dict[int, bool], cfg: dict, out_dir: Path) -> dict:
    dataset = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not dataset:
        raise SystemExit(f"数据集不存在：id={dataset_id}")
    meta = lerobot.load_meta(dataset["path"])
    known = {e["episode_index"] for e in
             db.query("SELECT episode_index FROM episodes WHERE dataset_id=?", (dataset_id,))}

    usable = sorted(i for i in labels if i in known)
    if len(usable) != len(labels):
        print(f"  提示：{len(labels) - len(usable)} 条标签对应的 episode 不在库里，已跳过")
    if not usable:
        raise SystemExit("没有任何可用标签（episode_index 对不上？）")

    print(f"数据集 #{dataset_id} {dataset['name']}：对 {len(usable)} 条已标注 episode 跑 VLM……")
    per_episode: list[dict] = []
    rules_seen: list[str] = []
    for done, index in enumerate(usable, start=1):
        result = vlm_qc.run_episode_safe(dataset["path"], meta, index, cfg, dataset_id)
        findings = result.get("findings") or []
        flags = [f["rule"] for f in findings if f["severity"] in ("error", "warning")]
        for rule in flags:
            if rule not in rules_seen:
                rules_seen.append(rule)
        per_episode.append({
            "episode_index": index,
            "human_ok": labels[index],
            "vlm_flagged": bool(flags),
            "flags": flags,
            "warnings": sum(1 for f in findings if f["severity"] == "warning"),
            "verdict": result.get("verdict"),
            "cached": result.get("cached"),
            "latency_ms": result.get("latency_ms"),
            "error": result.get("error"),
        })
        flag_text = "，".join(flags) if flags else "无"
        print(f"  [{done}/{len(usable)}] ep{index} 人工={'合格' if labels[index] else '不合格'}"
              f"｜VLM={'可疑' if flags else '通过'}（{flag_text}）")

    y_true = [not row["human_ok"] for row in per_episode]      # True = 不合规
    y_pred = [row["vlm_flagged"] for row in per_episode]

    overall = confusion(y_true, y_pred)

    # 按规则逐条看：哪条问题在误报、哪条在漏报
    per_rule: dict[str, dict] = {}
    for rule in rules_seen:
        pred = [rule in row["flags"] for row in per_episode]
        per_rule[rule] = confusion(y_true, pred)

    # 口径扫描：要求"至少 N 条 warning 才算不合规"，看 P/R 怎么变
    # （VLM 没有概率输出，能调的只有这个严格度）
    sweep = []
    max_warn = max((row["warnings"] for row in per_episode), default=0)
    for threshold in range(1, max(2, max_warn + 1)):
        pred = [row["warnings"] >= threshold for row in per_episode]
        entry = confusion(y_true, pred)
        entry["threshold"] = threshold
        sweep.append(entry)

    positives = sum(1 for row in per_episode if not row["human_ok"])
    report = {
        "dataset_id": dataset_id,
        "dataset": dataset["name"],
        "model": cfg.get("model"),
        "mode": cfg.get("mode"),
        "base_url": cfg.get("base_url"),
        "frames": cfg.get("frames"),
        "labelled": len(per_episode),
        "human_bad": positives,
        "human_good": len(per_episode) - positives,
        "cache_hits": sum(1 for row in per_episode if row["cached"]),
        "total_latency_ms": sum(row["latency_ms"] or 0 for row in per_episode),
        "errors": sum(1 for row in per_episode if row["error"]),
        "overall": overall,
        "per_rule": per_rule,
        "threshold_sweep": sweep,
        "per_episode": per_episode,
        "generated_at": db.now(),
    }

    stamp = db.now().replace("-", "").replace(":", "").replace(" ", "_")
    out_path = out_dir / f"vlm_calib_{dataset['name']}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)

    # ---- 打印给人看的结论 ----
    print("\n" + "=" * 68)
    print(f"校准结果：{dataset['name']}｜{cfg.get('model')}｜mode={cfg.get('mode')}")
    print(f"样本 {len(per_episode)} 条（人工判不合格 {positives} 条 / 合格 "
          f"{len(per_episode) - positives} 条）")
    print("-" * 68)
    print(f"  混淆矩阵   TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}  TN={overall['tn']}")
    print(f"  查准率 P   {fmt(overall['precision'])}   （VLM 报出来的里有多少是真的）")
    print(f"  查全率 R   {fmt(overall['recall'])}   （真的坏样本里抓到了多少）")
    print(f"  F1         {fmt(overall['f1'])}")
    print(f"  误报率     {fmt(overall['false_alarm_rate'])}   漏报率 {fmt(overall['miss_rate'])}")
    if len(per_episode) < 30:
        print("  ⚠ 样本量 < 30，指标只能作参考，不能当验收依据")

    if per_rule:
        print("\n  逐条规则的贡献：")
        print(f"    {'规则':<22}{'TP':>4}{'FP':>4}{'FN':>4}  {'P':>6}  {'R':>6}")
        for rule, item in sorted(per_rule.items(), key=lambda kv: -kv[1]["tp"]):
            print(f"    {rule:<22}{item['tp']:>4}{item['fp']:>4}{item['fn']:>4}"
                  f"  {fmt(item['precision']):>6}  {fmt(item['recall']):>6}")

    if len(sweep) > 1:
        print("\n  判定口径扫描（至少 N 条警告才算不合格）：")
        print(f"    {'N':>3}  {'P':>6}  {'R':>6}  {'F1':>6}  {'FP':>4}  {'FN':>4}")
        for item in sweep:
            print(f"    {item['threshold']:>3}  {fmt(item['precision']):>6}  "
                  f"{fmt(item['recall']):>6}  {fmt(item['f1']):>6}  {item['fp']:>4}  {item['fn']:>4}")
        print("  → 想要少误报就调大 N；想要少漏报就保持 N=1")

    print(f"\n  报告已写入：{out_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM 质检校准（精度/召回/混淆矩阵）")
    parser.add_argument("--dataset", type=int, help="数据集 id")
    parser.add_argument("--labels", type=str, help="人工标签文件（json 或 csv）")
    parser.add_argument("--dump-template", type=str, help="生成待标注模板后退出")
    parser.add_argument("--mode", default=None, choices=["checklist", "describe", "both"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存，强制重新问一遍 VLM")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.dump_template:
        if not args.dataset:
            parser.error("--dump-template 需要配合 --dataset")
        info = dump_template(args.dataset, Path(args.dump_template))
        print(f"已生成标注模板：{info['path']}（{info['episodes']} 条 episode）")
        print("请把每条 ok 填成 true/false（至少 50 条，好坏都要有），再回来跑校准。")
        return 0

    if not args.labels:
        parser.error("需要 --labels（或先用 --dump-template 生成模板）")

    file_dataset_id, labels = load_labels(Path(args.labels))
    dataset_id = args.dataset or file_dataset_id
    if dataset_id is None:
        parser.error("标签文件里没有 dataset_id，请显式传 --dataset")
    if not labels:
        parser.error("标签文件里没有填过 ok 的条目")

    db.init_db()
    cfg = vlm_qc.load_config({"mode": args.mode, "model": args.model,
                             "base_url": args.base_url, "frames": args.frames,
                             "use_cache": not args.no_cache})
    health = vlm_qc.health(cfg)
    print(f"VLM 后端 {cfg.get('base_url')}：{'可达' if health['ok'] else '不可达'} — {health['detail']}")
    if not health["ok"]:
        print("后端不可达就先别校准了：跑出来的每条都会是「服务不可用」，指标没有意义。")
        return 2
    calibrate(dataset_id, labels, cfg, Path(args.out_dir) if args.out_dir else db.REPORT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
