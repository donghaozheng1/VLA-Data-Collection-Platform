#!/usr/bin/env python3
"""从平台里已有的数据集切一个「几条数据」的小测试集，并接入平台。

用途：只想验证链路能不能跑通时，不必去 Hub 下新数据集（而且 Hub 上 lerobot 的数据集
现在多为 v3.0 布局，平台契约是 v2.1，见 README 的说明）。

    # 从 pusht-real 切前 2 条 -> 数据集 pusht-2ep（真实视频，v2.1）
    python tools/make_small_dataset.py --from pusht-real --episodes 2 --name pusht-2ep

切出来的仍是真的 LeRobot v2.1（parquet + 视频 + meta），标注/成功位会一起带过去。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db, jobs, lerobot  # noqa: E402
from core import pipeline  # noqa: F401,E402  导入即注册任务处理器


def main() -> int:
    parser = argparse.ArgumentParser(description="切一个几条数据的小测试集")
    parser.add_argument("--from", dest="source", default="pusht-real", help="源数据集名")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--mode", default="head", choices=["head", "random"],
                        help="取前 N 条 / 可复现随机取 N 条")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    db.init_db()
    row = db.query_one("SELECT * FROM datasets WHERE name=?", (args.source,))
    if not row:
        print(f"源数据集不存在：{args.source}")
        return 1
    name = args.name or f"{args.source}-{args.episodes}ep"
    meta = lerobot.load_meta(row["path"])
    indices = sorted(meta["episodes"]) or list(range(int(meta["info"].get("total_episodes") or 0)))
    if args.mode == "random":
        import hashlib
        indices = sorted(indices, key=lambda i: hashlib.sha256(f"small:{i}".encode()).hexdigest())
    chosen = sorted(indices[:args.episodes])
    print(f"[slice] 源 #{row['id']} {row['name']}（{len(indices)} 条）-> 取 {chosen}")

    out_dir = db.EXPORT_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    annotations = {a["episode_index"]: a for a in
                   db.query("SELECT * FROM annotations WHERE dataset_id=?", (row["id"],))}
    result = lerobot.export_subset(row["path"], meta, chosen, out_dir, name,
                                  annotations=annotations)
    print(f"[slice] 写出 {result['episodes']} 条 / {result['frames']} 帧 -> {out_dir}")

    job_id = jobs.submit("ingest", {"path": str(out_dir), "name": name,
                                    "with_video": bool(lerobot.video_keys(meta["info"])),
                                    "origin": {"kind": "slice", "derived_by": "slice:small-test",
                                               "parent_dataset_id": row["id"],
                                               "parent_version": row["version"],
                                               "params": {"episodes": len(chosen),
                                                          "mode": args.mode}}},
                         target_type="dataset", target_id=row["id"], wait=True)
    job = jobs.get(job_id)
    new = db.query_one("SELECT * FROM datasets WHERE name=?", (name,))
    print(f"[slice] 接入: job{job_id} {job['status']} -> 数据集 "
          f"#{new['id'] if new else '?'} v{new['version'] if new else '?'} "
          f"{new['total_episodes'] if new else '?'} 条 / "
          f"{new['total_frames'] if new else '?'} 帧 / "
          f"{new['total_videos'] if new else '?'} 视频 / codec={new['codec'] if new else '?'}")
    print(f"[slice] 血缘: slice:small-test <- #{row['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
