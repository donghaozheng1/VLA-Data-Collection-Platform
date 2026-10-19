#!/usr/bin/env python3
"""从 HuggingFace 拉一个**小切片**的 LeRobot 数据集（只下几条 episode）到本地。

用途：只想验证链路能不能跑通时，不必下整个数据集。默认经 **hf-mirror.com**（本机 huggingface.co 不可达）。

    # 取 lerobot/aloha_sim_insertion_human 的头 2 条，落到 storage/datasets/aloha-insertion-2ep
    python tools/fetch_small_dataset.py --repo lerobot/aloha_sim_insertion_human --episodes 2

    # 换仓库 / 换条数 / 换本地名字
    python tools/fetch_small_dataset.py --repo lerobot/aloha_sim_transfer_cube_human \
        --episodes 3 --name aloha-cube-3ep

下载内容：meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl} + 选中 episode 的
parquet 与视频，然后把 meta 里的总数改成切片后的真实值（否则平台会以为有 50 条）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import db  # noqa: E402

MIRROR = "https://hf-mirror.com"


def fetch(url: str, client: httpx.Client) -> bytes:
    resp = client.get(url, timeout=120, follow_redirects=True)
    if resp.status_code != 200:
        raise RuntimeError(f"{url} -> HTTP {resp.status_code}")
    return resp.content


def main() -> int:
    parser = argparse.ArgumentParser(description="拉一个小切片 LeRobot 数据集")
    parser.add_argument("--repo", default="lerobot/aloha_sim_insertion_human")
    parser.add_argument("--episodes", type=int, default=2, help="要几条（默认 2）")
    parser.add_argument("--name", default=None, help="本地数据集名")
    parser.add_argument("--mirror", default=MIRROR)
    parser.add_argument("--max-mb", type=float, default=60, help="总下载上限（保护用）")
    args = parser.parse_args()

    name = args.name or f"{args.repo.split('/')[-1]}-{args.episodes}ep"
    out_dir = db.STORAGE / "datasets" / name
    base = f"{args.mirror}/datasets/{args.repo}/resolve/main"

    with httpx.Client(headers={"User-Agent": "hkc-platform/1.0"}) as client:
        info = json.loads(fetch(f"{base}/meta/info.json", client))
        fps = info.get("fps")
        video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
        print(f"[fetch] {args.repo}: {info.get('total_episodes')} 条 / {fps}fps / 视频 {video_keys}")

        episodes_lines = fetch(f"{base}/meta/episodes.jsonl", client).decode("utf-8").strip().splitlines()
        all_episodes = [json.loads(line) for line in episodes_lines if line.strip()]
        chosen = all_episodes[:args.episodes]
        print(f"[fetch] 只取前 {len(chosen)} 条："
              f"{[(e['episode_index'], e['length']) for e in chosen]}")

        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "meta").mkdir(parents=True)
        (out_dir / "data" / "chunk-000").mkdir(parents=True)

        downloaded = 0
        total_mb = 0.0
        for episode in chosen:
            index = episode["episode_index"]
            rel_data = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
            rel_data = rel_data.format(episode_chunk=index // int(info.get("chunks_size", 1000)),
                                       episode_index=index)
            blob = fetch(f"{base}/{rel_data}", client)
            target = out_dir / rel_data
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            downloaded += 1
            total_mb += len(blob) / 1024 / 1024
            for key in video_keys:
                rel_video = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/"
                                                  "episode_{episode_index:06d}.mp4")
                rel_video = rel_video.format(episode_chunk=index // int(info.get("chunks_size", 1000)),
                                             video_key=key, episode_index=index)
                blob = fetch(f"{base}/{rel_video}", client)
                vtarget = out_dir / rel_video
                vtarget.parent.mkdir(parents=True, exist_ok=True)
                vtarget.write_bytes(blob)
                total_mb += len(blob) / 1024 / 1024
            print(f"[fetch]   episode {index}: {episode['length']} 帧，"
                  f"累计 {total_mb:.1f}MB")
            if total_mb > args.max_mb:
                raise SystemExit(f"超过 --max-mb {args.max_mb}MB，已停止（用更小的 --episodes）")

        # meta：tasks 全留；episodes / stats 只留切片内的
        (out_dir / "meta" / "tasks.jsonl").write_bytes(fetch(f"{base}/meta/tasks.jsonl", client))
        (out_dir / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in chosen) + "\n", encoding="utf-8")
        try:
            stats_lines = fetch(f"{base}/meta/episodes_stats.jsonl", client).decode("utf-8").strip().splitlines()
            stats = [json.loads(line) for line in stats_lines if line.strip()]
            keep = {e["episode_index"] for e in chosen}
            stats = [s for s in stats if s.get("episode_index") in keep]
            (out_dir / "meta" / "episodes_stats.jsonl").write_text(
                "\n".join(json.dumps(s, ensure_ascii=False) for s in stats) + "\n", encoding="utf-8")
        except RuntimeError:
            pass  # 有的仓库没有 stats

        # 总数改成切片后的真实值
        info["total_episodes"] = len(chosen)
        info["total_frames"] = sum(int(e["length"]) for e in chosen)
        info["total_videos"] = len(chosen) * len(video_keys)
        info["total_chunks"] = 1
        info["splits"] = {"train": f"0:{len(chosen)}"}
        info["source_repo"] = args.repo
        info["note"] = f"平台工具从 {args.repo} 截取的前 {len(chosen)} 条，仅用于链路验证"
        (out_dir / "meta" / "info.json").write_text(
            json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"\n[fetch] 已落盘: {out_dir}")
    print(f"[fetch] {info['total_episodes']} 条 / {info['total_frames']} 帧 / "
          f"{info['total_videos']} 视频 / 下载 {total_mb:.1f}MB")
    print(f"[fetch] 接入平台：POST /api/jobs/ingest "
          f'{{"path": "{out_dir}", "name": "{name}", "with_video": true}}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
