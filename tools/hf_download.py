#!/usr/bin/env python3
"""从 HuggingFace 下载数据集到指定目录（支持断点续传、体积核对）。

huggingface.co 与 hf-mirror.com 都试，用先能通的那个。

    python tools/hf_download.py --repo continuallearning/libero_40_image_task_0_8 \
        --dest "C:\\Users\\ll\\Desktop\\libero_40_image_task_0_8"

    python tools/hf_download.py --repo <repo> --dest <dir> --dry-run   # 只看清单和体积
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

BASES = ["https://huggingface.co", "https://hf-mirror.com"]
TIMEOUT = httpx.Timeout(60.0, read=900.0, write=60.0)


def pick_base(client: httpx.Client) -> str:
    for base in BASES:
        try:
            resp = client.get(f"{base}/api/datasets", timeout=20)
            if resp.status_code < 500:
                print(f"[base] 使用 {base}")
                return base
        except Exception as exc:  # noqa: BLE001
            print(f"[base] {base} 不可达（{type(exc).__name__}）")
    raise SystemExit("两个源都连不上（huggingface.co 与 hf-mirror.com）")


def list_files(client: httpx.Client, base: str, repo: str) -> list[dict]:
    resp = client.get(f"{base}/api/datasets/{repo}/tree/main?recursive=true", timeout=60)
    resp.raise_for_status()
    return [item for item in resp.json() if item.get("type") == "file"]


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def download(client: httpx.Client, base: str, repo: str, rel: str, dest: Path,
             expected: int) -> tuple[bool, int, float]:
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and expected and target.stat().st_size == expected:
        return True, expected, 0.0          # 已下好，跳过（断点续传的关键）
    part = target.with_suffix(target.suffix + ".part")
    url = f"{base}/datasets/{repo}/resolve/main/{rel}"
    started = time.time()
    for attempt in range(1, 4):
        try:
            with client.stream("GET", url, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in resp.iter_bytes(1024 * 512):
                        fh.write(chunk)
            got = part.stat().st_size
            if expected and got != expected:
                raise RuntimeError(f"体积不符：期望 {expected} 实际 {got}")
            part.replace(target)
            return True, got, time.time() - started
        except Exception as exc:  # noqa: BLE001
            print(f"      第 {attempt} 次失败：{type(exc).__name__}: {str(exc)[:90]}")
            if attempt == 3:
                return False, 0, time.time() - started
            time.sleep(2 * attempt)
    return False, 0, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 HF 数据集（可续传）")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-gb", type=float, default=20.0)
    args = parser.parse_args()

    dest = Path(args.dest)
    with httpx.Client(follow_redirects=True, headers={"User-Agent": "hkc-platform/1.0"}) as client:
        base = pick_base(client)
        files = list_files(client, base, args.repo)
        total = sum(f.get("size") or 0 for f in files)
        print(f"[list] {len(files)} 个文件，合计 {human(total)}")
        for item in sorted(files, key=lambda x: -(x.get("size") or 0))[:5]:
            print(f"   最大: {human(item.get('size') or 0):>9}  {item['path']}")
        if total / 1024 ** 3 > args.max_gb:
            print(f"[stop] 超过 --max-gb {args.max_gb}GB")
            return 1
        if args.dry_run:
            for item in files:
                print(f"   {human(item.get('size') or 0):>9}  {item['path']}")
            return 0

        print(f"[dest] {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        ok = 0
        failed: list[str] = []
        done_bytes = 0
        started = time.time()
        for index, item in enumerate(sorted(files, key=lambda x: x["path"]), start=1):
            rel, size = item["path"], item.get("size") or 0
            success, got, seconds = download(client, base, args.repo, rel, dest, size)
            if success:
                ok += 1
                done_bytes += got
                speed = got / seconds / 1024 / 1024 if seconds > 0 else 0
                print(f"[{index}/{len(files)}] {rel}  {human(got)}  {seconds:.1f}s"
                      f"{f'  {speed:.1f}MB/s' if speed else '  (已存在跳过)'}"
                      f"  累计 {human(done_bytes)}/{human(total)}")
            else:
                failed.append(rel)
                print(f"[{index}/{len(files)}] ✗ {rel} 下载失败")

        elapsed = time.time() - started
        print(f"\n[结果] 成功 {ok}/{len(files)}，耗时 {elapsed / 60:.1f} 分钟，"
              f"平均 {(done_bytes / 1024 / 1024) / max(elapsed, 1):.1f}MB/s")
        actual = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print(f"[核对] 落盘 {human(actual)}（清单 {human(total)}）")
        if failed:
            print(f"[失败] {len(failed)} 个：{failed[:5]}")
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
