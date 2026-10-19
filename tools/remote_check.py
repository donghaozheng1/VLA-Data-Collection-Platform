"""远程训练链路检查：连接 → 环境探测 → 数据同步 → 触发命令 → 回收结果。

    python tools/remote_check.py                 # 只做连接 + 环境探测
    python tools/remote_check.py --sync          # 额外同步 zhiyan 数据集
    python tools/remote_check.py --smoke         # 额外跑一次无害的远程命令（nvidia-smi + 写文件）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import remote  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="同步 zhiyan 数据集到服务器")
    ap.add_argument("--smoke", action="store_true", help="跑一次无害远程命令验证链路")
    ap.add_argument("--host")
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--remote-root")
    args = ap.parse_args()

    cfg = remote.load_config()
    for key, val in (("host", args.host), ("user", args.user), ("password", args.password),
                     ("remote_root", args.remote_root)):
        if val:
            cfg[key] = val
    if args.user or args.password or args.host:
        p = remote.save_config(cfg)
        print(f"已更新配置: {p}")

    print("=== 配置（密码不显示）===")
    for k, v in cfg.items():
        if k in ("password", "key_file"):
            print(f"   {k:20} {'<已设置>' if v else '<未设置>'}")
        else:
            print(f"   {k:20} {v}")

    print("\n=== 1. 连接测试 ===")
    try:
        with remote.make_remote(cfg) as r:
            code, out, err = r.run("echo CONNECTED && hostname && whoami", timeout=20)
            print(f"   退出码 {code}: {out.strip() or err.strip()[:200]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   ✗ {exc}")
        print("\n   排查建议：")
        print("     · 确认 remote.json 里 user/password 或 key_file 正确")
        print("     · 服务器 22 端口已验证开放 ✓")
        print("     · 若服务器只允许密钥：把私钥路径填到 key_file")
        return 1

    print("\n=== 2. 环境探测 ===")
    info = remote.probe(cfg)
    for k, v in info.items():
        print(f"   [{k}]")
        for line in str(v).splitlines()[:4]:
            print(f"      {line}")

    if args.sync:
        print("\n=== 3. 同步 zhiyan 数据集 ===")
        local = ROOT / "storage" / "datasets" / "zhiyan_pick_place"
        print(f"   本地: {local}")
        res = remote.sync_dataset(local, "zhiyan_pick_place", cfg)
        remote_dir, n, total = res.split("|")
        print(f"   远程: {remote_dir}")
        print(f"   上传 {n} 个文件；服务器上实际 {total} 个文件")

    if args.smoke:
        print("\n=== 4. 链路冒烟（写文件 + 读回 + 删除）===")
        with remote.make_remote(cfg) as r:
            probe_file = f"{cfg['remote_root']}/.platform_smoke"
            r.run(f"mkdir -p {cfg['remote_root']}")
            code, out, err = r.run(f"echo 'hello from {cfg['host']} at $(date)' > {probe_file} && cat {probe_file}")
            print(f"   写入/读回: {(out or err).strip()}")
            r.run(f"rm -f {probe_file}")
            code, out, _ = r.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
            print(f"   GPU: {out.strip() or '<无 nvidia-smi>'}")
            local_get = ROOT / "storage" / "remote_smoke.txt"
            r.upload_dir(ROOT / "docs", f"{cfg['remote_root']}/_docs_probe")
            r.run(f"du -sh {cfg['remote_root']}/_docs_probe")
            n = r.download_dir(f"{cfg['remote_root']}/_docs_probe", ROOT / "storage" / "_docs_back")
            print(f"   上传 docs/ 后回读 {n} 个文件到 storage/_docs_back（往返验证）")
            r.run(f"rm -rf {cfg['remote_root']}/_docs_probe")
    print("\n完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
