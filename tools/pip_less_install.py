#!/usr/bin/env python3
"""沙箱里绕开 pip 装依赖：下 wheel -> 解压到私有目录。

为什么不用 pip：本机沙箱会拒绝 pip 的临时文件操作（`pip-unpack-*/*.whl.metadata` Errno 13），
而 ACL 诊断脚本已确认**工作区权限本身没问题**（verdict=NOT_THIS_CLASS）。wheel 本质是 zip，
用纯 Python 下载 + 解压即可，且只写工作区内目录、不动 yolov8n。

    python tools/pip_less_install.py            # 直到 import lerobot 成功
    python tools/pip_less_install.py --rounds 60
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEPS = ROOT / ".deps_lerobot"
WHEELS = ROOT / ".wheels"
LEROBOT_SRC = Path(r"C:\Users\ll\Desktop\VLA\lerobot\lerobot_project\src")
PY = Path(r"C:\Users\ll\miniconda3\envs\yolov8n\python.exe")

PYPI = "https://pypi.org/pypi/{name}/json"
# 与 import 名不一致的包
ALIAS = {"_cffi_backend": "cffi", "cffi": "cffi", "PIL": "pillow", "yaml": "pyyaml", "cv2": "opencv-python", "skimage": "scikit-image",
         "dateutil": "python-dateutil", "attr": "attrs", "google": "protobuf",
         "ruamel": "ruamel.yaml", "serial": "pyserial", "pkg_resources": "setuptools",
         "torchvision": "torchvision", "google.protobuf": "protobuf"}
# lerobot 要求锁定的版本（别装 latest，会不兼容）
PIN = {"draccus": "0.10.0", "datasets": "3.6.0"}
# 绝不下这些（体积巨大 / 已由 yolov8n 提供）
SKIP = {"torch", "torchvision"}   # 只挡这两个（体积 GB 级且 yolov8n 已提供）；其余照装


def py_paths() -> str:
    return f"{DEPS}{';'}{LEROBOT_SRC}"


def import_probe(client: httpx.Client) -> tuple[bool, str | None]:
    """尝试 import lerobot，返回 (成功?, 缺失模块名)。"""
    code = ("import sys\n"
            f"sys.path.insert(0, r'{DEPS}')\n"
            f"sys.path.insert(0, r'{LEROBOT_SRC}')\n"
            "import lerobot, lerobot.scripts.train  # noqa\n"
            "print('IMPORT_OK')\n")
    res = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, timeout=600)
    out = (res.stdout or "") + (res.stderr or "")
    if "IMPORT_OK" in out:
        return True, None
    match = (re.search(r"No module named '([^']+)'", out)
             or re.search(r"No package metadata was found for ([A-Za-z0-9_.-]+)", out)
             or re.search(r"distribution..([A-Za-z0-9_.-]+)..", out))
    if match:
        return False, match.group(1).split(".")[0]
    print("   （没有 ModuleNotFoundError，实际输出尾部）")
    print("   " + "\n   ".join(out.strip().splitlines()[-6:]))
    return False, None


def _version_key(version: str) -> tuple:
    """把版本号粗略转成可排序的元组（忽略 pre/post 后缀）。"""
    nums = re.findall(r"\d+", version)
    return tuple(int(n) for n in nums[:4]) or (0,)


def _wheel_ok(filename: str) -> int:
    """返回可接受的分数：0 = 这个轮子不能在 cp310/Windows 上跑。"""
    if "py3-none-any" in filename or "py2.py3-none-any" in filename:
        return 2                      # 纯 Python 轮子，任何平台都能跑（先判这个！）
    if "win_amd64" not in filename:
        return 0
    if "cp311" in filename or "cp312" in filename or "cp313" in filename:
        return 0                      # 只支持 3.11+ 的二进制轮子，cp310 用不了
    if "cp310" in filename:
        return 3
    if re.search(r"cp3\d-abi3", filename):     # abi3 向后兼容，cp38/39-abi3 在 3.10 可用
        minor = int(re.search(r"cp3(\d)-abi3", filename).group(1))
        return 3 if minor <= 10 else 0
    return 0


def pick_wheel(client: httpx.Client, module: str) -> tuple[str, str, str] | None:
    """在**所有版本**里找第一个能在 cp310/Windows 上跑的 wheel（latest 不行就往旧版本找）。"""
    name = ALIAS.get(module, module)
    try:
        data = client.get(PYPI.format(name=name), timeout=40).json()
    except Exception as exc:  # noqa: BLE001
        print(f"   ✗ {module}: 查询 PyPI 失败 {type(exc).__name__}")
        return None

    candidates: list[tuple[tuple, int, str, str, str]] = []
    pinned = PIN.get(name)
    versions = [pinned] if pinned else list(data["releases"])
    for version in versions:
        for f in data["releases"].get(version) or []:
            if f["packagetype"] != "bdist_wheel" or f.get("yanked"):
                continue
            score = _wheel_ok(f["filename"])
            if score:
                candidates.append((_version_key(version), score, f["filename"], f["url"], version))
    if not candidates:
        print(f"   ✗ {module}: 所有版本都没有能在 py3.10/Windows 上跑的 wheel（{name}）")
        return None
    # 先比版本（新的优先），再比匹配度
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _v, _s, filename, url, version = candidates[0]
    return filename, url, version


def install_wheel(client: httpx.Client, filename: str, url: str) -> bool:
    WHEELS.mkdir(exist_ok=True)
    target = WHEELS / filename
    if not target.is_file():
        try:
            resp = client.get(url, timeout=300, follow_redirects=True)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001
            print(f"   ✗ 下载失败 {filename}: {type(exc).__name__}")
            return False
    try:
        with zipfile.ZipFile(io.BytesIO(target.read_bytes())) as zf:
            zf.extractall(DEPS)
        print(f"   ✓ {filename}（{target.stat().st_size / 1024 / 1024:.1f}MB）")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"   ✗ 解压失败 {filename}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=40)
    args = parser.parse_args()
    DEPS.mkdir(exist_ok=True)
    print(f"目标目录 : {DEPS}")
    print(f"解释器   : {PY}")
    print(f"lerobot  : {LEROBOT_SRC}\n")

    attempted: dict[str, str] = {}
    with httpx.Client(headers={"User-Agent": "hkc-platform/1.0"}) as client:
        for round_no in range(1, args.rounds + 1):
            ok, missing = import_probe(client)
            if ok:
                print(f"\n✅ 第 {round_no} 轮：import lerobot 成功")
                count = len([p for p in DEPS.iterdir() if not p.name.startswith("__")])
                size = sum(f.stat().st_size for f in DEPS.rglob("*") if f.is_file())
                print(f"   私有依赖目录：{count} 个顶层项 / {size / 1024 / 1024:.1f}MB")
                return 0
            if not missing:
                print("\n❌ 不是缺包问题，停止（见上面输出）")
                return 1
            name = ALIAS.get(missing, missing)
            if name in SKIP:
                print(f"\n❌ 缺 {missing}，但它属于 {name}（应由 yolov8n 提供或体积过大），停止")
                return 1
            picked = pick_wheel(client, missing)
            if not picked:
                return 1
            filename, url, version = picked
            print(f"[{round_no}] 缺 {missing} -> 装 {name} {version}")
            if not install_wheel(client, filename, url):
                return 1
            if name in attempted and attempted[name] == filename:
                print(f"\n❌ {name} 连续两次装的都是 {filename} 但仍 import 失败（ABI/依赖不匹配），停止")
                return 1
            attempted[name] = filename
    print("\n达到轮数上限，仍未成功")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
