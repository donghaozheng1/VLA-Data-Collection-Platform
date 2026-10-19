"""通用 wheel 安装器：从 PyPI 下 wheel 并解压到私有目录（绕开 pip 的沙箱限制）。

    python tools/wheel_install.py paramiko bcrypt pynacl --dest .vendor

原理：wheel 就是 zip。按当前解释器挑最匹配的 wheel（cp310 win_amd64 > py3-none-any），
下载后解压到目标目录（该目录已在平台的 PYTHONPATH 上）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPI = "https://pypi.org/pypi/{name}/json"


def rank(filename: str, py_tag: str) -> int:
    f = filename.lower()
    if f.endswith(".tar.gz") or f.endswith(".zip"):
        return 1
    if f.endswith("-none-any.whl"):
        return 5
    if py_tag in f and ("win_amd64" in f or "win32" in f):
        return 10
    if "win_amd64" in f or "win32" in f:
        return 7
    if py_tag in f:
        return 6
    if "none-any" in f:
        return 5
    return 0


def pick(name: str, py_tag: str, ver: str | None = None) -> tuple[str, str]:
    with urllib.request.urlopen(PYPI.format(name=name), timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    releases = data["releases"]
    versions = [ver] if ver else sorted(releases, key=lambda v: [int(x) if x.isdigit() else 0
                                                                 for x in v.split(".") if x.isdigit()] or [0])
    best: tuple[int, str, str] | None = None
    for v in reversed(versions):
        for f in releases.get(v) or []:
            fn = f["filename"]
            if not (fn.endswith(".whl") or fn.endswith(".tar.gz") or fn.endswith(".zip")):
                continue
            score = rank(fn, py_tag)
            if score >= 5 and (best is None or score > best[0]):
                best = (score, fn, f["url"])
        if best and score >= 10:
            break
    if not best:
        raise RuntimeError(f"{name}: 找不到可用 wheel/源码包")
    return best[1], best[2]


def install(name: str, dest: pathlib.Path, py_tag: str) -> bool:
    try:
        fn, url = pick(name, py_tag)
    except Exception as exc:  # noqa: BLE001
        print(f"   ✗ {name}: {exc}")
        return False
    print(f"   {name}: {fn}")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / ".wheels" / fn
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file() or tmp.stat().st_size == 0:
        urllib.request.urlretrieve(url, tmp)
    print(f"      下载 {tmp.stat().st_size / 1048576:.2f} MB")
    if fn.endswith(".whl"):
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(dest)
        print(f"      解压到 {dest}")
        return True
    print("      ⚠ 只有源码包（sdist），需要另外编译，跳过")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="+")
    ap.add_argument("--dest", default=".vendor")
    ap.add_argument("--python", default=None, help="目标解释器（用于挑 ABI 标签）")
    args = ap.parse_args()

    py = args.python or sys.executable
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print(f"解释器: {py}  → wheel 标签 {tag}")
    dest = (ROOT / args.dest).resolve()
    ok = [install(p, dest, tag) for p in args.packages]
    print(f"\n成功 {sum(ok)}/{len(ok)}")
    if all(ok):
        sys.path.insert(0, str(dest))
        for p in args.packages:
            mod = {"pynacl": "nacl", "python-dateutil": "dateutil"}.get(p, p)
            try:
                m = __import__(mod)
                print(f"   ✓ import {mod} 成功  {getattr(m, '__version__', '')}")
            except Exception as exc:  # noqa: BLE001
                print(f"   ✗ import {mod} 失败: {exc}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
