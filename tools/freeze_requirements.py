"""生成本地环境的依赖清单 requirements-freeze.txt（含私有 .deps 目录 + 服务器训练环境）。"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(r"C:\Users\ll\Desktop\HKC_Data_Platform")
OUT = ROOT / "requirements-freeze.txt"
lines: list[str] = []


def meta_of(dist_info: pathlib.Path) -> tuple[str, str, str]:
    """从 *.dist-info/METADATA 读 Name / Version / Requires-Python。"""
    meta = dist_info / "METADATA"
    name = version = req_py = ""
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines()[:200]:
            if line.startswith("Name: "):
                name = line[6:].strip()
            elif line.startswith("Version: "):
                version = line[9:].strip()
            elif line.startswith("Requires-Python: "):
                req_py = line[17:].strip()
            if name and version:
                break
    return name or dist_info.name.replace(".dist-info", ""), version or "?", req_py


def scan_site_dir(d: pathlib.Path, title: str, note: str) -> None:
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"# {title}")
    lines.append(f"# 路径: {d}")
    lines.append(f"# {note}")
    lines.append("=" * 78)
    if not d.is_dir():
        lines.append("# （目录不存在）")
        return
    pkgs = sorted([p for p in d.glob("*.dist-info")], key=lambda p: p.name.lower())
    rows = []
    for p in pkgs:
        name, version, req_py = meta_of(p)
        rows.append((name, version, req_py))
    lines.append(f"# 共 {len(rows)} 个包")
    lines.append("")
    for name, version, req_py in rows:
        line = f"{name}=={version}"
        if req_py:
            line += f"   ; python_version ok={req_py}"
        lines.append(line)


def section(title: str, body: str) -> None:
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"# {title}")
    lines.append("=" * 78)
    lines.append(body)


lines.append("# HKC_Data_Platform —— 环境依赖冻结清单")
lines.append("# 生成方式：扫描各私有依赖目录的 *.dist-info/METADATA + pip freeze + 服务器环境")
lines.append("# 用途：环境可重建；备份时不需要备份 .deps_* / .vendor 这类解压出来的库")
lines.append("#")
lines.append("# 重建方式（本机）：")
lines.append("#   1) 平台运行时：用 base python，把 .vendor 放进 PYTHONPATH（见 start.cmd）")
lines.append("#   2) 训练依赖：python tools/wheel_install.py  <包名…>   （或直接 pip install -r 本文件对应段）")
lines.append("#   3) 仿真依赖：同 .deps_libero 段")

# ---- 1) base python 的完整 freeze（平台运行时实际使用的解释器）----
try:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=180)
    section("1. base python（平台运行时解释器）pip freeze", out.stdout.strip() or "# （空）")
    section("1b. base python 版本", subprocess.run([sys.executable, "-V"], capture_output=True, text=True,
                                                   encoding="utf-8", errors="replace").stdout.strip())
except Exception as exc:  # noqa: BLE001
    section("1. base python pip freeze", f"# 生成失败: {exc}")

# ---- 2) .vendor（平台运行时的额外依赖）----
scan_site_dir(ROOT / ".vendor", "2. .vendor（平台运行时额外依赖，PYTHONPATH 注入）",
              "平台本体只需要 FastAPI/Jinja2/numpy/pyarrow 等，不含 torch")

# ---- 3) .deps_lerobot（lerobot 训练链路）----
scan_site_dir(ROOT / ".deps_lerobot", "3. .deps_lerobot（lerobot 训练/推理依赖，约 1.1GB）",
              "注意：torch/torchvision 由 envs/yolov8n 提供，这里不包含")

# ---- 4) .deps_libero（MuJoCo 仿真）----
scan_site_dir(ROOT / ".deps_libero", "4. .deps_libero（MuJoCo + robosuite 仿真依赖，约 600MB）",
              "MUJOCO_GL：Windows 用 wgl，Linux 用 egl")

# ---- 5) .wheels 缓存清单 ----
wheels_dir = ROOT / ".wheels"
lines.append("")
lines.append("=" * 78)
lines.append("# 5. .wheels 离线 wheel 缓存（84MB，重建环境的关键资产，建议备份）")
lines.append("=" * 78)
if wheels_dir.is_dir():
    wf = sorted(wheels_dir.glob("*"))
    lines.append(f"# 共 {len(wf)} 个文件")
    lines.append("")
    for f in wf:
        lines.append(f"# {f.name}   ({f.stat().st_size / 1048576:.2f} MB)")
else:
    lines.append("# （目录不存在）")

# ---- 6) 服务器训练环境（我们改过的：补了 transformers/tokenizers/safetensors）----
lines.append("")
lines.append("=" * 78)
lines.append("# 6. 服务器训练环境 /home/z123/miniconda3/envs/lerobot（关键子集）")
lines.append("#    注意：我们为 X-VLA 补装了 transformers / tokenizers / safetensors")
lines.append("=" * 78)
try:
    sys.path.insert(0, str(ROOT))
    from core import remote  # noqa: E402
    cfg = remote.load_config()
    r = remote.make_remote(cfg)
    code = ("import importlib;"
            "mods=['torch','torchvision','transformers','tokenizers','safetensors','accelerate','lerobot',"
            "'einops','av','diffusers','numpy','pandas','pyarrow','datasets','huggingface_hub'];"
            "print('\\n'.join(m+'=='+str(getattr(importlib.import_module(m),'__version__','?')) for m in mods "
            "if importlib.util.find_spec(m)))")
    out = r.run(f"/home/z123/miniconda3/envs/lerobot/bin/python -c \"{code}\" 2>&1 | tail -20", timeout=120)
    lines.append((out[1] or out[2]).strip() or "# （读取失败）")
    lines.append("")
    lines.append("# 服务器 GPU / 驱动（环境可复现所需）:")
    gpu = r.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; "
                "nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1", timeout=60)
    for l in (gpu[1] or "").strip().splitlines():
        lines.append(f"#   {l}")
    lines.append("#   OS: Ubuntu 22.04.5 LTS")
    lines.append("#   lerobot 0.4.4 需要 transformers>=4.57.1,<5.0.0（我们装的是 4.57.1 + tokenizers 0.22.1）")
except Exception as exc:  # noqa: BLE001
    lines.append(f"# 读取服务器环境失败: {exc}")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"✓ 已生成 {OUT}")
print(f"  {len(lines)} 行  {OUT.stat().st_size / 1024:.1f} KB")
print("\n各段包数量:")
cur = ""
count = 0
for line in lines:
    if line.startswith("# ") and (line[2:3].isdigit()):
        if cur:
            print(f"   {cur}  → {count} 个包")
        cur, count = line[2:60], 0
    elif "==" in line and not line.startswith("#"):
        count += 1
if cur:
    print(f"   {cur}  → {count} 个包")
