"""修 requirements-freeze 的服务器段 + 建 .gitignore + git init/commit。"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(r"C:\Users\ll\Desktop\HKC_Data_Platform")
FREEZE = ROOT / "requirements-freeze.txt"

# ---------- 1) 重新生成第 6 段（修 importlib.util）----------
sys.path.insert(0, str(ROOT))
from core import remote  # noqa: E402

cfg = remote.load_config()
r = remote.make_remote(cfg)
code = (
    "import importlib, importlib.util;"
    "mods=['torch','torchvision','transformers','tokenizers','safetensors','accelerate','lerobot',"
    "'einops','av','diffusers','numpy','pandas','pyarrow','datasets','huggingface_hub','peft'];"
    "print(chr(10).join(m+'=='+str(getattr(importlib.import_module(m),'__version__','?')) "
    "for m in mods if importlib.util.find_spec(m)))"
)
out = r.run(f"/home/z123/miniconda3/envs/lerobot/bin/python -c \"{code}\" 2>&1 | tail -20", timeout=120)
server_pkgs = (out[1] or out[2]).strip()

text = FREEZE.read_text(encoding="utf-8")
marker = "# 6. 服务器训练环境"
idx = text.find(marker)
head = text[:idx] if idx > 0 else text
tail_lines = [
    "# 6. 服务器训练环境 /home/z123/miniconda3/envs/lerobot（关键子集）",
    "#    注意：我们为 X-VLA 补装了 transformers / tokenizers / safetensors",
    "=" * 78,
    "# Python: 3.10.18   OS: Ubuntu 22.04.5 LTS",
    "# GPU: NVIDIA GeForce RTX 4090 D 24GB / driver 570.211.01 / CUDA 12.8",
    "# lerobot 0.4.4 要求 transformers>=4.57.1,<5.0.0（实装 4.57.1 + tokenizers 0.22.1）",
    "",
    server_pkgs,
]
FREEZE.write_text(head + "\n".join(tail_lines) + "\n", encoding="utf-8")
print(f"✓ requirements-freeze.txt 已修（{FREEZE.stat().st_size / 1024:.1f} KB）")
print("  服务器段内容:")
for l in server_pkgs.splitlines()[:16]:
    print("    " + l)

# ---------- 2) .gitignore ----------
GITIGNORE = """# ============================================================
# HKC_Data_Platform .gitignore
# 原则：源码入库；私有依赖/大数据/缓存/凭据 全部排除
# 依赖清单见 requirements-freeze.txt（可据此重建环境）
# ============================================================

# ---- 私有依赖（pip 解压出来的库，可重建，约 1.9GB）----
.deps_lerobot/
.deps_libero/
.vendor/
.wheels/
.venv_lerobot/
__pycache__/
*.py[cod]
.tmp_pip/

# ---- 大数据 / 缓存 / 中间产物（可再生，约 15GB）----
storage/exports/
storage/hf_cache/
storage/xvla/
storage/libero/
storage/zhiyan_urdf/
storage/sim_runs/
storage/eval_runs/
storage/pretrained/
storage/sessions/
storage/_docs_back/
storage/remote_qc_cache/
storage/datasets/
storage/models/
storage/train_runs/

# ---- 压缩包/原始数据（体积大，单独备份）----
*.zip
*.7z
zhiyan_dataset/

# ---- 敏感凭据（绝不入库！）----
storage/ssh/
storage/remote.json
storage/remote_qc.json
storage/vlm_config.json
*.pem
*.key

# ---- 数据库（二进制且频繁变化，单独备份；schema 见 docs/db-schema.sql）----
storage/platform.db
storage/platform.db-wal
storage/platform.db-shm

# ---- 运行时/日志 ----
*.log
*.log.done
storage/reports/qc_*.json
.DS_Store
Thumbs.db
"""
(ROOT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
print(f"\n✓ .gitignore 已写入（{len(GITIGNORE.splitlines())} 行）")

# ---------- 3) 导出 DB schema（代替把 .db 放进 git）----------
import sqlite3  # noqa: E402

dbp = ROOT / "storage" / "platform.db"
if dbp.is_file():
    con = sqlite3.connect(dbp)
    ddl = "\n".join(row[0] for row in con.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type DESC, name") if row[0])
    con.close()
    (ROOT / "docs" / "db-schema.sql").write_text(
        "-- 由 storage/platform.db 导出（git 不跟踪 .db，只跟踪 schema）\n" + ddl + "\n", encoding="utf-8")
    print(f"✓ docs/db-schema.sql 已导出（{len(ddl)} 字节）")

# ---------- 4) git init + add + commit ----------
def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(f"   $ git {' '.join(args)}  → {p.returncode}")
    if p.stdout.strip():
        print("     " + p.stdout.strip()[:900].replace("\n", "\n     "))
    if p.stderr.strip():
        print("     [stderr] " + p.stderr.strip()[:500].replace("\n", "\n     "))
    if check and p.returncode != 0:
        raise SystemExit(f"git {args[0]} 失败")
    return p


print("\n=== git 初始化 ===")
if (ROOT / ".git").is_dir():
    print("   已是 git 仓库，跳过 init")
else:
    git("init", "-b", "main")
    git("config", "user.name", "HKC Data Platform")
    git("config", "user.email", "dev@hkc.local")

git("add", "-A")
st = git("status", "--short", check=False)
lines = [l for l in st.stdout.splitlines() if l.strip()]
print(f"\n=== 暂存文件 {len(lines)} 个（只显示前 30）===")
for l in lines[:30]:
    print("   " + l)
if len(lines) > 30:
    print(f"   …还有 {len(lines) - 30} 个")

n = len(lines)
if n:
    git("commit", "-m",
        "初始提交：VLA 数采平台（数据接入/质检/预处理/标注/导出/训练/仿真）\n\n"
        "- 排除：私有依赖(.deps_*/.vendor)、大数据(storage 下 exports/hf_cache/xvla/models/datasets/train_runs)、\n"
        "        凭据(storage/ssh、remote*.json、vlm_config.json)、SQLite 二进制\n"
        "- 依赖清单：requirements-freeze.txt（含本机各目录 + 服务器训练环境）\n"
        "- 数据库 schema：docs/db-schema.sql")
    git("log", "--oneline", "-3")
print("\n=== 仓库体积统计 ===")
git("count-objects", "-vH", check=False)
print(f"\n工作区文件总大小: {sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file() and '.git' not in f.parts) / 1048576:.1f} MB（含被忽略的大目录）")
