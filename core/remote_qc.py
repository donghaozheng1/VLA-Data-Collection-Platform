"""服务器端质检（lerobot-doctor）——本机界面读结果，重活放服务器。

架构与 VLM 质检一致：**平台后端 → 网络 → 服务器上的工具**，只是这里走 SSH 而不是 HTTP。
（VLM 是常驻模型服务，适合 HTTP；lerobot-doctor 是一次性 CLI，走 SSH 更简单、不需要常驻进程。）

服务器上的部署（已验证，零 sudo、不依赖 HuggingFace）：
    /externalDisk/data/qcenv/lib     pip install --target 装的 lerobot-doctor + 依赖
    /externalDisk/data/qcenv/data    产出的 JSON
    /externalDisk/data/qcenv/logs    日志
所有缓存（XDG_CACHE_HOME / HF_HOME）都指到服务器 1.8T 盘，绝不写服务器系统盘。

鲁棒性要点（逐条都有对应测试）：
- 所有 SSH 调用**必有超时**，绝不挂死；BatchMode=yes 保证不会卡在密码提示上
- 远端路径**全部 shlex.quote**，杜绝命令注入；数据集名做字符白名单校验
- 解析出的服务器路径**必须落在配置的 roots 之内**，防止越权读别的目录
- lerobot-doctor 有 FAIL 时退出码=1，但 stdout 仍是合法 JSON —— **不能把非零退出当失败**
- 远端不可达 / 超时 / JSON 损坏 → 返回一条 warning finding，**绝不让质检任务崩溃**
- 缓存按**远端目录 mtime** 失效，不是按时间猜
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from . import audit, db, jobs

CONFIG_PATH = db.STORAGE / "remote_qc.json"
CACHE_DIR = db.STORAGE / "remote_qc_cache"

#: lerobot-doctor 的 11 项检查 → 我们的分类键与中文名
#: （顺带预留 kinematics，等它支持了直接接上）
CHECK_CATEGORY: dict[str, tuple[str, str]] = {
    "Metadata & Format Compliance": ("metadata", "元数据"),
    "Temporal Consistency": ("temporal", "时序"),
    "Action Quality": ("action", "动作质量"),
    "Video Integrity": ("video", "视频完整性"),
    "Data Distribution": ("stats", "统计分布"),
    "Episode Health": ("episodes", "episode 健康度"),
    "Feature Consistency": ("consistency", "特征一致性"),
    "Training Readiness": ("readiness", "训练就绪性"),
    "Anomaly Detection": ("anomaly", "异常检测"),
    "Portability": ("portability", "可移植性"),
    "Per-Episode Summary": ("per_episode", "逐 episode 汇总"),
    "Kinematics": ("kinematics", "运动学"),
}
CATEGORY_LABEL = {key: label for key, label in CHECK_CATEGORY.values()}

#: doctor 的级别 → 平台的级别。**FAIL 也不是 error 自动打回**：
#: 远端工具的判据和我们不同，落成 warning 交人复核更稳（要打回请用本地结构规则）。
SEVERITY_MAP = {"FAIL": "warning", "WARN": "warning", "PASS": "info", "ERROR": "warning"}

#: 默认值**刻意等于 lerobot-doctor 自己的硬编码值** —— 这个功能的目的是
#: "让你能改"，不是"我替你改"。想按遥操数据放宽，用页面上的预设按钮（显式动作）。
DOCTOR_THRESHOLD_DEFAULTS: dict = {
    "rejudge": True,
    # 冻结：doctor 取"最长连续相同段 / episode 长度 ≥ 5%"
    # 对遥操数据（块内保持、刻意停顿）偏严，一个 1 秒的停顿就会触发。
    "frozen_max_run_pct": 0.05,
    # 裁剪：doctor 拿"本 episode 自身的最小/最大值"当参照（不是关节限位）
    "clip_extreme_pct": 0.50,
    # 突变：doctor 用跨维 z 分数均值 > 8σ；零膨胀数据（大量 0）会把 σ 压低
    "jump_sigma": 8.0,
    # 重判为不合规时的级别（**永远不是 error**：VLM/第三方工具的判定不该触发打回）
    "severity": "warning",
    # 我们判定"没问题"但 doctor 报了，是否也留一条 info 说明分歧
    "report_disagreements": True,
}

#: 遥操数据放宽预设（用户显式点击才应用，不静默生效）
DOCTOR_PRESET_TELEOP: dict = {
    "frozen_max_run_pct": 0.50,
    "clip_extreme_pct": 0.90,
    "jump_sigma": 12.0,
}

DEFAULTS: dict = {
    "enabled": False,
    "host": "192.168.123.105",
    "user": "z123",
    "port": 22,
    "key_file": "",                 # 空 = 用 storage/ssh/platform_key
    "qcenv_base": "/externalDisk/data/qcenv",
    "python": "python3",
    # 允许的服务器数据集根目录（白名单，防止越权读别的路径）
    "roots": [
        "/externalDisk/data/vla/datasets/local",
        "/externalDisk/data/vla/datasets",
        "/externalDisk/data/datasets",
    ],
    "timeout_s": 1800,              # 单次 doctor 总超时
    "connect_timeout_s": 15,        # 连通性探测超时
    "max_episodes": 0,              # 0 = 全部
    "checks": "",                   # 逗号分隔，空 = 全部 11 项
    "use_cache": True,
    "keep_pass_messages": False,    # 是否把 PASS 明细也写成 info finding（默认不写，避免刷屏）
    # FAIL 是否升成 error（error 才会触发「打回重采」）。
    # 默认 False：远端工具的判据和我们不同，先把结论交人复核更稳。
    "fail_as_error": False,
    # ---- 上传（本机数据集在服务器上没有对应物时，先传上去再质检）----
    "upload_enabled": True,
    # 上传落点。**会被自动并进白名单**，所以传完就能自动匹配到。
    "upload_root": "/externalDisk/data/qcenv/data/datasets",
    "upload_max_bytes": 20 * 1024 ** 3,        # 超过这个大小不传（安全阀）
    "upload_min_free_bytes": 5 * 1024 ** 3,    # 服务器剩余空间低于此值不传
    # ---- 阈值可配：用平台阈值重判 doctor 的测量值（默认等于它自己的值）----
    "doctor": dict(DOCTOR_THRESHOLD_DEFAULTS),
}


# 预设定义见文件开头的 DOCTOR_PRESET_TELEOP（只保留一处，避免两处漂移）

# --------------------------------------------------------------------------- 配置


def config_path() -> Path:
    return CONFIG_PATH


def load_config(overrides: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if CONFIG_PATH.is_file():
        try:
            patch = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(patch, dict):
                cfg.update(patch)
        except Exception:  # noqa: BLE001
            pass
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    if not cfg.get("key_file"):
        cfg["key_file"] = str(db.STORAGE / "ssh" / "platform_key")
    return cfg


def save_config(patch: dict) -> dict:
    current: dict = {}
    if CONFIG_PATH.is_file():
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            current = {}
    for key, value in (patch or {}).items():
        current[key] = value
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_config()


# --------------------------------------------------------------------------- SSH 基础


class RemoteError(RuntimeError):
    """远端调用失败（网络/超时/命令出错）。"""


def _ssh_argv(cfg: dict) -> list[str]:
    argv = ["ssh", "-i", str(cfg["key_file"]),
            "-o", "BatchMode=yes",              # 绝不弹密码提示，避免挂死
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=%d" % int(cfg.get("connect_timeout_s") or 15),
            "-p", str(int(cfg.get("port") or 22)),
            f"{cfg['user']}@{cfg['host']}"]
    return argv


def ssh_run(cfg: dict, remote_command: str, timeout_s: int | None = None) -> tuple[int, str, str]:
    """跑一条远端命令（script 模式，避免引号被本地 shell 处理）。

    返回 (returncode, stdout, stderr)。**抛 RemoteError 只在链路层失败时**，
    命令本身返回非零不算异常（doctor 有 FAIL 时就是非零）。
    """
    argv = _ssh_argv(cfg) + ["bash -s"]
    timeout = int(timeout_s or cfg.get("timeout_s") or 1800)
    try:
        proc = subprocess.run(argv, input=remote_command.encode("utf-8"),
                              capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RemoteError(f"本机找不到 ssh 命令：{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteError(f"远端命令超时（{timeout}s）") from exc
    except Exception as exc:  # noqa: BLE001
        raise RemoteError(f"SSH 调用失败：{type(exc).__name__}: {exc}") from exc
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    return proc.returncode, out, err


def probe(cfg: dict | None = None) -> dict:
    """连通性 + 远端工具是否就位。不抛异常，只返回事实。"""
    cfg = cfg or load_config()
    result = {"ok": False, "host": cfg.get("host"), "user": cfg.get("user"),
              "doctor": False, "version": "", "detail": "",
              "base": cfg.get("qcenv_base"), "key_file": cfg.get("key_file")}
    if not Path(str(cfg.get("key_file"))).is_file():
        result["detail"] = f"本机找不到 SSH 私钥：{cfg.get('key_file')}"
        return result
    base = shlex.quote(str(cfg["qcenv_base"]))
    cmd = (f"echo HOST=$(hostname); "
           f"[ -x {base}/lib/bin/lerobot-doctor ] && echo DOCTOR=yes || echo DOCTOR=no; "
           f"PYTHONPATH={base}/lib {shlex.quote(str(cfg.get('python') or 'python3'))} "
           f"-c \"import lerobot_doctor as m;print('VER='+getattr(m,'__version__','?'))\" "
           f"2>/dev/null || true\n")
    try:
        rc, out, err = ssh_run(cfg, cmd, timeout_s=int(cfg.get("connect_timeout_s") or 15) + 10)
    except RemoteError as exc:
        result["detail"] = str(exc)
        return result
    result["ok"] = "HOST=" in out
    result["doctor"] = "DOCTOR=yes" in out
    for line in out.splitlines():
        if line.startswith("VER="):
            result["version"] = line[4:].strip()
        if line.startswith("HOST="):
            result["remote_host"] = line[5:].strip()
    if not result["ok"]:
        result["detail"] = f"SSH 没返回预期输出（rc={rc}）：{(err or out)[:200]}"
    elif not result["doctor"]:
        result["detail"] = (f"SSH 通了，但远端没找到 {cfg.get('qcenv_base')}/lib/bin/lerobot-doctor"
                            "（环境未安装或路径变了）")
    else:
        result["detail"] = f"就绪，lerobot-doctor {result['version']}"
    return result


# --------------------------------------------------------------------------- 路径解析


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")


def safe_name(name: str) -> bool:
    """数据集名白名单校验：杜绝 `../`、空格、引号、分号等注入手法。"""
    return bool(name) and bool(_NAME_RE.match(name)) and ".." not in name


def remote_stat_mtime(cfg: dict, path: str, timeout_s: int = 20) -> int:
    """远端目录（或文件）的 mtime，用来给缓存定版本。取不到返回 0。"""
    cmd = f"stat -c %Y {shlex.quote(path)} 2>/dev/null || echo 0\n"
    try:
        _rc, out, _err = ssh_run(cfg, cmd, timeout_s=timeout_s)
    except RemoteError:
        return 0
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return 0


def remote_isdir(cfg: dict, path: str, timeout_s: int = 20) -> bool:
    cmd = f"[ -d {shlex.quote(path)} ] && echo YES || echo NO\n"
    try:
        _rc, out, _err = ssh_run(cfg, cmd, timeout_s=timeout_s)
    except RemoteError:
        return False
    return "YES" in out


def effective_roots(cfg: dict) -> list[str]:
    """生效的白名单 = 配置的 roots + 上传落点。

    上传目录必须自动并入白名单，否则「同步上来」的数据集还是匹配不到。
    """
    roots = [r for r in (cfg.get("roots") or []) if r]
    up = (cfg.get("upload_root") or "").strip()
    if up and up not in roots:
        roots.append(up)
    return roots


def resolve_server_path(cfg: dict, dataset: dict) -> dict:
    """把本机数据集对应到服务器上的路径。

    顺序：① datasets.server_path 人工指定（优先）
          ② 在生效白名单里按「目录名 == 数据集名」找
    并**强制校验**结果落在白名单之内。
    """
    name = str(dataset.get("name") or "")
    roots = effective_roots(cfg)
    out = {"path": "", "source": "", "detail": "", "checked": []}

    explicit = (dataset.get("server_path") or "").strip()
    if explicit:
        safe, why = _safe_remote_path(explicit)
        if not safe:
            out["detail"] = f"server_path 被拒绝：{why}"
            return out
        if not _within_roots(explicit, roots):
            out["detail"] = (f"server_path 不在允许的 roots 内，已拒绝：{explicit}"
                             f"（roots={roots}）")
            return out
        out.update({"path": explicit, "source": "datasets.server_path",
                    "detail": "使用手工指定的 server_path"})
        return out

    if not safe_name(name):
        out["detail"] = f"数据集名不合法，无法自动匹配：{name!r}"
        return out
    candidates = [f"{r.rstrip('/')}/{name}" for r in roots]
    out["checked"] = candidates
    for cand in candidates:
        safe, _why = _safe_remote_path(cand)
        if not safe or not _within_roots(cand, roots):
            continue
        if remote_isdir(cfg, cand):
            out.update({"path": cand, "source": "roots 自动匹配",
                        "detail": f"在 roots 中命中：{cand}"})
            return out
    out["detail"] = ("在服务器上没找到该数据集。请把 server_path 填到 datasets.server_path，"
                     f"或确认 roots 配置（已查：{candidates}）")
    return out


def _posix_norm(path: str) -> str:
    """按 POSIX 语义规范化路径，**不碰本地文件系统**。

    踩过的坑：用 `Path(path).resolve()` 处理远端 Linux 路径，在 Windows 上会把
    `/externalDisk/...` 解析成 `C:\\externalDisk\\...`（分隔符变 `\\`），前缀比较永远失败。
    远端路径必须当纯字符串处理。

    顺带 `..` 会在归一化时被折叠掉，这本身就是防越权的一道保险。
    """
    parts: list[str] = []
    for part in str(path).split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)


#: shell 元字符 / 引号 / 通配符 / 换行 —— 远端路径里一律不允许出现
_META_RE = re.compile(r"""[;&|`$(){}<>*?\[\]!~'"\\\n\r\t]""")


def _safe_remote_path(path: str) -> tuple[bool, str]:
    """远端路径的第二层防御：拒绝 shell 元字符。

    第一层是 `shlex.quote`（保证即使有元字符也只当普通字符），
    这一层是"干脆不让可疑路径进来"——多层防御比只靠一层稳。
    """
    if not path or not str(path).startswith("/"):
        return False, "必须是绝对路径（以 / 开头）"
    if _META_RE.search(path):
        return False, f"路径含可疑字符（引号/分号/反引号/通配符/换行等）：{path!r}"
    if ".." in str(path).split("/"):
        return False, f"路径含 .. 分量：{path!r}"
    if len(path) > 1024:
        return False, "路径过长"
    return True, ""


def _within_roots(path: str, roots: list[str]) -> bool:
    """路径必须严格位于某个 roots 之下（防 `..` 逃逸与前缀伪装）。"""
    if not path or not str(path).startswith("/"):
        return False
    norm = _posix_norm(path)
    for root in roots or []:
        root_norm = _posix_norm(root)
        if root_norm == "/":
            return True
        if norm == root_norm or norm.startswith(root_norm + "/"):
            return True
    return False


# --------------------------------------------------------------------------- 调用与缓存


def _cache_key(cfg: dict, server_path: str, mtime: int) -> str:
    payload = json.dumps({"p": server_path, "m": mtime,
                          "checks": cfg.get("checks") or "",
                          "max": int(cfg.get("max_episodes") or 0),
                          "base": cfg.get("qcenv_base")}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _cache_read(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _cache_write(key: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def run_doctor(cfg: dict, server_path: str, *, force: bool = False,
               progress=None) -> dict:
    """在服务器上跑 lerobot-doctor，返回结构化结果。

    返回 dict：
      ok / json / stdout_tail / stderr / returncode / cached / mtime / seconds / error
    """
    started = time.time()
    safe, why = _safe_remote_path(server_path)
    if not safe:
        return {"ok": False, "error": f"远端路径被拒绝：{why}", "json": None,
                "cached": False, "mtime": 0, "seconds": 0.0,
                "stdout_tail": "", "stderr": "", "returncode": -1}
    mtime = remote_stat_mtime(cfg, server_path)
    key = _cache_key(cfg, server_path, mtime)
    if cfg.get("use_cache", True) and not force and mtime:
        cached = _cache_read(key)
        if cached:
            cached = dict(cached)
            cached["cached"] = True
            cached["seconds"] = round(time.time() - started, 2)
            return cached

    base = str(cfg.get("qcenv_base")).rstrip("/")
    python = str(cfg.get("python") or "python3")
    args = ["--json"]
    if cfg.get("checks"):
        args += ["--checks", str(cfg["checks"])]
    if int(cfg.get("max_episodes") or 0) > 0:
        args += ["--max-episodes", str(int(cfg["max_episodes"]))]

    # 用 `python -c '...' <path> <args>` 调入口：不依赖 bin/ 脚本是否存在、是否可执行，
    # 也不需要在远端拼一长串带引号的命令行。argparse 会忽略 argv[0]，所以位置参数照常生效。
    py_args = " ".join(shlex.quote(a) for a in [server_path, *args])
    entry = shlex.quote("from lerobot_doctor.cli import main; main()")
    qpython = shlex.quote(python)
    errlog = shlex.quote(base + "/logs/_doctor_stderr.txt")

    lines = [
        f"mkdir -p {shlex.quote(base + '/logs')} {shlex.quote(base + '/cache')} "
        f"{shlex.quote(base + '/hf_home')}",
        # 所有缓存都指到 1.8T 盘，绝不写服务器系统盘
        f"export PYTHONPATH={shlex.quote(base + '/lib')}",
        f"export XDG_CACHE_HOME={shlex.quote(base + '/cache')}",
        f"export HF_HOME={shlex.quote(base + '/hf_home')}",
        f"export HF_HUB_CACHE={shlex.quote(base + '/hf_home/hub')}",
        f"echo __CMD__ {qpython} -c lerobot_doctor.cli:main {py_args} >&2",
        # doctor 有 FAIL 时退出码=1，但 stdout 仍是完整 JSON —— 不用 || true，
        # 手动取 RC，避免把非零退出当失败而丢掉结果
        f"{qpython} -c {entry} {py_args} 2>{errlog}",
        "RC=$?",
        f"cat {errlog} >&2 2>/dev/null; rm -f {errlog}",
        "echo __RC__ $RC >&2",
    ]
    command = "\n".join(lines) + "\n"

    if progress:
        progress(f"已在服务器上启动质检：{server_path}")
    try:
        rc, out, err = ssh_run(cfg, command)
    except RemoteError as exc:
        return {"ok": False, "error": str(exc), "json": None, "cached": False,
                "mtime": mtime, "seconds": round(time.time() - started, 2),
                "stdout_tail": "", "stderr": "", "returncode": -1}

    payload = _extract_json(out)
    result = {
        "ok": payload is not None,
        "json": payload,
        "returncode": rc,
        "cached": False,
        "mtime": mtime,
        "seconds": round(time.time() - started, 2),
        "stdout_tail": out[-2000:],
        "stderr": err[-2000:],
        "error": "" if payload is not None else _diagnose(out, err),
    }
    if payload is not None and cfg.get("use_cache", True) and mtime:
        _cache_write(key, {k: v for k, v in result.items() if k != "cached"})
    return result


def _extract_json(text: str) -> dict | None:
    """从 stdout 里抠出 JSON 对象（容忍前后有日志/警告）。"""
    if not text:
        return None
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start:end + 1])
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _diagnose(out: str, err: str) -> str:
    blob = (err or "") + (out or "")
    if "No such file or directory" in blob:
        return "服务器上找不到该数据集路径"
    if "Permission denied" in blob or "publickey" in blob:
        return "SSH 认证失败（检查密钥/权限）"
    if "ModuleNotFoundError" in blob:
        return f"服务器上的 lerobot-doctor 环境异常：{blob[-200:]}"
    if not blob.strip():
        return "远端没有输出（可能超时或命令未执行）"
    return f"远端输出无法解析为 JSON：{blob[-300:]}"


# --------------------------------------------------------------------------- 映射成 findings


_EPISODE_RE = re.compile(r"[Ee]pisode\s+(\d+)")


def severity_for(doctor_severity: str, cfg: dict) -> str:
    """doctor 级别 → 平台级别。FAIL 默认降为 warning（见 fail_as_error 配置）。"""
    sev = str(doctor_severity or "").upper()
    if sev == "FAIL" and cfg.get("fail_as_error"):
        return "error"
    return SEVERITY_MAP.get(sev, "info")


def map_to_findings(payload: dict, cfg: dict) -> list[dict]:
    """把 lerobot-doctor 的 JSON 映射成平台 findings（结构与 lerobot.check_episode 一致）。"""
    findings: list[dict] = []
    checks = payload.get("checks") or []
    if not isinstance(checks, list):
        return findings
    keep_pass = bool(cfg.get("keep_pass_messages"))
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "未知检查")
        category, label = CHECK_CATEGORY.get(name, ("remote", name))
        for msg in check.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            sev = str(msg.get("severity") or "").upper()
            text = str(msg.get("message") or "").strip()
            if not text:
                continue
            if sev == "PASS" and not keep_pass:
                continue
            episode_index = None
            match = _EPISODE_RE.search(text)
            if match:
                try:
                    episode_index = int(match.group(1))
                except ValueError:
                    episode_index = None
            findings.append({
                "rule": f"{label}·{name}",
                "severity": severity_for(sev, cfg),
                "detail": text,
                "measured": sev,
                "expected": "PASS",
                "category": category,
                "check_version": f"doctor-{payload.get('version') or '?'}",
                "episode_index": episode_index,
                "source": "server",
            })
    return findings


def summary_of(payload: dict) -> dict:
    checks = payload.get("checks") or []
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for check in checks if isinstance(checks, list) else []:
        sev = str((check or {}).get("severity") or "").upper()
        if sev in counts:
            counts[sev] += 1
    return {
        "overall_severity": payload.get("overall_severity"),
        "counts": counts,
        "dataset_name": payload.get("dataset_name"),
        "codebase_version": payload.get("codebase_version"),
        "total_episodes": payload.get("total_episodes"),
        "total_frames": payload.get("total_frames"),
        "fps": payload.get("fps"),
        "checks": [{"name": (c or {}).get("name"),
                    "severity": (c or {}).get("severity"),
                    "label": CHECK_CATEGORY.get(str((c or {}).get("name")), ("", ""))[1],
                    "messages": len((c or {}).get("messages") or [])}
                   for c in (checks if isinstance(checks, list) else [])],
    }


def coverage_of(payload: dict) -> dict:
    """哪些检查真的跑了（doctor 输出里每一项都有名字，等于 100% 覆盖）。"""
    checks = payload.get("checks") or []
    names = [str((c or {}).get("name")) for c in (checks if isinstance(checks, list) else [])]
    total = len(names)
    return {"checks": total,
            "categories": {CHECK_CATEGORY.get(n, ("", n))[1] or n: 1.0 for n in names}}


# --------------------------------------------------------------------------- 上传


MARKER_NAME = ".upload_meta.json"


def local_fingerprint(local_dir) -> dict:
    """上传用的**廉价**指纹：文件数 + 总字节 + 最新 mtime。

    故意不做内容哈希：大目录（例如 1.4GB / 200+ episode）全量读太慢，
    而这里只需要判断「是不是和上次传的同一批文件」。
    """
    root = Path(str(local_dir))
    out = {"files": 0, "bytes": 0, "mtime": 0, "ok": False}
    if not root.is_dir():
        return out
    newest = 0.0
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        out["files"] += 1
        out["bytes"] += stat.st_size
        newest = max(newest, stat.st_mtime)
    out["mtime"] = int(newest)
    out["ok"] = out["files"] > 0
    return out


def remote_free_bytes(cfg: dict, path: str) -> int:
    """远端文件系统可用字节；取不到返回 -1（表示"未知"，不是"没有空间"）。"""
    cmd = (f"df -PB1 {shlex.quote(path)} 2>/dev/null | sed -n '2p' "
           f"| tr -s ' ' | cut -d' ' -f4\n")
    try:
        _rc, out, _err = ssh_run(cfg, cmd, timeout_s=20)
    except RemoteError:
        return -1
    for token in out.split():
        if token.isdigit():
            return int(token)
    return -1


def read_upload_marker(cfg: dict, target: str) -> dict:
    cmd = f"cat {shlex.quote(target.rstrip('/') + '/' + MARKER_NAME)} 2>/dev/null || true\n"
    try:
        _rc, out, _err = ssh_run(cfg, cmd, timeout_s=20)
    except RemoteError:
        return {}
    return _extract_json(out) or {}


def write_upload_marker(cfg: dict, target: str, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False)
    cmd = (f"mkdir -p {shlex.quote(target)}\n"
           f"cat > {shlex.quote(target.rstrip('/') + '/' + MARKER_NAME)} "
           f"<<'UPLOADEOF'\n{body}\nUPLOADEOF\n")
    try:
        ssh_run(cfg, cmd, timeout_s=30)
    except RemoteError:
        pass


def upload_status(cfg: dict, dataset: dict) -> dict:
    """上传前的"算账"：不发一个字节，先把能不能传、要不要传算清楚。"""
    name = str(dataset.get("name") or "")
    local = str(dataset.get("path") or "")
    root = str(cfg.get("upload_root") or "").rstrip("/")
    target = f"{root}/{name}" if (root and safe_name(name)) else ""
    info = {"name": name, "local_path": local, "target": target,
            "local": local_fingerprint(local), "already": False, "marker": {},
            "free_bytes": -1, "allowed": False, "reason": ""}

    if not safe_name(name):
        info["reason"] = f"数据集名不合法，不能上传：{name!r}"
        return info
    if not cfg.get("upload_enabled", True):
        info["reason"] = "配置里关闭了上传（upload_enabled=false）"
        return info
    if not info["local"]["ok"]:
        info["reason"] = f"本机数据集目录不可读或为空：{local or '(空)'}"
        return info
    if not target:
        info["reason"] = "无法确定上传目标路径（upload_root 未配置）"
        return info
    safe, why = _safe_remote_path(target)
    if not safe or not _within_roots(target, effective_roots(cfg)):
        info["reason"] = f"上传目标不在白名单内：{target}（{why}）"
        return info

    limit = int(cfg.get("upload_max_bytes") or 0)
    if limit and info["local"]["bytes"] > limit:
        info["reason"] = (f"数据集 {info['local']['bytes'] / 1024 ** 3:.2f}GB 超过上传上限 "
                          f"{limit / 1024 ** 3:.2f}GB，已跳过")
        return info

    info["free_bytes"] = remote_free_bytes(cfg, target if remote_isdir(cfg, target) else root)
    reserve = int(cfg.get("upload_min_free_bytes") or 0)
    if info["free_bytes"] >= 0 and info["free_bytes"] < info["local"]["bytes"] + reserve:
        info["reason"] = (f"服务器剩余空间不足：可用 {info['free_bytes'] / 1024 ** 3:.2f}GB，"
                          f"需要 {info['local']['bytes'] / 1024 ** 3:.2f}GB"
                          f" + 预留 {reserve / 1024 ** 3:.2f}GB")
        return info

    marker = read_upload_marker(cfg, target)
    info["marker"] = marker
    same = (marker.get("files") == info["local"]["files"]
            and marker.get("bytes") == info["local"]["bytes"]
            and marker.get("mtime") == info["local"]["mtime"])
    if same:
        info["already"] = True
        info["reason"] = "服务器上已有同名且指纹一致的副本，可跳过上传"
    else:
        info["reason"] = "需要在服务器上创建/更新副本"
    info["allowed"] = True
    return info


def upload_dataset(cfg: dict, dataset: dict, progress=None, force: bool = False) -> dict:
    """把本机数据集上传到服务器。复用 `core/remote.py` 里**已经跑通**的上传实现。

    core/remote 的 RemoteCli 走 tar 流，不依赖 paramiko，也不走 TLS/Schannel —— 本机可用。
    """
    status = upload_status(cfg, dataset)
    if not status["allowed"]:
        return {"ok": False, "reason": status["reason"], "status": status}
    if status["already"] and not force:
        return {"ok": True, "skipped": True, "target": status["target"],
                "reason": status["reason"], "status": status}

    started = time.time()
    if progress:
        progress(f"上传到服务器：{status['target']}"
                 f"（{status['local']['bytes'] / 1024 ** 2:.1f}MB / "
                 f"{status['local']['files']} 个文件）")
    try:
        from . import remote as remote_mod
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"无法加载 core.remote：{type(exc).__name__}: {exc}",
                "status": status}

    remote_cfg = {
        "host": cfg.get("host"), "port": int(cfg.get("port") or 22),
        "user": cfg.get("user"), "password": "", "key_file": cfg.get("key_file"),
        "remote_dataset_dir": status["target"], "enabled": True,
    }
    try:
        backend = remote_mod.make_remote(remote_cfg)
        with backend as client:
            client.mkdirs(status["target"])
            copied = client.upload_dir(status["local_path"], status["target"])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"上传失败：{type(exc).__name__}: {exc}",
                "status": status, "seconds": round(time.time() - started, 2)}

    write_upload_marker(cfg, status["target"], {
        "name": status["name"], "files": status["local"]["files"],
        "bytes": status["local"]["bytes"], "mtime": status["local"]["mtime"],
        "uploaded_at": db.now(),
    })
    return {"ok": True, "skipped": False, "target": status["target"],
            "copied": copied, "seconds": round(time.time() - started, 2),
            "reason": f"已上传到 {status['target']}", "status": status}


# --------------------------------------------------------------------------- 阈值可配：解析 + 重判

#: 从 doctor 的**文本消息**里抽结构化测量值。
#: 为什么只能解析文本：`runner.py` 里 `CheckResult` 只有 severity + 已格式化好的字符串，
#: 没有任何结构化数值字段。每个正则都对真实输出验证过；匹配不上就跳过，
#: **绝不让解析失败影响任务**。
#:
#: 注意：真实消息里 "Episode ..." 常带 `action: ` 前缀（实测
#: `action: Episode 2 has 1 sudden large action jumps`），所以前缀一律可选。
_COL_PREFIX = r"^(?:(?P<col>[\w.]+):\s*)?"
_METRIC_RES: list[tuple[str, re.Pattern]] = [
    ("clip_min", re.compile(
        r"^(?P<col>[\w.]+)\[(?P<dim>\d+)\]:\s*(?P<pct>[\d.]+)% of values at minimum")),
    ("clip_max", re.compile(
        r"^(?P<col>[\w.]+)\[(?P<dim>\d+)\]:\s*(?P<pct>[\d.]+)% of values at maximum")),
    ("frozen_run", re.compile(
        _COL_PREFIX + r"(?P<pct>[\d.]+)% of episode (?P<ep>\d+) is consecutive"
        r" identical actions \(frozen\)")),
    ("frozen_run", re.compile(
        _COL_PREFIX + r"Episode (?P<ep>\d+):\s*(?P<pct>[\d.]+)% of (?P<col2>[\w.]+) frozen"
        r" \(consecutive identical\)")),
    # 实测真实消息是 `action: Episode 2 has 1 sudden large action jumps` ——
    # **Episode 后面没有冒号**，所以这里冒号必须可选。
    ("jumps", re.compile(
        _COL_PREFIX + r"Episode (?P<ep>\d+):?\s*has (?P<n>\d+) sudden large action jumps")),
    ("jumps", re.compile(
        _COL_PREFIX + r"Episode (?P<ep>\d+):?\s*(?P<n>\d+) action jump\(s\)")),
    ("nonmono", re.compile(
        _COL_PREFIX + r"Episode (?P<ep>\d+):?\s*(?P<n>\d+) non-monotonic timestamp\(s\)")),
    ("gaps", re.compile(
        _COL_PREFIX + r"Episode (?P<ep>\d+):?\s*(?P<n>\d+) dropped frame gap\(s\)")),
]


def parse_doctor_metrics(payload: dict) -> dict:
    """把 doctor 的文本结论解析成结构化测量值。

    返回 {"metrics": [...], "parsed": n, "unparsed": [文本...], "coverage": 0~1}
    —— 有意给出 `coverage`，这样**解析失败是可见的**，不会静默丢数据。
    """
    metrics: list[dict] = []
    unparsed: list[str] = []
    candidates = 0
    for check in payload.get("checks") or []:
        if not isinstance(check, dict):
            continue
        for msg in check.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            sev = str(msg.get("severity") or "").upper()
            if sev == "PASS":
                continue
            text = str(msg.get("message") or "").strip()
            if not text:
                continue
            candidates += 1
            hit = False
            for kind, rx in _METRIC_RES:
                m = rx.match(text)
                if not m:
                    continue
                gd = m.groupdict()
                item = {"kind": kind, "check": check.get("name"), "severity": sev,
                        "text": text}
                # col2 是为避免正则组名重复而用的备选名，这里合并回 col
                col = gd.get("col") or gd.get("col2")
                if col:
                    item["col"] = col
                for key in ("dim", "ep", "n"):
                    if gd.get(key) is not None:
                        item[key] = int(gd[key])
                if gd.get("pct") is not None:
                    item["pct"] = float(gd["pct"]) / 100.0
                metrics.append(item)
                hit = True
                break
            if not hit:
                unparsed.append(text)
    return {"metrics": metrics, "parsed": len(metrics), "candidates": candidates,
            "unparsed": unparsed[:20],
            "coverage": round(len(metrics) / candidates, 3) if candidates else 1.0}


def rejudge_doctor(metrics: list[dict], thresholds: dict) -> list[dict]:
    """用平台阈值重新判定 —— 只对**有可配阈值**的那几类指标。"""
    sev = str(thresholds.get("severity") or "warning")
    findings: list[dict] = []
    for item in metrics:
        kind = item.get("kind")
        pct = item.get("pct")
        if kind == "frozen_run":
            limit = float(thresholds.get("frozen_max_run_pct") or 0.05)
            if pct is not None and pct > limit:
                findings.append({
                    "rule": "冻结动作（平台阈值）", "severity": sev,
                    "detail": (f"episode {item.get('ep')}：最长连续相同动作段占 {pct:.1%}，"
                               f"超过平台阈值 {limit:.0%}"
                               f"（lerobot-doctor 自己的阈值是 5%）"),
                    "measured": f"{pct:.3f}", "expected": f"< {limit:.2f}",
                    "category": "action", "episode_index": item.get("ep"),
                })
        elif kind in ("clip_min", "clip_max"):
            limit = float(thresholds.get("clip_extreme_pct") or 0.5)
            side = "最小值" if kind == "clip_min" else "最大值"
            if pct is not None and pct > limit:
                findings.append({
                    "rule": "数值贴极值（平台阈值）", "severity": sev,
                    "detail": (f"{item.get('col')}[{item.get('dim')}] 有 {pct:.1%} 的值落在"
                               f"该维{side}，超过平台阈值 {limit:.0%}"
                               f"（注意 doctor 是拿「本 episode 自身极值」比的，不是关节限位）"),
                    "measured": f"{pct:.3f}", "expected": f"< {limit:.2f}",
                    "category": "action", "episode_index": None,
                })
        elif kind == "jumps":
            n = item.get("n")
            # 突变：doctor 用 8σ 判，这里只把数量透传，阈值由平台声明
            limit = float(thresholds.get("jump_sigma") or 8.0)
            if limit >= 8.0 and n:
                findings.append({
                    "rule": "动作突变（平台阈值）", "severity": sev,
                    "detail": (f"episode {item.get('ep')}：{n} 处大幅跳变"
                               f"（判据 σ>{limit:g}；σ 在被大量 0 压缩的零膨胀数据上会失真，"
                               f"可调高；本机质检另有「幅度相对」判据可交叉参考）"),
                    "measured": f"{n} 处", "expected": "0 处",
                    "category": "action", "episode_index": item.get("ep"),
                })
    return findings


def disagreement_notes(metrics: list[dict], thresholds: dict,
                       platform_findings: list[dict]) -> list[dict]:
    """doctor 报了、但按平台阈值不算问题的，留一条 info 说明分歧（可关）。"""
    if not thresholds.get("report_disagreements", True):
        return []
    flagged = {(f.get("rule"), f.get("episode_index")) for f in platform_findings}
    out: list[dict] = []
    for item in metrics:
        kind, ep = item.get("kind"), item.get("ep")
        rule = {"frozen_run": "冻结动作（平台阈值）", "clip_min": "数值贴极值（平台阈值）",
                "clip_max": "数值贴极值（平台阈值）",
                "jumps": "动作突变（平台阈值）"}.get(kind)
        if not rule or (rule, ep) in flagged:
            continue
        out.append({
            "rule": "平台阈值判定：不构成问题", "severity": "info",
            "detail": (f"lerobot-doctor 报了「{item.get('text')[:90]}」，"
                       f"但按平台阈值不算问题（阈值可在质检配置里改）"),
            "measured": item.get("severity"), "expected": "低于平台阈值",
            "category": "action", "episode_index": ep,
        })
    return out


# --------------------------------------------------------------------------- 任务处理器

# 任务标签（jobs.KIND_LABELS 是注册表式，这里补一条，避免改 jobs.py）
jobs.KIND_LABELS.setdefault("qc_remote", "服务器质检")


@jobs.handler("qc_remote")
def handler_remote_qc(job, payload: dict) -> dict:
    """在服务器上跑 lerobot-doctor，把结果落进 qc_findings。

    **鲁棒性契约**：远端任何失败（SSH 不通 / 超时 / 路径找不到 / JSON 损坏 / 工具异常）
    都**不抛异常**，而是落一条 warning finding 并让任务正常结束。
    理由：远端是外部依赖，它挂了不该让平台的质检链路整个失败，
    就像 VLM 不可用时只标注"没看成"一样。
    """
    dataset_id = int(payload["dataset_id"])
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError(f"数据集不存在：id={dataset_id}")

    cfg = load_config(payload.get("config") or {})
    force = bool(payload.get("force"))

    run_id = db.execute(
        "INSERT INTO qc_runs (dataset_id, job_id, total, created_at) VALUES (?,?,?,?)",
        (dataset_id, job.id, 0, db.now()))
    db.execute("DELETE FROM qc_findings WHERE run_id=?", (run_id,))

    def finish(findings: list[dict], remote: dict, report_extra: dict | None = None) -> dict:
        rows = [(run_id, dataset_id, f.get("episode_index"), f["rule"], f["severity"],
                 f["detail"], f.get("measured"), f.get("expected"),
                 f.get("category") or "remote", f.get("check_version") or "",
                 f.get("source") or "server") for f in findings]
        if rows:
            db.execute_many(
                "INSERT INTO qc_findings (run_id, dataset_id, episode_index, rule, severity,"
                " detail, measured, expected, category, check_version, source)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        errors = sum(1 for f in findings if f["severity"] == "error")
        warnings = sum(1 for f in findings if f["severity"] == "warning")
        infos = sum(1 for f in findings if f["severity"] == "info")
        by_category: dict = {}
        for f in findings:
            key = f.get("category") or "remote"
            entry = by_category.setdefault(key, {
                "category": key, "label": CATEGORY_LABEL.get(key, key),
                "errors": 0, "warnings": 0, "infos": 0})
            entry["errors" if f["severity"] == "error"
                  else "warnings" if f["severity"] == "warning" else "infos"] += 1
        result = "打回重采" if errors else "通过"
        summary = {
            "dataset": row["name"], "dataset_id": dataset_id, "path": row["path"],
            "total": int(remote.get("total_episodes") or 0),
            "passed": (remote.get("counts") or {}).get("PASS", 0),
            "failed": errors,
            "result": result,
            "findings_count": {"error": errors, "warning": warnings, "info": infos},
            "categories": sorted(by_category.values(),
                                 key=lambda c: (-c["errors"], -c["warnings"])),
            "coverage": ({"episodes": remote.get("total_episodes") or 0,
                          "categories": remote.get("coverage") or {}}),
            "remote": dict(remote, **{
                "source": "server", "check_version": f"doctor-{remote.get('version') or '?'}"}),
            "qc_config": {"version": 1, "fingerprint": ""},
            "generated_at": db.now(),
        }
        if report_extra:
            summary["raw"] = report_extra
        report_path = db.REPORT_DIR / f"qc_remote_{row['name']}_{run_id}.json"
        report_path.write_text(json.dumps({"summary": summary, "episodes": []},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        db.execute(
            "UPDATE qc_runs SET report_path=?, total=?, passed=?, failed=?, result=?,"
            " summary_json=? WHERE id=?",
            (str(report_path), summary["total"], summary["passed"], summary["failed"],
             result, json.dumps(summary, ensure_ascii=False), run_id))
        db.execute("INSERT INTO artifacts (job_id, dataset_id, kind, path, bytes, created_at)"
                   " VALUES (?,?,?,?,?,?)",
                   (job.id, dataset_id, "qc_remote_report", str(report_path),
                    report_path.stat().st_size if report_path.is_file() else 0, db.now()))
        audit.log("WARNING" if (errors or not remote.get("ok")) else "INFO", "数据质检",
                  f"数据集 #{dataset_id} {row['name']} 服务器质检（run{run_id}）："
                  f"{remote.get('detail') or remote.get('reason') or ''}",
                  ref_type="dataset", ref_id=dataset_id,
                  detail={"run_id": run_id, "remote": {k: v for k, v in remote.items()
                                                       if k not in ("stdout_tail", "stderr",
                                                                    "coverage")}})
        return {**summary, "run_id": run_id, "report": str(report_path)}

    def fail(reason: str) -> dict:
        return finish([{
            "rule": "服务器质检·未执行", "severity": "warning",
            "detail": reason, "measured": cfg.get("host"), "expected": "服务器质检可用",
            "category": "remote", "check_version": "", "episode_index": None,
            "source": "server",
        }], {"ok": False, "reason": reason, "host": cfg.get("host"),
             "base": cfg.get("qcenv_base"), "detail": reason})

    # ---- 0) 可选：先把本机数据集传上去（本机数据在服务器上没有对应物时用）----
    upload_info = None
    if payload.get("upload"):
        job.progress(0, 5, "检查上传条件…")
        upload_info = upload_dataset(cfg, row, progress=job.log,
                                     force=bool(payload.get("force_upload")))
        if not upload_info.get("ok"):
            return fail(f"上传到服务器失败：{upload_info.get('reason')}")
        job.log(upload_info.get("reason") or "已上传")

    # ---- 1) 解析服务器路径 ----
    job.progress(1, 5, "解析服务器路径…")
    resolved = resolve_server_path(cfg, row)
    if not resolved.get("path"):
        return fail(f"无法定位服务器上的数据集：{resolved.get('detail')}")

    # ---- 2) 跑 doctor ----
    job.progress(2, 5, f"服务器上执行 lerobot-doctor：{resolved['path']}")
    result = run_doctor(cfg, resolved["path"], force=force, progress=job.log)

    if not result.get("ok") or not result.get("json"):
        return fail(f"{result.get('error') or '远端执行失败'}"
                    f"（路径 {resolved['path']}，退出码 {result.get('returncode')}）")

    data = result["json"]

    # ---- 3) 映射成 findings ----
    job.progress(3, 5, "映射检查结果…")
    findings = map_to_findings(data, cfg)

    # 用**平台阈值**重判 doctor 的测量值。阈值默认等于它自己的值，可在页面改；
    # 解析覆盖率一并返回，这样"解析不上"是可见的，不会静默丢数据。
    thresholds = cfg.get("doctor") or {}
    metric_info: dict = {"parsed": 0, "candidates": 0, "coverage": 1.0}
    rejudged: list[dict] = []
    disagreements: list[dict] = []
    if thresholds.get("rejudge", True):
        parsed = parse_doctor_metrics(data)
        metric_info = {k: parsed[k] for k in ("parsed", "candidates", "coverage", "unparsed")}
        metric_info["unparsed"] = metric_info["unparsed"][:8]
        rejudged = rejudge_doctor(parsed["metrics"], thresholds)
        for item in rejudged:
            item["check_version"] = f"doctor-{data.get('version') or '?'}"
            item["source"] = "server"
        disagreements = disagreement_notes(parsed["metrics"], thresholds, rejudged)
        for item in disagreements:
            item["check_version"] = f"doctor-{data.get('version') or '?'}"
            item["source"] = "server"
        findings.extend(rejudged)
        findings.extend(disagreements)
    summ = summary_of(data)
    cov = coverage_of(data)
    remote = {
        "ok": True, "host": cfg.get("host"), "path": resolved["path"],
        "path_source": resolved.get("source"),
        "cached": result.get("cached"), "seconds": result.get("seconds"),
        "returncode": result.get("returncode"), "version": data.get("version"),
        "overall_severity": summ["overall_severity"], "counts": summ["counts"],
        "total_episodes": summ["total_episodes"], "total_frames": summ["total_frames"],
        "fps": summ["fps"], "codebase_version": summ["codebase_version"],
        "checks": summ["checks"], "coverage": cov["categories"],
        "check_count": cov["checks"],
        "detail": (f"{summ['counts']['PASS']} PASS / {summ['counts']['WARN']} WARN / "
                   f"{summ['counts']['FAIL']} FAIL，映射出 {len(findings)} 条结论"
                   + (f"（其中平台阈值重判 {len(rejudged)} 条、分歧提示 {len(disagreements)} 条）"
                      if (rejudged or disagreements) else "")),
        "thresholds": {k: v for k, v in thresholds.items() if k != "unparsed"},
        "metric_parse": {"parsed": metric_info.get("parsed"),
                         "candidates": metric_info.get("candidates"),
                         "coverage": metric_info.get("coverage")},
        "rejudged": len(rejudged),
        "disagreements": len(disagreements),
        "unparsed_sample": metric_info.get("unparsed") or [],
    }
    if upload_info is not None:
        remote["upload"] = {
            "ok": upload_info.get("ok"), "skipped": upload_info.get("skipped"),
            "target": upload_info.get("target"), "seconds": upload_info.get("seconds"),
            "reason": upload_info.get("reason"),
            "local_bytes": (upload_info.get("status") or {}).get("local", {}).get("bytes"),
        }

    # ---- 4) 落库 + 报告（原始 JSON 一并存为可下载产物）----
    job.progress(4, 5, "写入结果…")
    raw_path = db.REPORT_DIR / f"qc_remote_{row['name']}_{run_id}_raw.json"
    try:
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        raw_path = None
    return finish(findings, remote,
                  {"doctor_json": str(raw_path) if raw_path else None,
                   "findings": len(findings)})
