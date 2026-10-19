"""远程训练后端骨架：通过 SSH/SCP 把训练任务发到服务器执行。

设计原则
--------
* 凭据只放 `storage/remote.json`（本机文件，不进版本库、不写进代码）
* 用 **paramiko**（自带加密实现，不受 Windows Schannel 限制影响）
* 每一步都可单独验证：连接 → 环境探测 → 数据同步 → 触发训练 → 日志回传 → 产物回收
* 训练在服务器上 nohup 后台跑，平台只轮询日志/完成标记 → 断线可恢复
"""
from __future__ import annotations

import json
import pathlib
import shlex
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "storage" / "remote.json"

DEFAULT_CONFIG = {
    "host": "192.168.123.105",
    "port": 22,
    "user": "",                  # 待填（注意别提交到版本库）
    "password": "",              # 待填；也支持 "key_file": "C:/path/id_rsa"
    "key_file": "",
    "remote_root": "/data/vla",              # 服务器上的工作根目录
    "remote_python": "python3",              # 训练用的解释器（venv/conda 里的）
    "remote_dataset_dir": "/data/vla/datasets",
    "remote_run_dir": "/data/vla/runs",
    "enabled": False,                        # 填好凭据 + 验证通过后改 true
}


def load_config(path: pathlib.Path | None = None) -> dict:
    p = path or CONFIG_PATH
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONFIG)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(json.loads(p.read_text(encoding="utf-8")) or {})
    return cfg


def save_config(cfg: dict, path: pathlib.Path | None = None) -> pathlib.Path:
    p = path or CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


class RemoteError(RuntimeError):
    pass


class Remote:
    """SSH 会话封装（paramiko）。用 with 保证关闭。"""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._client = None
        self._sftp = None

    # ---------- 连接 ----------
    def connect(self, timeout: int = 15):
        try:
            import paramiko
        except ImportError as exc:                 # pragma: no cover
            raise RemoteError(
                "缺少 paramiko（远程训练需要它）。安装：python tools/pip_less_install.py paramiko"
            ) from exc
        if not self.cfg.get("host") or not (self.cfg.get("user") or self.cfg.get("key_file")):
            raise RemoteError("remote.json 里的 host/user 未填")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = {"hostname": self.cfg["host"], "port": int(self.cfg.get("port") or 22),
                  "username": self.cfg.get("user") or None, "timeout": timeout,
                  "banner_timeout": timeout, "auth_timeout": timeout,
                  "look_for_keys": False, "allow_agent": False}
        if self.cfg.get("key_file"):
            kwargs["key_filename"] = self.cfg["key_file"]
        else:
            kwargs["password"] = self.cfg.get("password") or ""
        try:
            client.connect(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RemoteError(f"SSH 连接失败: {type(exc).__name__}: {exc}") from exc
        self._client = client
        return self

    def close(self):
        for obj in (self._sftp, self._client):
            try:
                if obj:
                    obj.close()
            except Exception:  # noqa: BLE001
                pass
        self._sftp = self._client = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---------- 执行 ----------
    def run(self, command: str, timeout: int = 300, check: bool = False) -> tuple[int, str, str]:
        if self._client is None:
            self.connect()
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        if check and code != 0:
            raise RemoteError(f"远程命令失败({code}): {command}\n{err[-500:]}")
        return code, out, err

    def run_detached(self, command: str, log_path: str) -> str:
        """后台跑（nohup），返回标记文件路径；平台后续轮询这个文件判断完成。"""
        marker = f"{log_path}.done"
        wrapped = (f"cd {shlex.quote(self.cfg['remote_root'])} && "
                   f"rm -f {shlex.quote(marker)} && "
                   f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(log_path)} 2>&1 ; "
                   f"echo $? > {shlex.quote(marker)} &")
        self.run(wrapped, timeout=60)
        return marker

    def tail(self, log_path: str, lines: int = 60) -> str:
        code, out, _ = self.run(f"tail -n {int(lines)} {shlex.quote(log_path)} 2>/dev/null || true")
        return out

    def done_code(self, marker: str) -> int | None:
        """完成返回退出码，未完成返回 None。"""
        code, out, _ = self.run(f"cat {shlex.quote(marker)} 2>/dev/null || true")
        out = out.strip()
        return int(out) if out.isdigit() else None

    # ---------- 文件 ----------
    def sftp(self):
        if self._sftp is None:
            if self._client is None:
                self.connect()
            self._sftp = self._client.open_sftp()
        return self._sftp

    def mkdirs(self, remote_dir: str):
        self.run(f"mkdir -p {shlex.quote(remote_dir)}", check=False)

    def upload_file(self, local: str | pathlib.Path, remote: str):
        self.sftp().put(str(local), remote)

    def upload_dir(self, local_dir: str | pathlib.Path, remote_dir: str, quiet: bool = True):
        """递归上传（SFTP 逐个文件；大数据集建议改用 tar+scp 或 rsync）。"""
        local_dir = pathlib.Path(local_dir)
        remote_dir = remote_dir.rstrip("/")
        sftp = self.sftp()
        made: set[str] = set()

        def ensure(rdir: str):
            if rdir in made or rdir in ("", "/"):
                return
            try:
                sftp.stat(rdir)
            except IOError:
                ensure(str(pathlib.PurePosixPath(rdir).parent))
                sftp.mkdir(rdir)
            made.add(rdir)

        ensure(remote_dir)
        count = 0
        for f in sorted(local_dir.rglob("*")):
            rel = f.relative_to(local_dir).as_posix()
            target = f"{remote_dir}/{rel}"
            if f.is_dir():
                ensure(target)
            else:
                ensure(str(pathlib.PurePosixPath(target).parent))
                sftp.put(str(f), target)
                count += 1
                if not quiet and count % 25 == 0:
                    print(f"      上传 {count} 个文件…")
        return count

    def download_file(self, remote: str, local: str | pathlib.Path):
        pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.sftp().get(remote, str(local))

    def download_dir(self, remote_dir: str, local_dir: str | pathlib.Path, quiet: bool = True):
        sftp = self.sftp()
        local_dir = pathlib.Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        def walk(rdir: str, ldir: pathlib.Path):
            nonlocal count
            for entry in sftp.listdir_attr(rdir):
                rpath = f"{rdir}/{entry.filename}"
                lpath = ldir / entry.filename
                if entry.st_mode & 0o040000:            # 目录
                    lpath.mkdir(exist_ok=True)
                    walk(rpath, lpath)
                else:
                    sftp.get(rpath, str(lpath))
                    count += 1
                    if not quiet and count % 10 == 0:
                        print(f"      下载 {count} 个文件…")

        walk(remote_dir.rstrip("/"), local_dir)
        return count


class RemoteCli:
    """不依赖任何第三方包的 SSH 后端：直接调系统的 ssh.exe / scp.exe / tar.exe。

    为什么需要它：本机 pip 被沙箱限制、PyPI 又常被网络抖动打断，paramiko 装不上；
    而 Windows 自带的 OpenSSH **不走 TLS/Schannel**（SSH 是独立协议），所以一定可用。
    认证用密钥（`ssh-keygen` 生成，公钥放进服务器 ~/.ssh/authorized_keys）。
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.ssh = shutil.which("ssh") or "ssh"
        self.scp = shutil.which("scp") or "scp"
        self.tar = shutil.which("tar") or "tar"

    # ---- 基础 ----
    def _base(self, batch: bool = True) -> list[str]:
        opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
                "-p", str(self.cfg.get("port") or 22)]
        if self.cfg.get("key_file"):
            opts += ["-i", self.cfg["key_file"], "-o", "IdentitiesOnly=yes"]
        if batch:
            opts += ["-o", "BatchMode=yes"]
        return opts

    def _target(self) -> str:
        user = self.cfg.get("user") or ""
        host = self.cfg["host"]
        return f"{user}@{host}" if user else host

    def run(self, command: str, timeout: int = 300, check: bool = False) -> tuple[int, str, str]:
        proc = subprocess.run([self.ssh, *self._base(), self._target(), command],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
        if check and proc.returncode != 0:
            raise RemoteError(f"远程命令失败({proc.returncode}): {command}\n{proc.stderr[-500:]}")
        return proc.returncode, proc.stdout, proc.stderr

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 文件（走 tar 流，避免逐个 SFTP 调用）----
    def mkdirs(self, remote_dir: str):
        self.run(f"mkdir -p {shlex.quote(remote_dir)}")

    def upload_dir(self, local_dir, remote_dir: str, quiet: bool = True) -> int:
        local_dir = pathlib.Path(local_dir)
        self.mkdirs(remote_dir)
        # 本机打包 -> ssh 解包
        archive = subprocess.run([self.tar, "-czf", "-", "-C", str(local_dir), "."],
                                 capture_output=True, check=True).stdout
        proc = subprocess.run([self.ssh, *self._base(), self._target(),
                               f"tar -xzf - -C {shlex.quote(remote_dir)}"],
                              input=archive, capture_output=False, timeout=1800)
        if proc.returncode != 0:
            raise RemoteError(f"上传失败（tar 管道退出码 {proc.returncode}）")
        _, out, _ = self.run(f"find {shlex.quote(remote_dir)} -type f | wc -l")
        try:
            return int(out.strip())
        except ValueError:
            return -1

    def download_dir(self, remote_dir: str, local_dir, quiet: bool = True) -> int:
        """远端 tar -czf - | 本机用 Python tarfile 解包。

        不能用 Windows 的 tar.exe 解包：它遇到中文文件名会报 "Invalid empty pathname"
        （bsdtar 管道模式下处理 UTF-8 的坑），Python tarfile 正常。
        """
        import io
        import tarfile

        local_dir = pathlib.Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([self.ssh, *self._base(), self._target(),
                               f"tar -czf - -C {shlex.quote(remote_dir)} ."],
                              capture_output=True, timeout=7200)
        if proc.returncode != 0:
            raise RemoteError(f"下载失败: {proc.stderr.decode('utf-8', 'replace')[-300:]}")
        with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:gz") as tf:
            tf.extractall(local_dir)
        return len([f for f in local_dir.rglob("*") if f.is_file()])

    def upload_file(self, local, remote: str):
        proc = subprocess.run([self.scp, "-P", str(self.cfg.get("port") or 22),
                               *(["-i", self.cfg["key_file"]] if self.cfg.get("key_file") else []),
                               "-o", "StrictHostKeyChecking=accept-new",
                               str(local), f"{self._target()}:{remote}"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RemoteError(f"scp 上传失败: {proc.stderr[-300:]}")

    def download_file(self, remote: str, local):
        pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([self.scp, "-P", str(self.cfg.get("port") or 22),
                               *(["-i", self.cfg["key_file"]] if self.cfg.get("key_file") else []),
                               f"{self._target()}:{remote}", str(local)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RemoteError(f"scp 下载失败: {proc.stderr[-300:]}")

    # ---- 后台任务 ----
    def run_detached(self, command: str, log_path: str) -> str:
        marker = f"{log_path}.done"
        # 必须把 stdin 也重定向、并用 setsid 脱离会话，否则 ssh 会一直等管道关闭（之前踩过）
        inner = (f"rm -f {shlex.quote(marker)}; "
                 f"setsid nohup bash -lc {shlex.quote(command)} "
                 f"< /dev/null > {shlex.quote(log_path)} 2>&1 & "
                 f"echo started")
        self.run(inner, timeout=30)
        return marker

    def tail(self, log_path: str, lines: int = 60) -> str:
        return self.run(f"tail -n {int(lines)} {shlex.quote(log_path)} 2>/dev/null || true")[1]

    def done_code(self, marker: str) -> int | None:
        out = self.run(f"cat {shlex.quote(marker)} 2>/dev/null || true")[1].strip()
        return int(out) if out.isdigit() else None


def make_remote(cfg: dict | None = None, prefer: str = "auto"):
    """优先用系统 ssh.exe（零依赖）；装了 paramiko 也可以选它。"""
    cfg = cfg or load_config()
    backend = (cfg.get("backend") or prefer or "auto")
    if backend == "paramiko":
        return Remote(cfg)
    if backend == "cli":
        return RemoteCli(cfg)
    try:
        import paramiko  # noqa: F401
        return Remote(cfg)
    except ImportError:
        return RemoteCli(cfg)


# ---------- 高层：环境探测 / 数据同步 / 训练 ----------
def probe(cfg: dict | None = None) -> dict:
    """探测服务器环境：OS / Python / GPU / 磁盘 / 已有框架。"""
    info: dict = {}
    with make_remote(cfg) as r:
        for name, cmd in {
            "uname": "uname -a",
            "os_release": "cat /etc/os-release 2>/dev/null | head -4",
            "python": f"{r.cfg['remote_python']} -V 2>&1 || python3 -V 2>&1",
            "pip": f"{r.cfg['remote_python']} -m pip -V 2>&1 | head -1",
            "nvidia_smi": "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1",
            "cuda": "nvcc --version 2>/dev/null | tail -2 || echo 'nvcc 无'",
            "disk": "df -h / /data 2>/dev/null | tail -3",
            "torch": f"{r.cfg['remote_python']} -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1",
            "conda_envs": "conda env list 2>/dev/null || echo '无 conda'",
            "outbound": "curl -sS -m 8 -o /dev/null -w '%{http_code}' https://hf-mirror.com 2>&1 || echo fail",
        }.items():
            try:
                code, out, err = r.run(cmd, timeout=30)
                info[name] = (out or err).strip()
            except Exception as exc:  # noqa: BLE001
                info[name] = f"<失败 {type(exc).__name__}: {exc}>"
    return info


def sync_dataset(local_dir: str | pathlib.Path, name: str, cfg: dict | None = None) -> str:
    """把本地 LeRobot v2.1 数据集目录同步到服务器，返回远程路径。"""
    cfg = cfg or load_config()
    remote_dir = f"{cfg['remote_dataset_dir'].rstrip('/')}/{name}"
    with make_remote(cfg) as r:
        r.mkdirs(remote_dir)
        n = r.upload_dir(local_dir, remote_dir)
        code, out, _ = r.run(f"find {shlex.quote(remote_dir)} -type f | wc -l")
    return f"{remote_dir}|{n}|{out.strip()}"


def start_remote_training(config_json: str, cfg: dict | None = None) -> dict:
    """在服务器上后台启动训练，返回日志/标记路径（平台轮询用）。"""
    cfg = cfg or load_config()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"{cfg['remote_run_dir'].rstrip('/')}/run_{stamp}"
    log_path = f"{run_dir}/train.log"
    with make_remote(cfg) as r:
        r.mkdirs(run_dir)
        tmp_cfg = ROOT / "storage" / "_remote_job.json"
        tmp_cfg.write_text(config_json, encoding="utf-8")
        r.upload_file(tmp_cfg, f"{run_dir}/job.json")
        cmd = (f"{cfg['remote_python']} -m xvla.train "
               f"--config {run_dir}/job.json --output_dir {run_dir}/ckpt")
        marker = r.run_detached(cmd, log_path)
    return {"run_dir": run_dir, "log": log_path, "marker": marker}
