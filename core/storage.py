"""存储清理：**先出计划，再执行**；只动可重建的中间产物，绝不碰数据集本体。

安全边界（写死在代码里，不靠调用方自觉）：

1. 只允许删除 `storage/` 下 `exports` / `reports` / `sessions` / `testdata` 里的路径；
2. **永远不删 `storage/datasets`**；
3. **不删任何被数据集登记引用的目录**（`datasets.path`）—— 那些目录就是数据集本体；
4. 导出目录只有在**同名 zip 已存在**时才算可回收（zip 是交付物，目录只是暂存副本）。

可回收的三类：

- `export_dir_with_zip`：`exports/<名>/` 且 `exports/<名>.zip` 存在 → 目录可重建
- `orphan_report`：`reports/` 里没有被 `qc_runs.report_path` / `artifacts.path` 引用的文件
- `stale_session_frames`：已提交/已中止会话的逐帧 PNG（mp4 已写进数据集，PNG 是中间产物）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import audit, db

CLEANABLE_DIRS = ("exports", "reports", "sessions", "testdata")


def _dir_bytes(path: Path) -> tuple[int, int]:
    files = [f for f in path.rglob("*") if f.is_file()] if path.is_dir() else []
    return len(files), sum(f.stat().st_size for f in files)


def _registered_paths() -> set[str]:
    """被登记为数据集本体的目录（绝不可删）。"""
    return {str(Path(r["path"]).resolve()) for r in db.query("SELECT path FROM datasets WHERE path IS NOT NULL")}


def _guarded(path: Path, registered: set[str]) -> bool:
    """True 表示允许删除该路径。"""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    storage = db.STORAGE.resolve()
    if not str(resolved).startswith(str(storage)):
        return False                                        # 必须在本平台 storage 下
    if str(resolved).startswith(str((db.STORAGE / "datasets").resolve())):
        return False                                        # 绝不碰数据集目录
    for registered_path in registered:
        if resolved == Path(registered_path) or str(resolved).startswith(registered_path.rstrip("\\/") + "\\") \
                or str(resolved).startswith(registered_path.rstrip("\\/") + "/"):
            return False                                    # 是/在某个数据集目录里
    return True


def scan() -> dict:
    """扫描并给出清理计划（**不删除任何东西**）。"""
    registered = _registered_paths()
    plan: list[dict] = []
    kept: list[dict] = []
    total_bytes = 0

    exports = db.STORAGE / "exports"
    if exports.is_dir():
        for item in sorted(exports.iterdir()):
            if not item.is_dir():
                continue
            files, size = _dir_bytes(item)
            total_bytes += size
            zip_path = item.with_suffix(".zip")
            reason = None
            if not _guarded(item, registered):
                reason = "已被数据集登记引用（就是数据集本体），不可删"
            elif not zip_path.is_file():
                reason = f"没有同名 zip（{zip_path.name}），删了就没了唯一副本"
            if reason:
                kept.append({"kind": "export_dir", "path": str(item), "files": files,
                             "bytes": size, "reason": reason})
            else:
                plan.append({"kind": "export_dir_with_zip", "path": str(item), "files": files,
                             "bytes": size, "safe": True,
                             "reason": f"同名 zip {zip_path.name} 已存在，目录可重建"})

    reports = db.STORAGE / "reports"
    if reports.is_dir():
        referenced = {str(Path(r["report_path"]).resolve())
                      for r in db.query("SELECT report_path FROM qc_runs WHERE report_path IS NOT NULL")}
        referenced |= {str(Path(r["path"]).resolve())
                       for r in db.query("SELECT path FROM artifacts WHERE path IS NOT NULL")}
        for file in sorted(reports.rglob("*")):
            if not file.is_file():
                continue
            if not _guarded(file, registered):
                continue
            if str(file.resolve()) in referenced:
                continue
            size = file.stat().st_size
            plan.append({"kind": "orphan_report", "path": str(file), "files": 1, "bytes": size,
                         "safe": True, "reason": "没有被任何质检记录或产物引用"})

    sessions = db.STORAGE / "sessions"
    if sessions.is_dir():
        done = {r["path"] for r in db.query(
            "SELECT path FROM collect_sessions WHERE status IN ('committed','aborted')"
            " AND path IS NOT NULL")}
        for session_dir in sorted(sessions.iterdir()):
            if str(session_dir) not in done:
                continue
            images = session_dir / "images"
            files, size = _dir_bytes(images)
            if files and _guarded(images, registered):
                plan.append({"kind": "stale_session_frames", "path": str(images), "files": files,
                             "bytes": size, "safe": True,
                             "reason": "会话已提交/中止，mp4 已写进数据集，逐帧 PNG 是中间产物"})

    by_kind: dict[str, dict] = {}
    for item in plan:
        slot = by_kind.setdefault(item["kind"], {"kind": item["kind"], "items": 0, "files": 0, "bytes": 0})
        slot["items"] += 1
        slot["files"] += item["files"]
        slot["bytes"] += item["bytes"]

    return {
        "storage": str(db.STORAGE),
        "scanned_bytes": total_bytes,
        "reclaimable_bytes": sum(item["bytes"] for item in plan),
        "reclaimable_files": sum(item["files"] for item in plan),
        "plan": plan,
        "plan_summary": sorted(by_kind.values(), key=lambda x: -x["bytes"]),
        "kept": kept,
        "note": "计划与执行分离：这里只算账，点「执行清理」才会删除；删除始终受同样三条边界保护。",
    }


def apply(kinds: list[str] | None = None, paths: list[str] | None = None,
          dry_run: bool = False) -> dict:
    """执行清理。kinds 限定类别、paths 限定具体路径；dry_run=True 只回显不删。"""
    plan = scan()["plan"]
    selected = []
    for item in plan:
        if paths and item["path"] not in paths:
            continue
        if kinds and item["kind"] not in kinds:
            continue
        selected.append(item)

    registered = _registered_paths()
    deleted_files = 0
    freed = 0
    errors: list[str] = []
    done: list[str] = []
    for item in selected:
        target = Path(item["path"])
        if not _guarded(target, registered):                # 执行时再查一次，防 TOCTOU
            errors.append(f"拒绝删除（越过安全边界）：{target}")
            continue
        if dry_run:
            done.append(str(target))
            deleted_files += item["files"]
            freed += item["bytes"]
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            done.append(str(target))
            deleted_files += item["files"]
            freed += item["bytes"]
        except OSError as exc:
            errors.append(f"{target}: {type(exc).__name__}: {exc}")

    if not dry_run and (deleted_files or errors):
        audit.log("WARNING" if errors else "INFO", "系统",
                  f"存储清理{'（预览）' if dry_run else ''}：删除 {deleted_files} 个文件 / "
                  f"释放 {freed / 1024 / 1024:.1f} MB"
                  + (f"，{len(errors)} 个失败" if errors else ""),
                  detail={"kinds": kinds, "paths": len(done), "freed_bytes": freed,
                          "errors": errors[:5]})
    return {"dry_run": dry_run, "targets": len(done), "deleted_files": deleted_files,
            "freed_bytes": freed, "freed_mb": round(freed / 1024 / 1024, 2),
            "done": done if dry_run else done[:20], "errors": errors,
            "remaining": scan()["reclaimable_bytes"] if not dry_run else None}


DEFAULT_KINDS = ["export_dir_with_zip", "stale_session_frames", "orphan_report"]


def policy() -> dict:
    """读存储策略（单行表，第一次访问时创建默认值：**关闭**）。"""
    row = db.query_one("SELECT * FROM storage_policy WHERE id=1")
    if not row:
        db.execute("INSERT INTO storage_policy (id, enabled, threshold_mb, kinds_json, interval_s)"
                   " VALUES (1, 0, 200, ?, 300)", (json.dumps(DEFAULT_KINDS),))
        row = db.query_one("SELECT * FROM storage_policy WHERE id=1")
    try:
        kinds = json.loads(row["kinds_json"] or "[]") or list(DEFAULT_KINDS)
    except Exception:  # noqa: BLE001
        kinds = list(DEFAULT_KINDS)
    return {"enabled": bool(row["enabled"]), "threshold_mb": row["threshold_mb"] or 0,
            "kinds": kinds, "interval_s": int(row["interval_s"] or 300),
            "last_check": row["last_check"], "last_action": row["last_action"],
            "last_result": json.loads(row["last_result"]) if row["last_result"] else None}


def set_policy(*, enabled: bool | None = None, threshold_mb: float | None = None,
               kinds: list[str] | None = None, interval_s: int | None = None) -> dict:
    policy()
    if enabled is not None:
        db.execute("UPDATE storage_policy SET enabled=? WHERE id=1", (1 if enabled else 0,))
    if threshold_mb is not None:
        db.execute("UPDATE storage_policy SET threshold_mb=? WHERE id=1", (float(threshold_mb),))
    if kinds is not None:
        unknown = [k for k in kinds if k not in _PLAN_KINDS]
        if unknown:
            raise ValueError(f"未知的清理类别：{unknown}（可用 {_PLAN_KINDS}）")
        db.execute("UPDATE storage_policy SET kinds_json=? WHERE id=1", (json.dumps(kinds),))
    if interval_s is not None:
        db.execute("UPDATE storage_policy SET interval_s=? WHERE id=1",
                   (max(30, int(interval_s)),))
    result = policy()
    audit.log("WARNING" if result["enabled"] else "INFO", "存储",
              f"更新存储策略：自动清理 {'开启' if result['enabled'] else '关闭'}，"
              f"阈值 {result['threshold_mb']}MB，类别 {result['kinds']}，"
              f"间隔 {result['interval_s']}s",
              detail=result)
    return result


_PLAN_KINDS = ("export_dir_with_zip", "orphan_report", "stale_session_frames")


def auto_check() -> dict:
    """调度线程调用：按策略决定是否执行清理。**默认关闭时只记账不动手。**"""
    plan = scan()
    reclaimable_mb = plan["reclaimable_bytes"] / 1024 / 1024
    policy_row = policy()
    decision = "disabled"
    result = None
    if policy_row["enabled"]:
        if reclaimable_mb < policy_row["threshold_mb"]:
            decision = "below_threshold"
        else:
            decision = "cleaned"
            result = apply(kinds=policy_row["kinds"])
    db.execute("UPDATE storage_policy SET last_check=?, last_action=?, last_result=? WHERE id=1",
               (db.now(), decision,
                json.dumps({"reclaimable_mb": round(reclaimable_mb, 2), "decision": decision,
                            "freed_mb": (result or {}).get("freed_mb"),
                            "deleted_files": (result or {}).get("deleted_files")},
                           ensure_ascii=False)))
    if decision == "cleaned":
        audit.log("WARNING", "存储",
                  f"自动清理已执行：释放 {(result or {}).get('freed_mb')}MB / "
                  f"{(result or {}).get('deleted_files')} 个文件（阈值 "
                  f"{policy_row['threshold_mb']}MB）", detail=result)
    return {"checked_at": db.now(), "reclaimable_mb": round(reclaimable_mb, 2),
            "decision": decision, "policy": policy_row, "result": result}
