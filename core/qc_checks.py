"""数据基础格式质检（分类扩展）：把质检拆成「12 项」维度。

与 `core/lerobot.py` 的分工：
- `lerobot.check_episode()` 是**格式契约层**（原来那 13 条），保持不动，零回归风险；
- 本模块是**分类扩展层**，补上原来完全没有的维度：统计分布、异常检测、可移植性、
  动作质量（突变/空闲/饱和）、元数据 schema 与自洽、黑帧/冻结/曝光、健康度评分。

设计照着 HFlow 的三条原则来（不管最终用不用 HFlow，这三条都该有）：

1. **证据 ≠ 判决**：所有阈值从 `storage/qc_config.json` 读，不写死在代码里；
   检查跑过就记录**测量值**，只有越过阈值才产出 finding。
2. **check 版本**：每条规则带 `check_version`，阈值或算法变了就 +1，
   这样 curation 能钉住"哪一代结果可比"。
3. **覆盖率分母**：返回 `checks_run`，报告里能说清"哪些检查跑在了多少比例的 episode 上"——
   覆盖一半的统计量不能看起来像覆盖全部。

并且明确：**任何规则都不删数据、不自行打回**。硬错误（error）只留给"训练一定会出错"的
契约破坏（元数据自洽、统计量不一致、schema 缺失）。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import db

CONFIG_PATH = db.STORAGE / "qc_config.json"

# 检查版本：改动某条规则的算法或默认阈值时把它 +1
CHECK_VERSION = "1"

#: 10 个分类（对应用户要求的「12 项」，其余 2 项待补）
CATEGORIES = [
    ("format", "格式契约", "原有 13 条规则（lerobot.check_episode）"),
    ("metadata", "元数据", "info.json 必需字段、元数据自洽"),
    ("temporal", "时序", "单调性/步长/重复戳/跨相机同步"),
    ("action", "动作质量", "突变、空闲、饱和、NaN"),
    ("video", "视频完整性", "帧数一致、可解码、黑帧、曝光、冻结"),
    ("stats", "统计分布", "归一化统计与真实数据是否一致、跨 episode 离群"),
    ("health", "episode 健康度", "综合评分（仅用于分诊，不预测训练效果）"),
    ("readiness", "训练就绪性", "视觉输入、语言指令、归一化统计齐备"),
    ("anomaly", "异常检测", "统计离群（开放集异常交给 VLM）"),
    ("portability", "可移植性", "绝对路径、文件名、编码"),
    ("kinematics", "运动学", "关节限位（需外部限位表）"),
]
CATEGORY_LABELS = {key: label for key, label, _desc in CATEGORIES}

#: 原有 13 条规则归到哪一类（用于页面按维度展示）
LEGACY_RULE_CATEGORY = {
    "文件完整性": "video",
    "视频帧数一致性": "video",
    "视频可解码性": "video",
    "帧率一致性": "video",
    "空 episode": "metadata",
    "长度一致性": "metadata",
    "特征维度一致性": "metadata",
    "视觉输入完整性": "readiness",
    "语言指令完整性": "readiness",
    "数值有效性": "action",
    "时间戳单调性": "temporal",
    "时间戳步长": "temporal",
    "动作有效性": "action",
}


DEFAULTS: dict = {
    "version": 1,
    # ---- 分类开关 ----
    "enabled_categories": {key: True for key, _l, _d in CATEGORIES},
    # ---- 元数据 ----
    "metadata": {
        "required_info_fields": ["codebase_version", "fps", "total_episodes", "total_frames",
                                 "chunks_size", "data_path", "features"],
        "check_self_consistency": True,
    },
    # ---- 时序 ----
    "temporal": {
        "duplicate_ts_tol": 1e-9,
        "cross_camera_fps_tol": 0.1,
        "cross_camera_frame_tol_ratio": 0.02,
    },
    # ---- 动作质量 ----
    "action": {
        # 突变判据：单帧动作变化 > max(整段动作幅度 × range_ratio, p95 变化 × factor)
        # 用「相对整段幅度」而不是「中位 + k×MAD」——遥操动作是零膨胀分布，后者会退化
        "spike_range_ratio": 0.10,
        "spike_quantile_factor": 2.0,
        "spike_unreliable_ratio": 0.5,   # 超过一半帧都超阈值 → 该指标不适用，降级为 info
        "idle_eps": 1e-4,             # 判定为"没动"的动作变化阈值
        "idle_ratio_warn": 0.85,      # 空闲帧占比超过它 → warning
        "saturation_ratio_warn": 0.5,  # 贴限位的占比（**需要 kinematics.limit_table**）
    },
    # ---- 视频 ----
    "video": {
        "sample_frames": 12,
        "dark_mean_warn": 12.0,       # 亮度均值低于它算黑帧
        "dark_ratio_warn": 0.5,       # 黑帧占比超过它 → warning
        "bright_mean_warn": 245.0,    # 过曝
        "bright_ratio_warn": 0.5,
        "freeze_ratio_warn": 0.5,     # 完全相同帧的占比
        "decode_checks": True,        # 关掉可省时间（只做帧数/可解码）
    },
    # ---- 统计分布 ----
    "stats": {
        "compare_episodes_stats": True,
        "mean_rel_tol": 0.05,         # 相对容差 5%
        "std_rel_tol": 0.05,
        "minmax_rel_tol": 0.05,
        "max_findings_per_episode": 3,  # 避免一个坏 episode 刷出几十条
        "outlier_mad_k": 6.0,
        "outlier_min_episodes": 8,    # episode 少于它就只出 info「样本太少」
    },
    # ---- 可移植性 ----
    "portability": {
        "check_absolute_paths": True,
        "check_filename_ascii": True,
    },
    # ---- 健康度评分权重（负向扣分）----
    "health": {
        "error_penalty": 25.0,
        "warning_penalty": 6.0,
        "info_penalty": 0.0,
        "grades": [[90, "A"], [80, "B"], [70, "C"], [60, "D"], [0, "F"]],
        "score_warn_below": 70,
    },
    # ---- 运动学（需要用户提供限位表；没有就只出 info）----
    "kinematics": {
        "limit_table": {},            # {"action": [[lo,hi], ...]} 或 {"observation.state": ...}
    },
}


# --------------------------------------------------------------------------- 配置


def load_config(overrides: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if CONFIG_PATH.is_file():
        try:
            patch = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(patch, dict):
                for key, value in patch.items():
                    if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                        cfg[key].update(value)
                    else:
                        cfg[key] = value
        except Exception:  # noqa: BLE001
            pass
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def save_config(patch: dict) -> dict:
    current: dict = {}
    if CONFIG_PATH.is_file():
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            current = {}
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_config()


def config_fingerprint(cfg: dict) -> str:
    """阈值指纹：写进质检 run，用来判断两次质检的结论可比性。"""
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- 工具


def _finding(category: str, rule: str, severity: str, detail: str,
             measured=None, expected=None) -> dict:
    return {"rule": rule, "severity": severity, "detail": detail,
            "measured": None if measured is None else str(measured),
            "expected": None if expected is None else str(expected),
            "category": category, "check_version": CHECK_VERSION}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _mad(values: list[float]) -> float:
    """中位数绝对偏差 × 1.4826（正态下等价于标准差）。"""
    if len(values) < 2:
        return 0.0
    med = _median(values)
    return 1.4826 * _median([abs(v - med) for v in values])


def _rel_diff(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale


def _series(table: dict, key: str) -> list[list[float]]:
    """把 parquet 里某列取成 [[...], [...]]；单值列也统一成 [ [v], ... ]。"""
    out = []
    for row in table.get(key) or []:
        if isinstance(row, (list, tuple)):
            out.append([float(v) for v in row if v is not None])
        elif row is None:
            out.append([])
        else:
            out.append([float(row)])
    return out


def _flatten(series: list[list[float]]) -> list[float]:
    return [v for row in series for v in row]


# --------------------------------------------------------------------------- 元数据


def check_metadata(ds_path, meta: dict, index: int, record: dict, cfg: dict) -> tuple[list, dict]:
    findings: list[dict] = []
    info = meta.get("info") or {}
    mc = cfg.get("metadata") or {}

    # 1) schema：必需字段
    missing = [f for f in (mc.get("required_info_fields") or []) if info.get(f) in (None, "", [])]
    if missing:
        findings.append(_finding("metadata", "元数据 schema", "error",
                                 f"info.json 缺少必需字段：{missing}", measured=len(missing),
                                 expected="全部存在"))

    features = info.get("features") or {}
    bad_feature = []
    for key, spec in features.items():
        if not isinstance(spec, dict) or not spec.get("dtype"):
            bad_feature.append(key)
        elif spec.get("dtype") == "video" and not (spec.get("video_info") or {}).get("video.fps"):
            bad_feature.append(f"{key}(缺 video_info.fps)")
    if bad_feature:
        findings.append(_finding("metadata", "特征声明完整性", "warning",
                                 f"特征声明不完整：{bad_feature}", measured=len(bad_feature),
                                 expected="每项有 dtype；video 有 video_info.fps"))

    # 2) 自洽：Σ length vs total_frames，episode 数 vs total_episodes
    if mc.get("check_self_consistency"):
        episodes = meta.get("episodes") or {}
        declared_total = info.get("total_frames")
        if episodes and declared_total:
            summed = sum((e or {}).get("length") or 0 for e in episodes.values())
            if summed and int(summed) != int(declared_total):
                findings.append(_finding(
                    "metadata", "元数据自洽", "error",
                    "Σ episodes.jsonl 的 length 与 info.json 的 total_frames 对不上"
                    "——在读任何数据文件之前元数据就自相矛盾",
                    measured=f"Σlength={summed}", expected=f"total_frames={declared_total}"))
        declared_eps = info.get("total_episodes")
        if declared_eps and episodes and len(episodes) != int(declared_eps):
            findings.append(_finding(
                "metadata", "元数据自洽", "error",
                "episodes.jsonl 的条数与 info.json 的 total_episodes 对不上",
                measured=f"{len(episodes)} 条", expected=f"total_episodes={declared_eps}"))

    measurements = {"total_frames_declared": info.get("total_frames"),
                    "total_episodes_declared": info.get("total_episodes"),
                    "feature_count": len(features)}
    return findings, measurements


# --------------------------------------------------------------------------- 时序


def check_temporal(ds_path, meta: dict, index: int, record: dict, table: dict,
                   cfg: dict) -> tuple[list, dict]:
    findings: list[dict] = []
    tc = cfg.get("temporal") or {}
    timestamps = [float(v) for v in (table.get("timestamp") or []) if v is not None]
    measurements: dict = {"frames": len(timestamps)}

    if timestamps:
        tol = float(tc.get("duplicate_ts_tol") or 1e-9)
        duplicates = sum(1 for i in range(1, len(timestamps))
                         if abs(timestamps[i] - timestamps[i - 1]) <= tol)
        measurements["duplicate_timestamps"] = duplicates
        if duplicates:
            findings.append(_finding(
                "temporal", "时间戳重复", "warning",
                f"存在 {duplicates} 个与前一帧相同的时间戳（采样时钟异常或丢帧重写）",
                measured=duplicates, expected="0"))
        measurements["timestamp_start"] = timestamps[0]
        measurements["timestamp_end"] = timestamps[-1]

    # 跨相机一致性（多相机多模态同步）
    videos = record.get("videos") or {}
    if len(videos) > 1:
        fps_values = [v.get("fps") for v in videos.values() if v.get("fps")]
        frame_values = [v.get("frames") for v in videos.values() if v.get("frames")]
        if len(set(fps_values)) > 1:
            spread = max(fps_values) - min(fps_values)
            if spread > float(tc.get("cross_camera_fps_tol") or 0.1):
                findings.append(_finding(
                    "temporal", "跨相机帧率不一致", "warning",
                    "多路相机的 fps 不一致，多模态同步会漂移",
                    measured=f"{sorted(set(fps_values))}", expected="各相机 fps 相同"))
        if len(frame_values) > 1:
            lo, hi = min(frame_values), max(frame_values)
            ratio = (hi - lo) / max(hi, 1)
            measurements["camera_frame_spread_ratio"] = round(ratio, 4)
            if ratio > float(tc.get("cross_camera_frame_tol_ratio") or 0.02):
                findings.append(_finding(
                    "temporal", "跨相机帧数不一致", "warning",
                    "多路相机的帧数差异超过 2%，同一时刻的取帧会错位",
                    measured=f"{frame_values}（差 {ratio:.1%}）", expected="帧数一致"))
    return findings, measurements


# --------------------------------------------------------------------------- 动作质量


def check_action(ds_path, meta: dict, index: int, record: dict, table: dict,
                 cfg: dict) -> tuple[list, dict]:
    findings: list[dict] = []
    ac = cfg.get("action") or {}
    actions = _series(table, "action")
    measurements: dict = {"action_frames": len(actions)}
    if len(actions) < 3:
        return findings, measurements

    # ---- 动作突变：**相对整段动作幅度**判定，而不是拿中位数当基线 ----
    # 教训（实测 zhiyan 真机数据）：遥操动作是"零膨胀"分布 —— 57% 的帧动作完全不变，
    # 中位数和 MAD 都是 0，用 median + k*MAD 会退化成兜底阈值，把正常运动全报成突变
    # （实测误报 131 处/399 帧，而真实最大单帧变化只占整段幅度的 5.2%）。
    steps = []
    for i in range(1, len(actions)):
        a, b = actions[i - 1], actions[i]
        n = min(len(a), len(b))
        if n == 0:
            continue
        steps.append(math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(n))))
    if steps:
        dims = max(len(r) for r in actions)
        total_range = math.sqrt(sum(
            (max(r[j] for r in actions if len(r) > j) - min(r[j] for r in actions if len(r) > j)) ** 2
            for j in range(dims) if any(len(r) > j for r in actions)))
        ordered = sorted(steps)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        range_ratio = float(ac.get("spike_range_ratio") or 0.10)
        factor = float(ac.get("spike_quantile_factor") or 2.0)
        threshold = max(total_range * range_ratio, p95 * factor, 1e-9)
        spikes = sum(1 for s in steps if s > threshold)
        ratio = spikes / len(steps)
        measurements.update({"action_step_median": round(_median(steps), 5),
                             "action_step_p95": round(p95, 5),
                             "action_step_max": round(max(steps), 5),
                             "action_total_range": round(total_range, 4),
                             "action_spikes": spikes,
                             "action_spike_ratio": round(ratio, 4)})
        unreliable = ratio > float(ac.get("spike_unreliable_ratio") or 0.5)
        if spikes and unreliable:
            # 一半以上的帧都"超阈值"，说明该指标在这个数据集上不适用，别硬报
            findings.append(_finding(
                "action", "动作突变（不连续）", "info",
                f"有 {ratio:.0%} 的帧动作变化超过阈值，说明该数据的动作变化过于集中，"
                f"本方法无法可靠判断突变（已降级为提示，不作为质量问题）",
                measured=f"{spikes} 处 / {len(steps)} 帧",
                expected="该指标不适用于本数据集，建议按限位表判断"))
        elif spikes:
            findings.append(_finding(
                "action", "动作突变（不连续）", "warning",
                f"存在 {spikes} 处相邻帧动作跳变超过整段动作幅度的 {range_ratio:.0%}"
                f"（可能撞限位、遥操抖动或数据错帧）",
                measured=f"{spikes} 处（最大单帧 {max(steps):.3f}，"
                         f"占整段幅度 {max(steps)/total_range:.1%}）",
                expected=f"单帧变化 ≤ 整段幅度的 {range_ratio:.0%}（阈值 {threshold:.3f}）"))

        # 空闲比例
        eps = float(ac.get("idle_eps") or 1e-4)
        idle = sum(1 for s in steps if s <= eps)
        idle_ratio = idle / len(steps)
        measurements["idle_ratio"] = round(idle_ratio, 4)
        if idle_ratio > float(ac.get("idle_ratio_warn") or 0.85):
            findings.append(_finding(
                "action", "空闲比例过高", "warning",
                f"{idle_ratio:.0%} 的帧动作几乎没变化（大量静止 = 低信息量示范）",
                measured=f"{idle_ratio:.1%}", expected=f"< {(ac.get('idle_ratio_warn') or 0.85):.0%}"))

    # ---- 动作饱和：**必须有量程（限位表）才有意义** ----
    # 用"贴本 episode 极值"当饱和是错的：任何在某个静止位停留的示范都会中招
    # （实测 zhiyan ep5 报 97.2%，其实只是长时间停在同一位姿，那是 idle 不是 saturation）。
    limits = (cfg.get("kinematics") or {}).get("limit_table") or {}
    action_limits = limits.get("action") if isinstance(limits, dict) else None
    if action_limits:
        sat = total = 0
        for row in actions:
            for j, value in enumerate(row):
                if j >= len(action_limits):
                    break
                bounds = action_limits[j]
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    span = float(bounds[1]) - float(bounds[0])
                    if span <= 0:
                        continue
                    total += 1
                    if (value <= float(bounds[0]) + 0.01 * span
                            or value >= float(bounds[1]) - 0.01 * span):
                        sat += 1
        if total:
            ratio = sat / total
            measurements["action_saturation_ratio"] = round(ratio, 4)
            if ratio > float(ac.get("saturation_ratio_warn") or 0.5):
                findings.append(_finding(
                    "action", "动作饱和", "warning",
                    f"{ratio:.0%} 的动作值贴在实际关节限位上（指令被裁剪或顶到限位）",
                    measured=f"{ratio:.1%}",
                    expected=f"< {(ac.get('saturation_ratio_warn') or 0.5):.0%}"))
    return findings, measurements


# --------------------------------------------------------------------------- 视频（黑帧/曝光/冻结）


def check_video_decode(ds_path, meta: dict, index: int, record: dict,
                       cfg: dict) -> tuple[list, dict]:
    findings: list[dict] = []
    vc = cfg.get("video") or {}
    measurements: dict = {}
    if not vc.get("decode_checks", True):
        return findings, measurements

    import cv2
    import numpy as np

    for key, entry in (record.get("videos") or {}).items():
        if not entry.get("exists") or not entry.get("opened"):
            continue
        cap = cv2.VideoCapture(str(entry.get("path")))
        if not cap.isOpened():
            cap.release()
            continue
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            want = max(2, int(vc.get("sample_frames") or 12))
            if total <= 0:
                continue
            idxs = sorted({int(round(i * (total - 1) / max(1, want - 1))) for i in range(want)})
            means, digests = [], []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                means.append(float(np.mean(gray)))
                digests.append(hashlib.sha256(np.ascontiguousarray(gray)).hexdigest()[:16])
        finally:
            cap.release()

        if not means:
            continue
        dark = sum(1 for m in means if m < float(vc.get("dark_mean_warn") or 12.0))
        bright = sum(1 for m in means if m > float(vc.get("bright_mean_warn") or 245.0))
        dup = len(digests) - len(set(digests))
        dark_ratio = dark / len(means)
        bright_ratio = bright / len(means)
        freeze_ratio = dup / max(1, len(digests) - 1) if len(digests) > 1 else 0.0
        measurements[f"{key}/mean_brightness"] = round(float(sum(means) / len(means)), 2)
        measurements[f"{key}/dark_ratio"] = round(dark_ratio, 3)
        measurements[f"{key}/freeze_ratio"] = round(freeze_ratio, 3)

        if dark_ratio > float(vc.get("dark_ratio_warn") or 0.5):
            findings.append(_finding(
                "video", "画面过暗/黑帧", "warning",
                f"{key}：{dark_ratio:.0%} 的采样帧接近全黑（镜头被挡或补光失效）",
                measured=f"{dark_ratio:.1%}（均值 {sum(means)/len(means):.1f}）",
                expected=f"< {(vc.get('dark_ratio_warn') or 0.5):.0%}"))
        if bright_ratio > float(vc.get("bright_ratio_warn") or 0.5):
            findings.append(_finding(
                "video", "画面过曝", "warning",
                f"{key}：{bright_ratio:.0%} 的采样帧接近全白（曝光过高）",
                measured=f"{bright_ratio:.1%}", expected=f"< {(vc.get('bright_ratio_warn') or 0.5):.0%}"))
        if freeze_ratio > float(vc.get("freeze_ratio_warn") or 0.5):
            findings.append(_finding(
                "video", "视频冻结", "warning",
                f"{key}：{dup} 组采样帧完全相同（相机卡死或录制复用同一帧）",
                measured=f"冻结比例 {freeze_ratio:.1%}", expected="< 50%"))
    return findings, measurements


# --------------------------------------------------------------------------- 统计分布


_STATS_CACHE: dict[str, dict] = {}


def _load_episodes_stats(ds_path) -> dict:
    """读 `meta/episodes_stats.jsonl`。

    **不能依赖 `lerobot.load_meta`** —— 它只返回 {info, tasks, episodes}，
    没有 episodes_stats（实测），会导致本检查永远不执行。
    """
    key = str(ds_path)
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]
    out: dict = {}
    path = Path(ds_path) / "meta" / "episodes_stats.jsonl"
    if path.is_file():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out[int(row.get("episode_index"))] = row
        except Exception:  # noqa: BLE001
            out = {}
    # v3 的 meta/stats.json（全局统计）也兼容一下
    if not out:
        alt = Path(ds_path) / "meta" / "stats.json"
        if alt.is_file():
            try:
                payload = json.loads(alt.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    for index, stat in payload.items():
                        if str(index).isdigit() and isinstance(stat, dict):
                            out[int(index)] = {"episode_index": int(index), "stats": stat}
            except Exception:  # noqa: BLE001
                pass
    _STATS_CACHE[key] = out
    return out


def _episode_stats(meta: dict, index: int, ds_path=None) -> dict:
    """取该 episode 的统计块：优先 meta 里带的，其次自己读文件。"""
    entry = (meta.get("episodes_stats") or {}).get(index)
    if not entry and ds_path is not None:
        entry = _load_episodes_stats(ds_path).get(index)
    if not entry:
        return {}
    return entry.get("stats") or entry


def check_stats(ds_path, meta: dict, index: int, record: dict, table: dict,
                cfg: dict) -> tuple[list, dict]:
    """归一化统计一致性：`episodes_stats.jsonl` 存的统计与实时重算比对。

    这是最值钱的一条：**训练用的就是这些数做归一化**，它们错了
    （HFlow/robovet 叫 "normalization poison"）训练会静默跑偏。
    """
    findings: list[dict] = []
    sc = cfg.get("stats") or {}
    measurements: dict = {}
    if not sc.get("compare_episodes_stats", True):
        return findings, measurements

    stored = _episode_stats(meta, index, ds_path)
    if not stored:
        measurements["stats_available"] = False
        return findings, measurements
    measurements["stats_available"] = True

    mean_tol = float(sc.get("mean_rel_tol") or 0.05)
    std_tol = float(sc.get("std_rel_tol") or 0.05)
    mm_tol = float(sc.get("minmax_rel_tol") or 0.05)
    max_findings = int(sc.get("max_findings_per_episode") or 3)

    for feature, stat in stored.items():
        if not isinstance(stat, dict):
            continue
        series = _series(table, feature)
        if not series:
            continue
        dims = max(len(r) for r in series)
        # 逐维重算
        for j in range(dims):
            col = [r[j] for r in series if len(r) > j]
            if not col:
                continue
            real_mean = sum(col) / len(col)
            real_min, real_max = min(col), max(col)
            var = sum((v - real_mean) ** 2 for v in col) / max(1, len(col) - 1)
            real_std = math.sqrt(var)

            def get(key):
                vals = stat.get(key)
                if isinstance(vals, (list, tuple)) and len(vals) > j:
                    try:
                        return float(vals[j])
                    except (TypeError, ValueError):
                        return None
                if isinstance(vals, (int, float)) and j == 0:
                    return float(vals)
                return None

            for label, real, tol in (("mean", real_mean, mean_tol),
                                     ("std", real_std, std_tol),
                                     ("min", real_min, mm_tol),
                                     ("max", real_max, mm_tol)):
                claimed = get(label)
                if claimed is None or not math.isfinite(claimed):
                    continue
                diff = _rel_diff(real, claimed)
                if diff > tol:
                    findings.append(_finding(
                        "stats", "归一化统计不一致", "error",
                        f"{feature}[{j}] 的 {label} 与真实数据不符（相对偏差 {diff:.1%}）"
                        f"——训练会用这个数做归一化，必须修（robovet 称 normalization poison）",
                        measured=f"记录值 {claimed:.6g} vs 实际 {real:.6g}",
                        expected=f"相对偏差 < {tol:.0%}"))
                    break   # 每维只报一条，避免刷屏
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break
    return findings, measurements


def dataset_level_stats(per_episode: list[dict], cfg: dict) -> dict[int, list[dict]]:
    """数据集级：用 MAD 找离群 episode（时长/帧数/空闲比例/动作幅度）。

    MAD 自适应每个数据集，不用硬编码 3σ（RDA 的做法）。
    """
    out: dict[int, list[dict]] = {p["episode_index"]: [] for p in per_episode}
    # 分类开关要在这里也生效，否则关掉 anomaly 仍然会产出离群结论
    if not (cfg.get("enabled_categories") or {}).get("anomaly", True):
        return out

    sc = cfg.get("stats") or {}
    min_n = int(sc.get("outlier_min_episodes") or 8)
    k = float(sc.get("outlier_mad_k") or 6.0)
    metrics = ["frames", "duration_s", "idle_ratio", "action_step_median"]

    if len(per_episode) < min_n:
        for p in per_episode:
            out[p["episode_index"]].append(_finding(
                "anomaly", "跨 episode 离群", "info",
                f"只有 {len(per_episode)} 条 episode，少于 {min_n} 条，不做离群判断"
                "（样本太少时 MAD 会把正常波动报成异常）",
                measured=len(per_episode), expected=f"≥ {min_n} 条"))
        return out

    labels = {"frames": "帧数", "duration_s": "时长", "idle_ratio": "空闲比例",
              "action_step_median": "动作步长"}
    for metric in metrics:
        values = []
        for p in per_episode:
            v = (p.get("measurements") or {}).get(metric)
            if isinstance(v, (int, float)) and math.isfinite(v):
                values.append(float(v))
        if len(values) < min_n:
            continue
        med = _median(values)
        mad = _mad(values)
        if mad <= 0:
            continue
        for p in per_episode:
            v = (p.get("measurements") or {}).get(metric)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            z = abs(float(v) - med) / mad
            if z > k:
                direction = "偏大" if v > med else "偏小"
                out[p["episode_index"]].append(_finding(
                    "anomaly", "跨 episode 离群", "warning",
                    f"{labels[metric]}{direction}：{float(v):.4g}，数据集典型值 {med:.4g}"
                    f"（偏离 {z:.1f}×MAD）",
                    measured=round(float(v), 4), expected=f"{med:.4g} ± {k}×MAD"))
    return out


# --------------------------------------------------------------------------- 可移植性


def check_portability(ds_path, meta: dict, index: int, record: dict, cfg: dict) -> tuple[list, dict]:
    findings: list[dict] = []
    pc = cfg.get("portability") or {}
    info = meta.get("info") or {}
    measurements: dict = {}

    if pc.get("check_absolute_paths", True):
        suspicious = []
        for field in ("data_path", "video_path"):
            value = info.get(field)
            if isinstance(value, str) and (":" in value.split("{")[0] or value.startswith("/")):
                suspicious.append(f"{field}={value}")
        if suspicious:
            findings.append(_finding(
                "portability", "元数据含绝对路径", "warning",
                f"元数据里出现绝对路径，换机器/迁移到 Linux 会失效：{suspicious}",
                measured=len(suspicious), expected="只用相对路径模板"))

    if pc.get("check_filename_ascii", True):
        root = Path(ds_path)
        name = root.name
        if any(ord(ch) > 127 for ch in name):
            findings.append(_finding(
                "portability", "数据集名含非 ASCII", "info",
                f"数据集目录名含非 ASCII 字符（{name}）——跨平台/命令行容易出编码问题",
                measured=name, expected="建议用 ASCII 命名"))
        if " " in name:
            findings.append(_finding(
                "portability", "数据集名含空格", "info",
                f"数据集目录名含空格（{name}）——脚本里容易漏引号",
                measured=name, expected="建议用下划线"))
    return findings, measurements


# --------------------------------------------------------------------------- 运动学


def check_kinematics(ds_path, meta: dict, index: int, record: dict, table: dict,
                     cfg: dict) -> tuple[list, dict]:
    kc = cfg.get("kinematics") or {}
    limits = kc.get("limit_table") or {}
    findings: list[dict] = []
    measurements: dict = {}
    if not limits:
        findings.append(_finding(
            "kinematics", "关节限位", "info",
            "未配置关节限位表，跳过限位检查（在 qc_config.json 的 kinematics.limit_table 里填）",
            measured="未配置", expected="提供 [[lo,hi], ...] 后可查限位"))
        return findings, measurements

    for feature, table_limits in limits.items():
        series = _series(table, feature)
        if not series:
            continue
        violations = 0
        for row in series:
            for j, value in enumerate(row):
                if j >= len(table_limits):
                    break
                bounds = table_limits[j]
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    if value < float(bounds[0]) or value > float(bounds[1]):
                        violations += 1
        if violations:
            findings.append(_finding(
                "kinematics", "超出关节限位", "error",
                f"{feature} 有 {violations} 个值超出配置的关节限位（机械臂不可能到达的位姿）",
                measured=violations, expected="0（对照 kinematics.limit_table）"))
    return findings, measurements


# --------------------------------------------------------------------------- 健康度


def health_score(findings: list[dict], cfg: dict) -> dict:
    """把 findings 加权成 0~100 分 + A~F 等级。

    ⚠️ **仅用于分诊排序，不代表训练效果。**
    有研究表明示范评分无法预测保留它能否改善策略（orbit-robotics 用 701 个策略
    的对照实验撤回了自家 A~F 评分）。所以这里明确标注、且不参与任何自动删除决策。
    """
    hc = cfg.get("health") or {}
    score = 100.0
    for f in findings:
        sev = f.get("severity")
        if sev == "error":
            score -= float(hc.get("error_penalty") or 25.0)
        elif sev == "warning":
            score -= float(hc.get("warning_penalty") or 6.0)
        elif sev == "info":
            score -= float(hc.get("info_penalty") or 0.0)
    score = max(0.0, min(100.0, score))
    grade = "F"
    for threshold, letter in hc.get("grades") or []:
        if score >= float(threshold):
            grade = letter
            break
    return {"score": round(score, 1), "grade": grade,
            "note": "仅用于分诊排序，不预测训练效果，也不参与自动删除"}


# --------------------------------------------------------------------------- 主编排


def check_episode_extended(ds_path, meta: dict, index: int, record: dict,
                           cfg: dict | None = None) -> dict:
    """跑一遍扩展检查。返回 findings / measurements / checks_run（覆盖率用）。"""
    cfg = cfg or load_config()
    enabled = cfg.get("enabled_categories") or {}
    findings: list[dict] = []
    measurements: dict = {}
    checks_run: list[str] = []

    def on(category: str) -> bool:
        return bool(enabled.get(category, True))

    # parquet 读一次，后面复用
    table: dict = {}
    if record.get("parquet_path") and Path(record["parquet_path"]).is_file():
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(record["parquet_path"]).to_pydict()
        except Exception as exc:  # noqa: BLE001
            findings.append(_finding("format", "扩展质检读取", "warning",
                                     f"扩展检查无法读取 parquet：{type(exc).__name__}: {exc}",
                                     measured="-", expected="可读取"))
            table = {}

    # (分类, 函数, 是否需要 parquet 表) —— 需要表的传 table，不需要的少传一个参数
    registry = (
        ("metadata", check_metadata, False),
        ("temporal", check_temporal, True),
        ("action", check_action, True),
        ("video", check_video_decode, False),
        ("stats", check_stats, True),
        ("portability", check_portability, False),
        ("kinematics", check_kinematics, True),
    )
    for category, fn, needs_table in registry:
        if not on(category):
            continue
        if needs_table and not table:
            continue
        try:
            if needs_table:
                f, m = fn(ds_path, meta, index, record, table, cfg)
            else:
                f, m = fn(ds_path, meta, index, record, cfg)
            findings.extend(f)
            measurements.update(m)
            checks_run.append(category)
        except Exception as exc:  # noqa: BLE001
            # 检查崩溃是基础设施问题，不是坏数据 —— 记 warning 但不判数据不合格
            findings.append(_finding(category, f"{category} 检查异常", "warning",
                                     f"检查过程出错（这是基础设施问题，不代表数据坏）："
                                     f"{type(exc).__name__}: {exc}",
                                     measured="-", expected="检查正常完成"))
    return {"episode_index": index, "findings": findings,
            "measurements": measurements, "checks_run": checks_run}


def coverage(all_runs: list[dict]) -> dict:
    """覆盖率分母：哪些分类跑在了多少比例的 episode 上。"""
    total = len(all_runs)
    if not total:
        return {"episodes": 0, "categories": {}}
    counts: dict[str, int] = {}
    for run in all_runs:
        for category in run.get("checks_run") or []:
            counts[category] = counts.get(category, 0) + 1
    return {"episodes": total,
            "categories": {CATEGORY_LABELS.get(k, k): round(v / total, 3)
                           for k, v in sorted(counts.items())}}


#: 一条规则命中多少比例的 episode 就算「数据集级普遍问题」
WIDESPREAD_RATIO = 0.9


def episode_problem_summary(dataset_id: int, run_id: int | None = None) -> dict:
    """按 episode 汇总最近一次质检的问题 —— 「挑出坏 episode」与「数据浏览」共用。

    关键处理：**命中几乎全部 episode 的规则其实是数据集级问题**（例如两个 video 特征
    都缺 video_info.fps），靠排除单条根本解决不了；若留在逐条计数里，会把每条 episode
    都染成 warning，把真正该丢的那几条淹没。所以这类规则单列到 `dataset_level`。

    只做汇总与建议，**不删任何东西**。
    """
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        return {"error": "数据集不存在"}

    if run_id:
        run = db.query_one("SELECT * FROM qc_runs WHERE id=?", (run_id,))
    else:
        run = db.query_one("SELECT * FROM qc_runs WHERE dataset_id=? ORDER BY id DESC LIMIT 1",
                           (dataset_id,))

    episodes = db.query("SELECT episode_index, length, task_text, success_count"
                        " FROM episodes WHERE dataset_id=? ORDER BY episode_index", (dataset_id,))
    if not episodes:
        total = int(row.get("total_episodes") or 0)
        episodes = [{"episode_index": i, "length": None, "task_text": None,
                     "success_count": None} for i in range(total)]

    findings = db.query(
        "SELECT episode_index, rule, severity, detail, category, source FROM qc_findings"
        " WHERE run_id=? ORDER BY id", (run["id"],)) if run else []

    labels = dict(CATEGORY_LABELS)
    try:
        from . import remote_qc
        labels.update(remote_qc.CATEGORY_LABEL)
    except Exception:  # noqa: BLE001
        pass

    by_ep: dict[int, list] = {}
    dataset_level: list[dict] = []
    for item in findings:
        if item["episode_index"] is None:
            dataset_level.append(item)
        else:
            by_ep.setdefault(int(item["episode_index"]), []).append(item)

    n_eps = max(1, len(episodes))
    rule_hits: dict[str, set] = {}
    for idx, items in by_ep.items():
        for f in items:
            if f["severity"] == "info":
                continue
            rule_hits.setdefault(f["rule"], set()).add(idx)
    threshold = max(3, int(n_eps * WIDESPREAD_RATIO + 0.5))
    widespread: set[str] = set()
    for rule, hits in rule_hits.items():
        if len(hits) >= threshold and len(hits) >= n_eps - 1:
            widespread.add(rule)
            sample = next(f for f in findings if f["rule"] == rule)
            dataset_level.append({
                "rule": rule, "severity": sample["severity"],
                "category": labels.get(sample.get("category") or "",
                                       sample.get("category") or ""),
                "detail": f"命中 {len(hits)}/{n_eps} 条 episode —— {sample['detail']}",
            })

    out = []
    for ep in episodes:
        idx = int(ep["episode_index"])
        items = [f for f in by_ep.get(idx, []) if f["rule"] not in widespread]
        errors = [f for f in items if f["severity"] == "error"]
        warnings = [f for f in items if f["severity"] == "warning"]
        infos = [f for f in items if f["severity"] == "info"]
        out.append({
            "episode_index": idx,
            "length": ep.get("length"),
            "task_text": (ep.get("task_text") or "").strip(),
            "success_count": ep.get("success_count"),
            "errors": len(errors),
            "warnings": len(warnings),
            "infos": len(infos),
            "categories": sorted({labels.get(f.get("category") or "", f.get("category") or "其他")
                                  for f in items if f["severity"] != "info"}),
            "sources": sorted({(f.get("source") or "local") for f in items
                               if f["severity"] != "info"}),
            # 建议：有 error 才"建议排除"；只有 warning 标"待定"交人决定。
            # 刻意如此 —— VLM 判定只到 warning，不该由它单独决定丢数据。
            "suggest": "drop" if errors else ("review" if warnings else "keep"),
            "issues": [f"{f['rule']}：{f['detail']}" for f in (errors + warnings)[:4]]
                      or [f"{f['rule']}：{f['detail']}" for f in infos[:2]],
        })

    return {
        "dataset": {"id": row["id"], "name": row["name"],
                    "total_episodes": row.get("total_episodes"),
                    "total_frames": row.get("total_frames"),
                    "fps": row.get("fps"), "path": row.get("path"),
                    "codebase_version": row.get("codebase_version"),
                    "codec": row.get("codec")},
        "run": ({"id": run["id"], "created_at": run["created_at"],
                 "result": run.get("result"), "report_path": run.get("report_path")}
                if run else None),
        "dataset_level": [{"rule": f["rule"], "severity": f["severity"],
                           "category": labels.get(f.get("category") or "",
                                                  f.get("category") or ""),
                           "detail": f["detail"]} for f in dataset_level],
        "episodes": out,
        "widespread_rules": sorted(widespread),
        "counts": {"total": len(out),
                   "with_error": sum(1 for e in out if e["errors"]),
                   "with_warning": sum(1 for e in out if e["warnings"] and not e["errors"]),
                   "clean": sum(1 for e in out if not e["errors"] and not e["warnings"])},
    }
