"""VLM 质检：用视觉语言模型判断一条 episode 的**内容**是否合规。

与 `core/lerobot.py` 的 13 条结构规则并列，分工明确：

- `lerobot.check_episode()` 管**格式与事实**：行数、帧数、NaN、时间戳、维度……
- 本模块管**画面里发生了什么**：任务到底做没做成、有没有失败迹象、画面能不能用。

两者产出的 findings 结构**完全一致**（rule / severity / detail / measured / expected），
所以直接写进同一张 `qc_findings` 表，下游「打回重采」「按质检通过导出」
「修正类预处理 qc_pass」全部**一行都不用改**。

后端只依赖 **OpenAI 兼容的 `/v1/chat/completions`**（图片走 `image_url`），
因此 ollama / llama.cpp(llama-server) / vLLM / LM Studio / 云端 API 都能当后端；
平台侧不需要装 transformers、不需要 CUDA、不需要 requests。

两种质检模式：

- `checklist`（判别式）：预设问题 + yes/no。输出确定、便宜、好校准，
  但**只能查你想到的问题**。每条问题可带 `q_neg`（反向问法），用于抵消
  VLM 的 yes 偏置：正向问「是否已完成」、反向问「是否未完成」，两者必须一致。
- `describe`（描述式）：先让 VLM 客观描述「初始状态 / 做了什么 / 结束状态」，
  再让它给判定。这是开放集，**能发现你没预设的问题**（遮挡、异常光照、
  多余人手入镜、重复采集……），顺带产出的描述就是 VLA 要的语言指令。
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import db

# --------------------------------------------------------------------------- 配置

CONFIG_PATH = db.STORAGE / "vlm_config.json"

#: 默认问题清单 —— **语义级**判断，与底层质检互补。
#: 定位：底层质检（结构规则 / lerobot-doctor）管"数据格式、时序、统计、视频能不能解码"；
#: 这里管"画面里到底发生了什么"——任务是否执行完、动作成败、有人经过、开放世界意外事件。
#:
#: 约定：每条含正向问法 `q` 与反向问法 `q_neg`，`expect` 声明正向问法的合规答案。
#:   - `expect: "yes"` —— 「是不是做到了」型问题（做到了才合规）
#:   - `expect: "no"`  —— 「有没有出问题」型问题（没出问题才合规）
#: 一组正反成对出现，VLM 若「无脑答 yes」会在反向问题上暴露，
#: 被 `_parse_checklist` 判为「正反不一致 → 待人工确认」，而不是蒙混过关。
#:
#: **所有问题在同一次调用里问**（见 `_build_prompts`），所以加问题几乎不增加成本。
#: 想加问题：照着写一条 q / q_neg / expect 即可，不需要改代码。
DEFAULT_CHECKLIST = [
    {
        "rule": "VLM·任务完成",
        "q": "该片段结束时，机器人是否已经完成了任务目标（目标物体已被抓取并放到指定位置或容器中）？",
        "q_neg": "该片段结束时，机器人是否尚未完成任务目标（物体仍在原处、掉落在半路、或被放到了错误位置）？",
        "expect": "yes",
        "severity": "warning",
    },
    {
        "rule": "VLM·动作成败",
        "q": ("该片段的操作是否全程成功、没有出现失败迹象（没有物体掉落、没有卡住、"
              "没有反复重试同一步、没有中途放弃）？"),
        "q_neg": "该片段中是否出现了明显的失败迹象（物体掉落、卡住、反复重试同一步、中途放弃）？",
        "expect": "yes",
        "severity": "warning",
    },
    {
        # 刻意把「人手介入操作」与「旁人经过」**拆成两项**。
        # 起因：实测（zhiyan #38）末帧出现操作员的手在桌面上操作物体，
        # 而原来的问法把两者混在一起问「是否有人经过或进入画面（哪怕只是路过）」，
        # 模型对"手在操作算不算经过"没有稳定标准 → 正反问自相矛盾。
        # 两者对数据质量的含义完全不同：
        #   人手介入 = 遥操作辅助 / 人工完成 / 收尾干预 → 任务层面的问题
        #   旁人经过 = 环境噪声污染 → 训练数据层面的问题
        "rule": "VLM·人手介入",
        "q": ("该片段中是否有人的手或手臂参与操作（接触物体、工具或设备），"
              "而不只是出现在背景里？"),
        "q_neg": ("该片段中是否始终只有机器人本体在操作，没有任何人的手或手臂"
                  "接触过物体、工具或设备？"),
        "expect": "no",
        "severity": "warning",
    },
    {
        "rule": "VLM·旁人经过",
        "q": "该片段中是否有人从旁边经过或进入画面，但没有接触物体、工具或设备？",
        "q_neg": "该片段中是否始终没有任何人进入画面（背景里也没有人出现）？",
        "expect": "no",
        "severity": "warning",
    },
    {
        # 开放世界：不预设类别，让模型自己报"任何与任务无关的意外事件"。
        # 用 info 而不是 warning —— 开放式问题的触发率天然很高，
        # 一律告警会造成告警疲劳；落成 info 交人工复核，也守住"VLM 不产生 error"的红线。
        "rule": "VLM·开放世界事件",
        "q": ("除机器人自身操作外，该片段中是否出现了任何与任务无关的意外事件"
              "（物体被碰动或移位、人以外的主体进入画面、灯光或画面突变、设备异常、"
              "画面剧烈抖动等）？"),
        "q_neg": "该片段中是否完全没有出现任何与任务无关的意外事件（画面自始至终只有机器人正常操作）？",
        "expect": "no",
        "severity": "info",
    },
]

DEFAULTS: dict = {
    # 后端（OpenAI 兼容）。默认指向 ollama；llama-server 用 http://127.0.0.1:8080/v1
    "base_url": "http://127.0.0.1:11434/v1",
    "model": "qwen3-vl:4b",
    "api_key": "ollama",
    # checklist | describe | both
    "mode": "both",
    # 抽帧与拼图
    "frames": 6,          # 每条相机抽几帧（首帧与末帧必抽）
    "tile_width": 320,    # 每格宽度，高度按源画幅比例；越小越省显存
    "max_tiles": 12,      # 全图最大格数（多相机时按相机数摊分）
    "jpeg_quality": 88,
    # 调用
    "max_tokens": 700,
    "timeout_s": 180,
    "temperature": 0.0,
    # 判定
    "severity": "warning",            # checklist 判不合规时的默认级别
    "fail_verdict_severity": "warning",  # describe 判 fail 时的级别
    "uncertain_severity": "info",     # 判定不确定 / 正反不一致时的级别
    "emit_description": True,         # 把描述式结果作为 info finding 写进报告
    # 行为
    "use_cache": True,
    "checklist": DEFAULT_CHECKLIST,
    "describe_instruction": (
        "请先用三句话客观描述这条片段：①初始状态（物体与机器人的位置）"
        "②机器人做了哪些动作 ③结束状态（物体最终在哪、机械臂停在哪）。"
        "然后判断任务是否完成、是否存在失败迹象。"
    ),
}

#: describe 模式的 JSON 契约（同时写进提示词，便于人看与校正）
DESCRIBE_SCHEMA = {
    "description": "对初始状态/动作/结束状态的客观描述",
    "suggested_instruction": "用一句中文祈使句概括这条片段的任务（VLA 语言指令用）",
    "task_completed": "true|false",
    "failure_signs": ["失败迹象，没有就留空数组"],
    "verdict": "pass|fail|uncertain",
    "reason": "给判定的理由",
}


def config_path() -> Path:
    return CONFIG_PATH


def load_config(overrides: dict | None = None) -> dict:
    """默认值 ← storage/vlm_config.json ← 本次任务传入的 overrides（优先级递增）。"""
    cfg = json.loads(json.dumps(DEFAULTS))  # 深拷贝，避免调用方改到 DEFAULTS
    if CONFIG_PATH.is_file():
        try:
            patch = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(patch, dict):
                cfg.update(patch)
        except Exception:  # noqa: BLE001 - 配置文件坏了不该让质检跑不起来
            pass
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


def save_config(patch: dict) -> dict:
    """保存配置（只覆盖传入的键）。checklist 传空列表表示恢复默认。"""
    current: dict = {}
    if CONFIG_PATH.is_file():
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            current = {}
    for key, value in (patch or {}).items():
        if key == "checklist" and value == []:
            current.pop("checklist", None)
            continue
        current[key] = value
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_config()


# --------------------------------------------------------------------------- 抽帧与拼图


def _sample_indices(total: int, count: int) -> list[int]:
    """在 [0, total) 上均匀取 count 个下标，**首帧与末帧一定包含**。"""
    if total <= 0 or count <= 0:
        return []
    if count == 1:
        return [0]
    if total <= count:
        return list(range(total))
    step = (total - 1) / (count - 1)
    out = sorted({int(round(i * step)) for i in range(count)})
    out[0], out[-1] = 0, total - 1
    return out


def _letterbox(image, width: int, height: int):
    """等比缩放到 (width, height) 内并补边，避免不同画幅被拉伸变形。"""
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    scale = min(width / float(w), height / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _label(image, text: str, scale: float = 0.5):
    """在左上角写帧号/相机名（cv2 只能画 ASCII，所以标签一律用英文数字）。"""
    import cv2

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(image, (0, 0), (tw + 8, th + 8), (0, 0, 0), -1)
    cv2.putText(image, text, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)
    return image


def _grid_image(tiles: list[list], width: int, height: int, cols: int):
    """把二维列表的图块拼成一张大图（行内横stack，行间竖stack）。"""
    import cv2
    import numpy as np

    rows = []
    for line in tiles:
        while len(line) < cols:                      # 补齐，避免 hstack 形状不一致
            line = line + [np.full((height, width, 3), 32, dtype=np.uint8)]
        rows.append(cv2.hconcat(line[:cols]))
    if not rows:
        return None
    return cv2.vconcat(rows) if len(rows) > 1 else rows[0]


def _encode_jpeg(image, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def _read_video_frame(video_path: Path, index: int):
    """按帧号从 mp4 取一帧（BGR ndarray）；取不到返回 None。"""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok and frame is not None else None


def _probe_cameras(ds_path, meta: dict, episode_index: int) -> list[dict]:
    """这条 episode 有哪些视觉源：返回 [{key, kind, path/frames}]，video 优先。"""
    from . import lerobot, inline_images

    info = meta["info"]
    paths = lerobot.episode_paths(info, ds_path, episode_index)
    cams: list[dict] = []
    for key, video in paths["videos"].items():
        if Path(video).is_file():
            import cv2

            cap = cv2.VideoCapture(str(video))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            if frames > 0:
                cams.append({"key": key, "kind": "video", "path": str(video), "frames": frames})
    if cams:
        return cams

    # 没有 mp4：LIBERO 这类把图像内嵌在 parquet 里的数据集
    keys = inline_images.image_keys(info)
    if keys:
        import pyarrow.parquet as pq

        parquet = paths["parquet"]
        rows = pq.read_table(parquet).num_rows if Path(parquet).is_file() else 0
        if rows > 0:
            cams = [{"key": k, "kind": "inline", "path": str(parquet), "frames": rows} for k in keys]
    return cams


def build_images(ds_path, meta: dict, episode_index: int, cfg: dict) -> dict:
    """抽帧 + 拼成一张大图，返回 data URL 与缓存用的内容指纹。

    布局：**列 = 时间（左→右），行 = 相机**。同一行是同一台相机的时序，
    同一列是同一时刻的多视角 —— 让 VLM 直接做首末对比与跨视角对照。
    """
    import cv2
    import numpy as np
    from . import inline_images

    cams = _probe_cameras(ds_path, meta, episode_index)
    if not cams:
        return {"images": [], "tiles": 0, "cameras": [], "source": "none", "sha": "",
                "note": "该 episode 没有可用的视觉数据（无 mp4、也无内嵌图像）"}

    max_tiles = max(1, int(cfg.get("max_tiles") or 12))
    want = max(2, int(cfg.get("frames") or 6))
    # 相机多时按相机数摊分帧数，总格数不超过 max_tiles
    per_cam = max(2, min(want, max_tiles // max(1, len(cams))))
    cols = per_cam

    first = None
    grid: list[list] = []
    used_cams: list[dict] = []
    for cam in cams:
        indices = _sample_indices(cam["frames"], per_cam)
        tiles = []
        for idx in indices:
            frame = None
            try:
                if cam["kind"] == "video":
                    frame = _read_video_frame(Path(cam["path"]), idx)
                else:
                    raw, _ctype, _key = inline_images.frame_bytes(
                        ds_path, meta, episode_index, idx, cam["key"])
                    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001 - 单帧失败不该让整条 episode 质检崩掉
                frame = None
            if frame is None:
                continue
            if first is None:
                h, w = frame.shape[:2]
                aspect = (h / float(w)) if w else 1.0
                tile_w = max(96, int(cfg.get("tile_width") or 320))
                tile_h = int(max(96, min(480, round(tile_w * aspect))))
            tiles.append(_label(_letterbox(frame, tile_w, tile_h),
                                f"{cam['key'][-14:]} f{idx}", 0.45))
        if not tiles:
            continue
        # 补齐到 cols，保证每行宽度一致
        while len(tiles) < cols:
            tiles.append(np.full((tile_h, tile_w, 3), 32, dtype=np.uint8))
        grid.append(tiles[:cols])
        used_cams.append({"key": cam["key"], "kind": cam["kind"],
                          "indices": indices, "frames": cam["frames"]})

    image = _grid_image(grid, tile_w, tile_h, cols)
    if image is None:
        return {"images": [], "tiles": 0, "cameras": [], "source": "none", "sha": "",
                "note": "抽帧失败：所有采样帧都解不出来"}

    payload = _encode_jpeg(image, cfg.get("jpeg_quality") or 88)
    sha = hashlib.sha256(payload).hexdigest()
    height, width = image.shape[:2]
    return {
        "images": ["data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")],
        "tiles": sum(len(r) for r in grid),
        "cameras": used_cams,
        "source": used_cams[0]["kind"] if used_cams else "none",
        "size": f"{width}x{height}",
        "bytes": len(payload),
        "sha": sha,
        "note": "",
    }


# --------------------------------------------------------------------------- 提示词


def _layout_note(cams: list[dict], cols: int) -> str:
    lines = [f"图像是一张拼图，共 {len(cams)} 行 × {cols} 列（列按时间从左到右，行是不同相机）。"]
    for row, cam in enumerate(cams):
        indices = cam.get("indices") or []
        lines.append(f"第 {row + 1} 行 = 相机 {cam['key']}，抽帧号 {indices}（已按时间排序）。")
    lines.append("每格左上角的白色文字是「相机名 帧号」。")
    return "\n".join(lines)


def _as_bool(value):
    """把模型返回的 true/"true"/1/"yes" 统一成 bool；无法判断时返回 None。

    实测 Qwen3-VL 会把 `task_completed` 返回成**字符串** "true"，
    如果只用 `is False` 判断，字符串 "false" 会被当成"不是 False"而漏掉。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "y", "1", "是", "完成"):
            return True
        if text in ("false", "no", "n", "0", "否", "未完成"):
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def task_text_of(meta: dict, episode_index: int) -> str:
    """取这条 episode 的任务描述（LeRobot 的 tasks）。

    **必须喂给 VLM**：不然"任务是否完成"这个问题没有判据 ——
    实测不给任务描述时，模型会把画面理解成完全不相干的事情（例如把移动的方块说成"拼图"）。
    """
    declared = (meta.get("episodes") or {}).get(episode_index) or {}
    tasks = [str(t).strip() for t in (declared.get("tasks") or []) if str(t).strip()]
    if tasks:
        return "；".join(tasks)
    return ""


def build_prompts(cfg: dict, image_meta: dict, task_text: str = "") -> dict:
    """构造两种模式要发的提示词（以及需要解析的 JSON 契约）。"""
    cams = image_meta.get("cameras") or []
    cols = max((len(c.get("indices") or []) for c in cams), default=0) or 1
    layout = _layout_note(cams, cols)

    # 把任务描述明确写进提示词：这是"任务是否完成"的唯一判据。
    # 没有它，模型只能瞎猜画面在干什么（实测会把机械臂抓放说成"拼图"）。
    if task_text:
        task_note = f"\n本条片段的任务是：**{task_text}**\n请以这个任务为准判断是否完成，不要臆测成别的任务。\n"
    else:
        task_note = ("\n本条片段**没有提供任务描述**，请只依据画面判断「这一串操作本身是否完整、"
                     "有没有中断/掉落/反复重试」，不要猜测具体任务名。\n")

    prompts: dict = {}

    items = cfg.get("checklist") or []
    if items and cfg.get("mode") in ("checklist", "both"):
        numbered: list[dict] = []
        for pos, item in enumerate(items, start=1):
            numbered.append({"id": f"{pos}", "q": item.get("q"),
                             "polarity": "positive", "rule": item.get("rule") or f"问题{pos}"})
            if item.get("q_neg"):
                numbered.append({"id": f"{pos}n", "q": item.get("q_neg"),
                                 "polarity": "negative", "rule": item.get("rule") or f"问题{pos}"})
        question_text = "\n".join(f'{n["id"]}. {n["q"]}' for n in numbered)
        prompts["checklist"] = {
            "questions": numbered,
            "text": (
                "你是机器人操作数据的质检员。下面给你的是**同一条操作片段**的抽帧画面。\n"
                f"{layout}\n"
                f"{task_note}\n"
                "请**只依据画面**逐条回答下列问题，每条都只回答 yes 或 no，不要含糊。\n"
                "注意：问题里有正向问法也有反向问法，请分别独立判断，不要为了保持一致而改答案。\n\n"
                f"{question_text}\n\n"
                "严格输出如下 JSON（不要输出任何其它文字）：\n"
                '{"answers": [{"id": "1", "answer": "yes|no", "why": "一句话理由"}], '
                '"overall": "一句话总体印象"}'
            ),
        }

    if cfg.get("mode") in ("describe", "both"):
        prompts["describe"] = {
            "text": (
                "你是机器人操作数据的质检员。下面给你的是**同一条操作片段**的抽帧画面。\n"
                f"{layout}\n"
                f"{task_note}\n"
                f"{cfg.get('describe_instruction') or DEFAULTS['describe_instruction']}\n\n"
                "严格输出如下 JSON（不要输出任何其它文字）：\n"
                + json.dumps(DESCRIBE_SCHEMA, ensure_ascii=False, indent=2)
            ),
        }

    return prompts


def prompt_hash(cfg: dict) -> str:
    """影响结果的所有配置的指纹：配置一变，缓存自动失效。"""
    payload = {
        "model": cfg.get("model"),
        "mode": cfg.get("mode"),
        "checklist": cfg.get("checklist") or [],
        "describe_instruction": cfg.get("describe_instruction"),
        "frames": cfg.get("frames"),
        "tile_width": cfg.get("tile_width"),
        "max_tiles": cfg.get("max_tiles"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- 调用后端


def _endpoint(cfg: dict) -> str:
    base = (cfg.get("base_url") or "").rstrip("/")
    return f"{base}/chat/completions"


def _post_json(url: str, body: dict, cfg: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg.get('api_key') or 'none'}"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_json(text: str) -> dict:
    """从模型输出里抠出第一个 JSON 对象（兼容 ```json 包裹与前后废话）。"""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:  # noqa: BLE001
        return {}


def chat(cfg: dict, images: list[str], prompt: str) -> dict:
    """发一次多模态请求，返回解析后的 JSON（失败抛异常，由上层决定如何降级）。"""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    body = {
        "model": cfg.get("model"),
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(cfg.get("max_tokens") or 700),
        "temperature": float(cfg.get("temperature") or 0.0),
        "response_format": {"type": "json_object"},
    }
    timeout = float(cfg.get("timeout_s") or 180)
    started = time.time()
    try:
        raw = _post_json(_endpoint(cfg), body, cfg, timeout)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        # 有些后端不认 response_format，去掉重试一次
        if exc.code in (400, 404, 422):
            body.pop("response_format", None)
            raw = _post_json(_endpoint(cfg), body, cfg, timeout)
        else:
            raise RuntimeError(f"VLM 返回 HTTP {exc.code}：{detail}") from exc
    latency = int((time.time() - started) * 1000)

    choices = raw.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    text = message.get("content")
    if isinstance(text, list):     # 有的后端把 content 拆成数组
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    parsed = _extract_json(text or "")
    return {"json": parsed, "text": text or "", "latency_ms": latency, "usage": raw.get("usage") or {}}


def health(cfg: dict | None = None) -> dict:
    """探活：后端是否可达、模型是否在列。不抛异常，只返回事实。"""
    cfg = cfg or load_config()
    base = (cfg.get("base_url") or "").rstrip("/")
    result = {"ok": False, "base_url": base, "model": cfg.get("model"), "detail": ""}
    try:
        request = urllib.request.Request(
            f"{base}/models",
            headers={"Authorization": f"Bearer {cfg.get('api_key') or 'none'}"})
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [m.get("id") for m in (payload.get("data") or []) if isinstance(m, dict)]
        result["ok"] = True
        result["models"] = names[:50]
        result["model_found"] = (not names) or (cfg.get("model") in names) or any(
            str(cfg.get("model") or "").split(":")[0] in str(n) for n in names)
        result["detail"] = f"后端可达，返回 {len(names)} 个模型"
        if names and not result["model_found"]:
            result["detail"] += f"，但没找到 {cfg.get('model')}"
    except Exception as exc:  # noqa: BLE001
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result


# --------------------------------------------------------------------------- 判定 → findings


def _finding(rule: str, severity: str, detail: str, measured=None, expected=None) -> dict:
    return {"rule": rule, "severity": severity, "detail": detail,
            "measured": None if measured is None else str(measured),
            "expected": None if expected is None else str(expected)}


def _parse_checklist(cfg: dict, prompts: dict, answer_json: dict, raw_text: str) -> tuple[list, list]:
    """把模型的逐条回答对回配置，产出 (checklist 结果, findings)。"""
    asked = (prompts.get("checklist") or {}).get("questions") or []
    answers = {str(a.get("id")): a for a in (answer_json.get("answers") or [])
               if isinstance(a, dict)}
    items = cfg.get("checklist") or []
    default_sev = cfg.get("severity") or "warning"
    uncertain_sev = cfg.get("uncertain_severity") or "info"

    by_group: dict[str, dict] = {}
    results: list[dict] = []
    for entry in asked:
        answer = answers.get(str(entry["id"])) or {}
        value = str(answer.get("answer") or "").strip().lower()
        value = "yes" if value.startswith("y") else ("no" if value.startswith("n") else "unknown")
        record = {"id": entry["id"], "rule": entry["rule"], "q": entry["q"],
                  "polarity": entry["polarity"], "answer": value,
                  "why": str(answer.get("why") or "")[:200]}
        results.append(record)
        group = by_group.setdefault(entry["rule"], {})
        group["negative" if entry["polarity"] == "negative" else "positive"] = record

    findings: list[dict] = []
    for pos, item in enumerate(items, start=1):
        rule = item.get("rule") or f"问题{pos}"
        severity = item.get("severity") or default_sev
        expect = str(item.get("expect") or "yes").strip().lower()
        group = by_group.get(rule) or {}
        positive = group.get("positive")
        negative = group.get("negative")

        if not positive or positive["answer"] == "unknown":
            findings.append(_finding(
                f"{rule}（未取到回答）", uncertain_sev,
                f"VLM 没有给出可解析的回答，原始输出片段：{raw_text[:120]!r}",
                measured="unknown", expected=expect))
            continue

        ok = positive["answer"] == expect
        # 正反问自一致：反向问法的期望是 expect 的反面
        agreement = None
        if negative and negative["answer"] != "unknown":
            expect_neg = "no" if expect == "yes" else "yes"
            agreement = (positive["answer"] == expect) == (negative["answer"] == expect_neg)
        positive["ok"] = ok
        positive["agreement"] = agreement

        if agreement is False:
            findings.append(_finding(
                f"{rule}（正反不一致）", uncertain_sev,
                f"正向「{positive['q']}」答 {positive['answer']}，"
                f"反向「{negative['q']}」答 {negative['answer']}，两次判断矛盾，标为待人工确认",
                measured=f"正 {positive['answer']} / 反 {negative['answer']}", expected="两次一致"))
            continue

        if not ok:
            why = positive.get("why") or ""
            findings.append(_finding(
                rule, severity,
                f"VLM 判定不合规：{positive['q']} → {positive['answer']}"
                + (f"（理由：{why}）" if why else ""),
                measured=f"{positive['answer']}（{'正反一致' if agreement else '单向'}）",
                expected=expect))
    return results, findings


def _parse_describe(cfg: dict, answer_json: dict) -> tuple[dict, list[dict]]:
    """把描述式结果转成 (describe, findings)。"""
    severity = cfg.get("fail_verdict_severity") or "warning"
    uncertain_sev = cfg.get("uncertain_severity") or "info"
    verdict = str(answer_json.get("verdict") or "").strip().lower()
    description = str(answer_json.get("description") or "").strip()
    reason = str(answer_json.get("reason") or "").strip()
    signs = answer_json.get("failure_signs") or []
    if isinstance(signs, str):
        signs = [signs]
    completed = _as_bool(answer_json.get("task_completed"))
    describe = {
        "description": description,
        "suggested_instruction": str(answer_json.get("suggested_instruction") or "").strip(),
        "task_completed": completed,
        "failure_signs": [str(s) for s in signs if str(s).strip()],
        "verdict": verdict or "uncertain",
        "reason": reason,
    }

    findings: list[dict] = []
    if verdict == "fail":
        findings.append(_finding(
            "VLM·描述判定", severity,
            "VLM 判该片段不合规" + (f"：{reason}" if reason else "")
            + (f"；失败迹象：{'；'.join(describe['failure_signs'])}" if describe["failure_signs"] else ""),
            measured="fail", expected="pass"))
    elif verdict not in ("pass",):
        findings.append(_finding(
            "VLM·描述判定（不确定）", uncertain_sev,
            f"VLM 无法确定该片段是否合规{('：' + reason) if reason else ''}，建议人工复核",
            measured=verdict or "unknown", expected="pass|fail"))

    if completed is False and verdict == "pass":
        # 自相矛盾：判定 pass 但明确说任务没完成 → 降级为待人工确认，不当作不合规
        findings.append(_finding(
            "VLM·判定自相矛盾", uncertain_sev,
            "VLM 给出 pass，但同时认为任务未完成，判定不可信，建议人工复核",
            measured="task_completed=false 且 verdict=pass", expected="两者一致"))
    return describe, findings


# --------------------------------------------------------------------------- 缓存


def _cache_get(dataset_id, episode_index, cfg, p_hash, content_sha) -> dict | None:
    if not cfg.get("use_cache") or not dataset_id or not content_sha:
        return None
    try:
        row = db.query_one(
            "SELECT * FROM vlm_checks WHERE dataset_id=? AND episode_index=? AND model=?"
            " AND prompt_hash=? AND content_sha=? ORDER BY id DESC LIMIT 1",
            (dataset_id, episode_index, cfg.get("model"), p_hash, content_sha))
    except Exception:  # noqa: BLE001 - 老库还没建表
        return None
    if not row:
        return None
    try:
        row["result"] = json.loads(row.get("findings_json") or "{}")
    except Exception:  # noqa: BLE001
        return None
    return row


def _cache_put(dataset_id, episode_index, cfg, p_hash, content_sha, result) -> None:
    # use_cache=False 是"完全不碰缓存"：既不读也不写。
    # （只判断读不判断写会让关掉缓存的调用悄悄污染缓存，下次开缓存时读到过期结论）
    if not cfg.get("use_cache") or not dataset_id or not content_sha:
        return
    try:
        db.execute(
            "INSERT INTO vlm_checks (dataset_id, episode_index, model, endpoint, mode, prompt_hash,"
            " content_sha, verdict, compliant, findings_json, latency_ms, images, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(dataset_id, episode_index, model, prompt_hash, content_sha) DO UPDATE SET"
            "  verdict=excluded.verdict, compliant=excluded.compliant,"
            "  findings_json=excluded.findings_json, latency_ms=excluded.latency_ms,"
            "  images=excluded.images, created_at=excluded.created_at",
            (dataset_id, episode_index, cfg.get("model"), cfg.get("base_url"), cfg.get("mode"),
             p_hash, content_sha, result.get("verdict"), int(bool(result.get("compliant"))),
             json.dumps(result, ensure_ascii=False, default=str),
             result.get("latency_ms"), result.get("images"), db.now()))
    except Exception:  # noqa: BLE001 - 缓存写失败不该影响质检结论
        pass


# --------------------------------------------------------------------------- 对外主函数


def run_episode(ds_path, meta: dict, episode_index: int, cfg: dict | None = None,
                dataset_id: int | None = None) -> dict:
    """对一条 episode 跑 VLM 质检，返回完整结果（含 findings 与描述）。"""
    cfg = cfg or load_config()
    p_hash = prompt_hash(cfg)

    image_meta = build_images(ds_path, meta, episode_index, cfg)
    if not image_meta["images"]:
        return {"episode_index": episode_index, "source": "none", "cached": False,
                "images": 0, "sha": image_meta.get("sha") or "", "tiles": 0,
                "checklist": [], "describe": {}, "latency_ms": 0,
                "note": image_meta.get("note") or "没有可用的视觉数据",
                "findings": [_finding("VLM·无可检视觉", "info",
                                      image_meta.get("note")
                                      or "该 episode 没有 mp4 也没有内嵌图像，VLM 无法质检",
                                      measured="0 帧", expected="至少 1 路可用视觉")]}

    cached = _cache_get(dataset_id, episode_index, cfg, p_hash, image_meta["sha"])
    if cached and isinstance(cached.get("result"), dict):
        result = cached["result"]
        result["cached"] = True
        result["episode_index"] = episode_index
        return result

    prompts = build_prompts(cfg, image_meta, task_text_of(meta, episode_index))
    if not prompts:
        raise ValueError("vlm_config.json 的 mode 与 checklist 都不产生任何提示词，"
                         "请检查 mode（checklist/describe/both）与 checklist 配置")

    result: dict = {"episode_index": episode_index, "model": cfg.get("model"), "mode": cfg.get("mode"),
                    "source": image_meta["source"], "images": len(image_meta["images"]),
                    "tiles": image_meta["tiles"], "size": image_meta.get("size"),
                    "cameras": [c["key"] for c in image_meta["cameras"]],
                    "task_text": task_text_of(meta, episode_index),
                    "sha": image_meta["sha"], "cached": False, "checklist": [], "describe": {},
                    "findings": [], "latency_ms": 0}
    raw_all: dict = {}

    if "checklist" in prompts:
        reply = chat(cfg, image_meta["images"], prompts["checklist"]["text"])
        result["latency_ms"] += reply["latency_ms"]
        raw_all["checklist_text"] = reply["text"][:2000]
        checks, findings = _parse_checklist(cfg, prompts, reply["json"], reply["text"])
        result["checklist"] = checks
        result["findings"].extend(findings)

    if "describe" in prompts:
        reply = chat(cfg, image_meta["images"], prompts["describe"]["text"])
        result["latency_ms"] += reply["latency_ms"]
        raw_all["describe_text"] = reply["text"][:2000]
        describe, findings = _parse_describe(cfg, reply["json"])
        result["describe"] = describe
        result["findings"].extend(findings)
        if describe.get("description") and cfg.get("emit_description", True):
            result["findings"].append(_finding(
                "VLM·场景描述", "info",
                describe["description"] + (f"（建议指令：{describe['suggested_instruction']}）"
                                          if describe.get("suggested_instruction") else ""),
                measured=describe.get("verdict"), expected="作为人工复核与语言标注的依据"))

    raw_all["prompt_hash"] = p_hash
    result["raw"] = raw_all

    # 跨模式一致性：实测两种模式会对同一条数据给出相反结论
    # （判别式"没完成"、描述式"pass"）。矛盾时不能替人二选一，标出来送人工。
    uncertain_sev = cfg.get("uncertain_severity") or "info"
    done_item = next((c for c in result["checklist"]
                      if c.get("polarity") == "positive"
                      and "任务完成" in (c.get("rule") or "")), None)
    describe = result.get("describe") or {}
    describe_done = _as_bool(describe.get("task_completed"))
    if (done_item and done_item.get("answer") in ("yes", "no")
            and done_item.get("agreement") is not False        # 正反不一致的已在别处报过
            and describe_done is not None):
        checklist_done = done_item["answer"] == "yes"
        if checklist_done != describe_done:
            result["findings"].append(_finding(
                "VLM·两种模式结论矛盾", uncertain_sev,
                f"判别式认为任务{'已完成' if checklist_done else '未完成'}，"
                f"描述式认为任务{'已完成' if describe_done else '未完成'}"
                f"（verdict={describe.get('verdict')}）—— 两种模式对同一条数据判断冲突，"
                f"建议人工复核。描述式理由：{describe.get('reason') or '无'}",
                measured=f"判别 {'yes' if checklist_done else 'no'} / 描述 {describe_done}",
                expected="两者一致"))

    result["verdict"] = "fail" if any(f["severity"] == "error" for f in result["findings"]) else (
        "suspect" if any(f["severity"] == "warning" for f in result["findings"]) else "pass")
    result["compliant"] = result["verdict"] == "pass"
    _cache_put(dataset_id, episode_index, cfg, p_hash, image_meta["sha"], result)
    return result


def run_episode_safe(ds_path, meta: dict, episode_index: int, cfg: dict | None = None,
                     dataset_id: int | None = None) -> dict:
    """`run_episode` 的降级包装：**任何异常都转成一条 warning，绝不向外抛**。

    VLM 服务没起来、模型名写错、超时、显存不够……都不该让整个数据集的质检失败 ——
    结构规则必须照常出结论，VLM 那部分如实标注"没跑成"。
    """
    cfg = cfg or load_config()
    try:
        return run_episode(ds_path, meta, episode_index, cfg, dataset_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "episode_index": episode_index, "source": "error", "tiles": 0, "images": 0,
            "sha": "", "cached": False, "latency_ms": 0, "checklist": [], "describe": {},
            "verdict": "suspect", "compliant": False, "error": f"{type(exc).__name__}: {exc}",
            "findings": [_finding("VLM·服务不可用", "warning",
                                  f"VLM 质检未能完成（{type(exc).__name__}: {exc}），"
                                  f"本条只跑了结构规则；修好后重新质检即可",
                                  measured=cfg.get("base_url"), expected="可达且模型可用")],
        }


def check_episode_vlm(ds_path, meta: dict, episode_index: int, cfg: dict | None = None,
                      dataset_id: int | None = None) -> list[dict]:
    """只返回 findings（结构与 `lerobot.check_episode` 一致），供需要轻量结果的调用方使用。"""
    return run_episode_safe(ds_path, meta, episode_index, cfg, dataset_id).get("findings") or []


def summary_line(result: dict) -> str:
    """把一条结果压成一行，便于日志/终端输出。"""
    parts = [f"ep{result.get('episode_index')}"]
    if result.get("cached"):
        parts.append("缓存")
    parts.append(f"{result.get('tiles')} 格/{result.get('source')}")
    for item in result.get("checklist") or []:
        if item.get("polarity") == "positive":
            parts.append(f"{item.get('rule')}={item.get('answer')}")
    describe = result.get("describe") or {}
    if describe:
        parts.append(f"描述判定={describe.get('verdict')}")
    parts.append(f"{result.get('latency_ms')}ms")
    return "  ".join(parts)
