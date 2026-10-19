"""内嵌图像（LeRobot `dtype: image` 存在 parquet 列里）的读取。

LIBERO 这类数据集把图像以 `struct<bytes: binary, path: string>` 直接存在 parquet 里
（HuggingFace Image 特征的原样序列化），**没有 mp4**。平台的视频预览 / 取帧原来只认 mp4，
所以这类数据集点「预览」什么都没有。这里补上从 parquet 直接解码取帧。
"""
from __future__ import annotations

from pathlib import Path

from . import lerobot


def image_keys(info: dict) -> list[str]:
    """dtype == image 的特征（相机）。"""
    return [k for k, spec in (info.get("features") or {}).items()
            if isinstance(spec, dict) and spec.get("dtype") == "image"]


def _decode(raw: bytes, shape: list | None):
    """把 parquet 里存的图像解码成 BGR ndarray；支持容器格式与裸 HWC 数组。"""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        return image                      # PNG / JPEG 等容器
    if shape and len(shape) == 3:         # 裸 HWC uint8
        array = np.frombuffer(raw, np.uint8)
        if array.size == shape[0] * shape[1] * shape[2]:
            return array.reshape(shape)
    return None


def frame_bytes(ds_path: str | Path, meta: dict, episode_index: int,
                frame_index: int = 0, key: str | None = None) -> tuple[bytes, str, str]:
    """取某帧的图像字节。返回 (数据, content-type, 相机 key)。"""
    import cv2
    import pyarrow.parquet as pq

    info = meta["info"]
    keys = image_keys(info)
    if not keys:
        raise ValueError("这个数据集没有 dtype=image 的特征（不是内嵌图像数据集）")
    key = key or keys[0]
    if key not in keys:
        raise ValueError(f"没有这个相机：{key}（可用 {keys}）")

    paths = lerobot.episode_paths(info, ds_path, episode_index)
    table = pq.read_table(paths["parquet"], columns=[key])
    rows = table.num_rows
    if rows == 0:
        raise ValueError("这条 episode 是空的")
    frame_index = max(0, min(int(frame_index), rows - 1))
    cell = table.column(key)[frame_index].as_py() or {}
    raw = cell.get("bytes") if isinstance(cell, dict) else None
    if not raw:
        raise ValueError(f"第 {frame_index} 帧没有图像字节")

    shape = (info.get("features", {}).get(key) or {}).get("shape")
    image = _decode(raw, shape)
    if image is None:
        # 解不出来就把原始字节回给前端（至少能看到"有数据"）
        head = raw[:4]
        ctype = "image/png" if head[:2] == b"\x89P" else ("image/jpeg" if head[:2] == b"\xff\xd8" else "application/octet-stream")
        return raw, ctype, key
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return buf.tobytes(), "image/jpeg", key


def episode_meta(ds_path: str | Path, meta: dict, episode_index: int) -> dict:
    """这条 episode 有哪些视觉源（给前端决定用 <video> 还是逐帧查看器）。"""
    info = meta["info"]
    videos = lerobot.video_keys(info)
    images = image_keys(info)
    paths = lerobot.episode_paths(info, ds_path, episode_index)
    video_file = next(iter(paths["videos"].values()), None)
    rows = lerobot.probe_episode(ds_path, meta, episode_index).get("data_rows") if False else None
    try:
        rows = lerobot.probe_episode(ds_path, meta, episode_index, with_video=False)["data_rows"]
    except Exception:  # noqa: BLE001
        import pyarrow.parquet as pq

        rows = pq.read_table(paths["parquet"]).num_rows if paths["parquet"].is_file() else 0
    return {
        "episode_index": episode_index,
        "rows": int(rows or 0),
        "video_keys": videos,
        "image_keys": images,
        "has_video": bool(video_file and Path(video_file).is_file()),
        "has_inline_images": bool(images),
        "fps": info.get("fps"),
    }
