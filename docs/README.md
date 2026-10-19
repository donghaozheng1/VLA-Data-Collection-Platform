# HKC · VLA 数采平台 文档索引

面向 VLA（Vision-Language-Action）模型的数据采集与治理规范。三份文档分别对应全链路的
**第 1 项（数据采集方案设计）** 与 **第 3 项（采集流程与规范）**。

| 文档 | 内容 | 对应任务 |
| --- | --- | --- |
| [01-采集方案设计.md](01-采集方案设计.md) | 数据类型、采集频率、时间同步、场景覆盖、Episode 结构 | 第 1 项 |
| [02-采集SOP与判定标准.md](02-采集SOP与判定标准.md) | 采前/采中/采后 SOP，成功/失败/边界案例定义，打回重采 | 第 3 项 |
| [03-标注规范与验收标准.md](03-标注规范与验收标准.md) | 标注对象、指令撰写规范、抽检与一致性验收 | 第 3、6、7 项 |

## 这些文档的原则

**每一条可执行的要求，都能在平台上被机器检查。** 文档里出现的规则名、字段名、阈值，
全部与平台代码一致（`core/lerobot.py` 的质检规则、`core/reviews.py` 的 Kappa 分档、
`core/repair.py` 的修复操作），因此规范不是"写在纸上"，而是"跑在流水线里"。

| 规范要求 | 平台里的检查点 | 位置 |
| --- | --- | --- |
| parquet 行数 == `episodes.jsonl` 的 length | 规则 `长度一致性` | `core/lerobot.py` |
| parquet 行数 == 视频帧数 | 规则 `视频帧数一致性` | 同上 |
| 视频必须能解码 | 规则 `视频可解码性` | 同上 |
| parquet / 视频文件必须存在 | 规则 `文件完整性` | 同上 |
| 视频 fps == `info.json` 的 fps | 规则 `帧率一致性` | 同上 |
| 特征维度 == `info.json` 的 features.shape | 规则 `特征维度一致性` | 同上 |
| 轨迹不得含 NaN/Inf | 规则 `数值有效性` | 同上 |
| 时间戳必须单调递增 | 规则 `时间戳单调性` | 同上 |
| 时间戳步长偏差 < 20% | 规则 `时间戳步长` | 同上 |
| action 不得全程不变 | 规则 `动作有效性` | 同上 |
| 每条 episode 必须有语言指令 | 规则 `语言指令完整性` | 同上 |
| parquet 不能为空（0 帧） | 规则 `空 episode` | `core/lerobot.py` |
| 数据集必须有视觉观测（VLA 前提） | 规则 `视觉输入完整性` | 同上 |
| 标注一致率 / Cohen's Kappa 达标 | `/reviews` 一致性面板 | `core/reviews.py` |
| 坏样本可修可剔（不凭空造数据） | `/preprocess` 修正类预处理 | `core/repair.py` |

## 平台库表与字段（文档口径）

规范里提到的每个概念，在平台里都有确切的存放位置：

| 表 | 关键字段 | 文档里对应什么 |
| --- | --- | --- |
| `datasets` | `version` / `content_hash` / `parent_dataset_id` / `derived_by` | 当前版本 / 内容哈希 / 上游数据集 / 由何操作派生 |
| `dataset_versions` | `version` / `kind` / `content_hash` / `parent_version` | 版本历史：版本号 / 类型（ingest、convert、preprocess、repair、export）/ 哈希 / 父版本 |
| `episodes` | `data_rows` / `video_frames` / `task_text` / `success_count` | 逐条 episode 的真实事实（质检与明细页读它） |
| `qc_runs` / `qc_findings` | `rule` / `severity` / `detail` / `measured` / `expected` | 质检报告与逐条问题（实测值 vs 期望值） |
| `annotations` | `task_text` / `success` / `note` / `annotator` | 主标注：语言指令 / 结果位 / 备注 / 标注员 |
| `annotation_reviews` | `reviewer` / `task_text` / `success` / `resolved` / `resolution_note` | 交叉复核标签与是否已裁决 |
| `review_batches` | `salt` / `requested` / `episodes_json` / `note` | 抽检批次（可复现抽样的依据） |
| `artifacts` | `kind` / `path` / `bytes` | 产物（质检报告 / 修复报告 / 数据集包） |
| `jobs` | `kind` / `status` / `progress` / `result_json` | 任务流（ingest、convert、preprocess、qc、repair、export） |

## 实测基准（2026-10-09，本机）

| 项目 | 实测值 |
| --- | --- |
| 自有 RealMan 7 轴轨迹 | 14 段，29 / 96 / 614 行（min/中位/max），合计 3088 行，采样步长中位 **0.1006s ≈ 9.94 Hz** |
| 数据契约 | LeRobot **v2.1**（`meta/info.json` + `data/chunk-*/episode_*.parquet` + `videos/chunk-*/*.mp4`） |
| 参考数据集（pusht） | 206 条 / 25650 帧 / 10 fps / av1 / 96×96×3，11 个 feature |
| 自有数据集（hkc-joint-7dof） | 14 条 / 3088 帧 / 9.936 fps / 7 维 state+action，**无视频**（缺「V」） |
| 相机 | RealSense **D405** 1280×720（精定位 7–50cm）、**D435** 640×480（全局 30cm–2m），均 30 fps |
| 主机 | RTX 5070 Laptop，torch 2.11.0+cu128（`envs\yolov8n`），CUDA 可用 |
| 质检规则 | 13 类规则（上表），任一 error → 结论「打回重采」 |
| Kappa 分档 | ≥0.8 几乎完全 / ≥0.6 高度 / ≥0.4 中等 / ≥0.2 一般 / 否则需重训标注规范 |

> 本文档中的参数如与实际设备不符，以实测为准并回写本文档；平台侧的阈值改动必须同步改这里。
