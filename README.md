#


cd C:\Users\ll\Desktop\HKC_Data_Platform
.\start.cmd                 Window上用一键启动
.\start.ps1                 # 本机访问（默认）
.\start.ps1 -Lan            # 允许局域网/手机访问（会打印你的 IP）
.\start.ps1 -Port 8080      # 换端口
.\start.ps1 -NoBrowser      # 不自动开浏览器
# 跑平台的窗口 Ctrl+C，然后
cd C:\Users\ll\Desktop\HKC_Data_Platform
$env:PYTHONPATH = "$PWD\.vendor"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
#以上3条也为启动途径


# HKC · VLA 数采平台

面向 **VLA（Vision-Language-Action）模型**的数据采集平台。以 LeRobot v2.1 数据集格式为数据契约，
把「采集接入 → 预处理 → 质检 → 标注 → 导出」做成了**真实在跑的任务流**：数据从磁盘上的真实数据集读取，
所有结论都是算出来的，产物落盘可下载，每一次操作都留下版本与血缘。

## 现在什么是真的

| 环节 | 真实行为 | 产物 |
| --- | --- | --- |
| **采集接入** `/collect` | 扫描磁盘上的 LeRobot 数据集（`meta/info.json`），逐条解析 parquet 与视频头部，登记入库 | `datasets` / `episodes` 表 |
| **原始轨迹转换** `/collect` | 把机器人**自己采集**的关节轨迹（`timestamp + joint_1..joint_7`，RealMan 7 轴 ≈10Hz）转成 LeRobot v2.1：`observation.state[7]`、`action[7]`（取下一帧）、相对时间戳、`next.done`，并直接接入 | `storage/datasets/<name>/` |
| **实时采集会话** `/collect` | 客户端（真机/回放）流式上报「相机帧 + 关节角 + 主机时间戳」；提交时**直接落成一条真实 LeRobot episode**（含 mp4 编码），每次会话 = 数据集的一个新版本（kind=collect） | `storage/datasets/<name>/`、`collect_sessions` 表 |
| **预处理** `/preprocess` | 按规则筛 episode（质检通过 / 剔除动作全程不变 / 仅成功样本 / 全部），**真实写出**一个新 LeRobot 数据集 | `storage/exports/<name>/` + `.zip` |
| **修正类预处理** `/preprocess` | 对质检发现的坏样本做**真改帧**修复：剔除 NaN/重复时间戳帧、时间戳对齐、按保留帧重建视频（使行数==帧数）、丢弃视频缺失/损坏的 episode、跨 episode 去重 | 新数据集 + `storage/reports/repair_*.json` |
| **质检** `/qc` | 逐条跑 10 类规则（见下），产出真实报告 | `storage/reports/qc_*.json` |
| **标注** `/annotation` | 给 episode 写语言指令（VLA 的「L」）与成功位，落库；原始数据只读 | `annotations` 表 |
| **标注质量抽检** `/reviews` | 可复现抽检（salt 决定批次）→ 盲标复核 → 一致率与 Cohen's Kappa → 争议裁决（结果写回主标注） | `annotation_reviews` / `review_batches` 表 |
| **训练与离线评估** `/training` | 直接解析 LeRobot 训练产物：`train_config.json` + 训练日志指标序列 + `eval_info.json`（pc_success）；登记实验时绑定数据集版本与内容哈希并给出复现命令；评估失败/质检不过/标注失败汇成困难集回灌 | 新数据集（血缘 `badcase:return`）+ 产物 |
| **真实模型推理** `/models` | 权重从磁盘扫描，推理在 `envs\yolov8n`（torch 2.11+cu128 / RTX 5070）里跑，平台只做代理；返回**真实检测框 + 真实耗时 + ultralytics 画的标注图** | 推理服务 `tools/infer_service.py` |
| **系统日志** `/log` | 展示平台**真实做过的操作**（来自 `audit_log`：任务/版本/质检/标注/复核/采集/训练），可按级别与模块筛选 | `audit_log` 表 |
| **场景管理** `/scenes` | 按任务聚合真实覆盖度（数据集/episode/帧/标注/质检打回/bad case），并发现相似写法做**任务归一**（显式别名，不静默合并） | `task_aliases` 表 |
| **机器人管理** `/robots` | 按本体聚合真实数据 + **设备在环探测**（真的打开相机读一帧、读注册表枚举串口、加载 RealSense SDK） | `device_scans` 表 |
| **数据划分** `/export` | 可复现的 train/val/test 划分（random 按 `sha256(salt:episode)`，ordered 写 LeRobot 原生 splits）+ 按划分导出 | `dataset_splits` 表、`meta/splits.json` |
| **存储清理** `/system` | **计划与执行分离**：先算账（哪些可回收、哪些受保护及原因）再删除；边界写死（不碰 `datasets`、不删被数据集登记引用的目录） | `core/storage.py` |
| **画布类标注** `/annotation` | 目标框（拖拽）/ 关键点（点击）/ 阶段切分（帧区间）；坐标归一化，入库前强校验；导出时写进 `meta/annotations.json` | `spatial_annotations` 表 |
| **跨数据集重复检测** `/data` | 按**内容**（state/action/timestamp 取值）而非文件名找重复，精确/近似分开；按**血缘族**判断是否可解释，跨族共享内容给出数据泄漏告警 | `episode_hashes` 表 |
| **标注账号与工时** `/annotation` | `annotators` 花名册（既有记录自动补登）+ 上工/收工工时；**操作数从审计日志数**，未收工的会话不计入工时 | `annotators` / `work_sessions` / `audit_log.actor` |
| **系统管理** `/system` | 真实运行状态：存储用量、数据规模、推理服务（含 GPU）、最近告警 | 页面实时计算 |
| **导出** `/export` | 导出 LeRobot v2.1 目录（parquet 重编号 + 视频 + `info.json`/`tasks.jsonl`/`episodes.jsonl`/`episodes_stats.jsonl`）并打包；**标注优先写进导出物** | `storage/exports/*.zip` |
| **版本管理** `/data` | 每次接入/转换/导出都登记版本：内容哈希、父版本、操作参数；同一操作 + 同样内容**不新增版本**；预处理产物登记为**新数据集**并记录血缘 | `dataset_versions` 表 |

**任务流基础设施**：SQLite 持久化（`storage/platform.db`，WAL）、后台线程执行、进度上报、前端轮询
（`static/js/platform.js` 的 `runJob/pollJob`）、产物登记与下载（`/api/artifacts/{id}/download`）。

### 质检规则（全部基于真实测量，不是写死的）

长度一致性（parquet 行数 vs `episodes.jsonl` 声明长度）、视频帧数一致性（**VLA 最常见的坏样本**）、
视频可解码性、文件完整性（缺 parquet / 缺视频）、帧率一致性、特征维度一致性（对比 `info.json` 的
`features.shape`）、数值有效性（NaN/Inf）、时间戳单调性与步长、动作有效性（action 全程不变）、
视觉输入完整性（数据集没有视频特征时提示「VLA 缺少视觉观测」）。
任一 error → 结论为「打回重采」。

### VLM 内容质检（Qwen3-VL，可选）

结构规则只能查「文件对不对」，查不出**内容对不对**：遥操人员没放成功、重采后忘了删、
画面被遮挡、多了一只手入镜 —— 这些在文件层面完全正常。

`/qc` 页面的「🧠 VLM 内容质检」按画面判断，两种模式（详见
[docs/07-VLM质检方案.md](docs/07-VLM质检方案.md)）：

- **checklist 判别式**：预设问题 + yes/no，快、便宜、好校准，但只能查预设的问题。
  每条问题配一条**反向问法**，两次矛盾判为「待人工确认」，用来抵消 VLM 的 yes 偏置。
- **describe 描述式**：先客观描述「初始状态/动作/结束状态」再给判定，属开放集，
  能发现没预设的问题；描述里顺带产出的**建议指令可直接回填成 VLA 语言标注**。

关键设计：

- 结论写进**同一张 `qc_findings`**（rule 以 `VLM·` 开头），所以
  「打回重采 / 按质检通过导出 / 修正类预处理」全部不用改。
- VLM 判定**只落 warning/info，永远不产生 error** —— 打回重采的决定权留给人。
- `vlm_checks` 表按（内容指纹 + prompt 指纹）缓存，**重复质检不重复烧 GPU**。
- 后端只要 OpenAI 兼容接口（ollama / llama.cpp / vLLM / 云端 API 都行），
  平台侧不需要装 transformers、不需要 CUDA。
- **必须先校准**：`tools/vlm_calibrate.py` 用人工标注集算出查准率/查全率/混淆矩阵，
  没做过校准的结论不能用于打回重采。

```powershell
# 起后端（示例：ollama，Q4 量化约 3.3GB，8GB 显存够用）
ollama pull qwen3-vl:4b

# 自检（自带 mock，不需要模型、不碰真实数据）
python vlm_selftest.py

# 校准：先生成标注模板，人工填 ok 后计算指标
python tools\vlm_calibrate.py --dataset 1 --dump-template labels.json
python tools\vlm_calibrate.py --dataset 1 --labels labels.json
```

## 实时采集（真机 / 回放）

采集客户端把相机帧与关节角流式上报，**提交即落成一条真实 episode**：

```bash
# 真机（需要 pyrealsense2 与 RealMan SDK，在装好驱动的环境里运行）
python tools/collect_client.py realtime --dataset realman7-grasp \
    --task "抓取红色连接器并插入左侧插槽" --seconds 6 --success

# 回放已有数据集（不需要硬件，用来验证链路）
python tools/collect_client.py replay --dataset collect-replay \
    --task "推动 T 形块到 T 形目标上" --success \
    --from "C:\Users\ll\Desktop\VLA\lerobot\.hf_cache\lerobot\lerobot\pusht" --episode 0
```

接口链路：`POST /api/collect/session` → `POST /api/collect/session/{id}/frames` → `POST /api/collect/session/{id}/commit`。

**时间同步契约**：客户端只用主机单调时钟给每帧打 Unix 秒时间戳，不做对齐；
服务端提交时校验单调性与步长偏差并写进结果，抖动会在 `/qc` 里被「时间戳步长」规则报出。

**实测**（回放 pusht episode 0）：161 帧真实视频帧上报 → episode **161 行 / 161 视频帧 / 时长 16.0s**，
自动接入为数据集 v1（hash 已登记），质检 **0 error**，mp4 可播放。
时间同步体检：正常会话步长偏差 0.0；故意插入 0.5s 跳变 → 5.0 → 命中「时间戳步长」。

**增量登记**：采集每次只探测新增的那条 episode，不再全量重扫。
实测（206 条的数据集）：**全量 4.43s → 增量 0.23s（19x）**；指定单条（会话提交用法）0.27s。

**真机部分尚未接完**：`run_realtime()` 里的 `read_joints()` 是刻意留出的钩子（调用会明确报错提示），
需要在实际设备环境里接上 RealMan SDK（参考 `dhz\realman_ctrl.py`）；
相机侧：**`pyrealsense2` 2.58.2 已在 base 环境可用**（`envs\yolov8n` 里没有），但 `query_devices()` 返回 **0 个设备** —— 缺的是硬件而不是 SDK；
采集客户端会把这两种情况分开报告（「SDK 缺失」vs「没检测到相机设备」）。

## 版本管理与血缘

- **版本号**：每个数据集有 `version`，`dataset_versions` 表记录历史
  （版本、类型 ingest/convert/preprocess/export、内容哈希、条数/帧数、父版本、操作参数、时间）。
- **内容哈希**：`meta/` 与 `data/` 按**文件内容**参与 SHA256；`videos/` 只用（相对路径, 字节数），
  避免为了算哈希把上百 MB 视频读一遍。同一数据集重复计算结果一致；内容一变（哪怕只改一条标注）哈希就变。
- **幂等**：去重键是 **(kind, content_hash)**，而不是「最新一版」——因为接入的哈希是源目录、
  导出的是快照，两者会交替出现，只和最新版比会每轮凭空多出一个假版本。
- **血缘**：预处理产物登记为**新数据集**（`parent_dataset_id` + `derived_by`），之后可继续质检/标注/
  再导出；`GET /api/datasets/{id}/lineage` 返回从源头到当前的链，`/derived` 返回它的下游。
- **界面**：`/data` 页底部「版本与血缘」面板，可看版本表、血缘链、派生数据集，并直接下载某个版本的包。

## 修正类预处理（会改帧）

与「按规则筛 episode」不同，这里会**真的改帧**，并把结果登记为新数据集（带血缘）。

| 操作 | 含义 | 默认 |
| --- | --- | --- |
| `drop_broken` | 丢弃视频缺失/损坏的 episode（这类问题无法修，只能剔） | 开 |
| `drop_dup_episodes` | 跨 episode 去重：轨迹（重采样后逐点比较）近似相同的只留最早一条 | 开 |
| `drop_invalid` | 剔除 state/action 里含 NaN/Inf 的帧 | 开 |
| `dedup_ts` | 剔除时间戳重复或回退的帧 | 开 |
| `align_ts` | 时间戳对齐到 1/fps 均匀网格（不改帧数） | 开 |
| `resample` | 按 fps 线性插值到目标长度（**会生成中间帧**） | 关 |
| `sync_video` | 按保留帧重建视频，使「行数 == 视频帧数」重新成立 | 开 |

关键设计：修复后 `meta/episodes.jsonl` 的 `length` 会按**修复后的真实长度**重写，
所以「长度一致性」检查会通过；视频统一重建为 mp4v 并同步更新 `video_info.video.codec`，
避免同一数据集里编码混杂。**不会凭空造数据**：默认只删不补；要补回声明长度需显式启用 `resample`。

实测（受控夹具 + 真实坏样本集）：

```
A 受控夹具（NaN×3 + 重复时间戳 + parquet 截短）: 3 error -> 0 error
      ep0 161->158 行（剔掉 NaN 与重复戳），视频 161->158 帧
      ep1 108->108 行，视频 118->108 帧（以 parquet 为准重建）
B 重采样变体（启用 resample）  : ep1 108->118 行（补回声明长度），视频 ->118 帧 -> 0 error
C 真实坏样本集 pusht-broken: 4 error -> 0 error
      丢弃 ep1（视频缺失）、ep3（视频损坏），ep2 视频 141->131 帧
```

## 标注质量抽检

四步闭环：**可复现抽检 → 盲标复核 → 一致性指标 → 争议裁决**。

- **可复现抽样**：按 `sha256(dataset:episode:salt)` 排序取前 N 条，所以同一个 salt 抽出的结果稳定、
  换 salt 就是下一轮（批次记录在 `review_batches`，可审计）。
- **盲标**：待复核队列只给原始指令与视频，**不返回主标结果**，避免影响复核判断。
- **一致性指标**（成功位二分类）：
  `po=(a+d)/n`、`pe=((a+b)/n)((a+c)/n)+((c+d)/n)((b+d)/n)`、`kappa=(po-pe)/(1-pe)`，
  其中 a=双方成功 b=仅主标成功 c=仅复核成功 d=双方失败；另给出指令文本一致率。
  Kappa 分级：≥0.8 几乎完全 / ≥0.6 高度 / ≥0.4 中等 / ≥0.2 一般 / 否则「一致性差，需重训标注规范」。
- **争议裁决**：成功位或文本不一致的 episode 进入争议队列，可裁决为
  `primary` / `review` / `custom`；最终标签写回主标注表，复核记录标记为已解决。

实测（构造已知 2×2 与手算对照）：

```
构造 a=3 b=1 c=1 d=0（另有 1 条仅文本不一致）
   po=0.6（手算 0.6） pe=0.68（手算 0.68） Kappa=-0.25（手算 -0.25）
   文本一致率=0.8 争议=3 未解决=3   结论：与手算完全一致
裁决一条（采纳复核）：主标被改写、该复核 resolved=1、争议 3 -> 2
抽样可复现：salt=round-1 两次都得 [2,0,4]；salt=round-2 得 [4,2,3]
无标注的数据集抽检：正确报错「该数据集还没有任何标注」
```

## 训练与离线评估（真实产物）

`/training` 直接解析本机 LeRobot 训练产物，**不编造指标**：

- **产物扫描**：找 `checkpoints/<step>/pretrained_model/train_config.json`（干净 JSON）与 `*_log.txt`
- **训练指标**：从日志行 `step:... smpl:... ep:... epch:... loss:... grdn:... lr:... updt_s:...`
  解析成序列落库（页面画 loss 曲线）
- **版本复现**：登记实验时绑定平台数据集的 `version` + `content_hash`，并按配置重建复现命令
- **离线评估**：解析 `eval_info.json` → `pc_success` / `avg_sum_reward` / `avg_max_reward` / 逐 episode 结果
- **bad case 回灌**：评估失败、质检不通过、标注失败的样本统一进队列；「回灌」把它们汇成一个
  **困难集数据集**（带血缘 `badcase:return`），可直接再训练

实测（本机 `VLA\lerobot\lerobot_project\outputs`）：

```
实验: pusht_train_20k (diffusion)  steps=100000  bs=16  lr=1e-4  seed=1000  指标点 27
      绑定数据集 #1 pusht-real v2  hash=37c0751f85...
评估: 11-25-31_pusht_diffusion  20 条  pc_success=15.0%   avg_max_reward=0.7483
      pusht_eval                 5 条  pc_success=0.0%    avg_max_reward=0.0495
bad case: 未处理 24（eval 22 / annotation 4 / qc 3），回灌后 5 条转为 returned
困难集: #26 pusht-broken_hardcases  3 条 episode / 408 帧  血缘 badcase:return -> #2
```

> 注意：`checkpoints/last` 在 Windows 上是**断链的符号链接**，平台的目录遍历做了健壮处理（跳过坏链与无权限目录）。

## 真实模型推理（GPU）

`/models` 已从「前端随机画框」换成**真实推理**。为什么是独立进程：
本机 **base 环境的 torch 是坏的**（`torch_cuda.dll` 加载失败），而 `envs\yolov8n` 里 torch/GPU 正常且**没有 fastapi**。
所以推理服务用**纯标准库** `http.server` 写，零新增依赖；平台侧通过 `/api/inference/*` 代理。

启动（项目根目录；页面在服务未启动时会直接把这条命令显示出来）：

```powershell
& "C:\Users\ll\miniconda3\envs\yolov8n\python.exe" tools\infer_service.py `
    --model "C:\Users\ll\yolo26n.pt" --port 8100 `
    --scan-dirs "C:\Users\ll;C:\Users\ll\Desktop\dhz\model"
```

接口：`GET /health`、`GET /models?probe=1`、`POST /predict {image_b64, model, conf, render}`
→ 真实检测框（类别/置信度/bbox）+ 端到端耗时 + ultralytics 分项耗时 + 渲染图 base64。
只允许加载 `--scan-dirs` 下的权重，且只监听 127.0.0.1。

实测（本机）：

```
/health : device=NVIDIA GeForce RTX 5070 Laptop GPU  cuda=True  torch=2.11.0+cu128  ultralytics=8.4.75
/models : 11 个真实权重（yolo26n.pt 80 类；best_0/7/all/best.pt 2 类；best_14/new 1 类；best_wb.pt = ['w','b']）
          dhz\hkc\best.pt 加载失败（是 YOLOv5，装着的 ultralytics 拒绝）——界面如实标示
/predict: 真实视频帧 1280x720 -> 4 个目标：person 0.767、cup 0.482、dining table 0.434、cup 0.379
          端到端 40.2ms（分项 preprocess 5.71 / inference 38.55 / postprocess 28.72），标注图 199KB
```

> 工业图片（如 `dhz\hkc\391.jpg`）用 COCO 模型检不到目标是正常的；
> 界面会如实显示「该模型在这张图上没有检出目标」，不会造假。

## 审计日志与系统状态

- **`/log` 是真实审计**（原来是随机生成的 mock 日志）。
  平台在这些地方写入 `audit_log`：任务创建/完成/失败、数据集接入与版本登记、
  质检结果、修正处理、采集会话提交、标注保存、复核与争议裁决、实验登记与 bad case 回灌。
  接口形状保持不变（`/log/api/logs?level=&module=&limit=` → `{code,data,total,modules,stats}`）。
- **历史回填**：引入审计前平台已跑过大量真实操作，`audit.backfill_if_empty()` 从既有表（jobs、
  dataset_versions、qc_runs、collect_sessions、experiments、annotations、annotation_reviews、bad_cases）
  按**原时间戳**回填了 187 条，不编造。
- **`/system` 是真实运行状态**（原来是写死的日志列表）：Python/操作系统、数据库路径与大小、
  `datasets/exports/reports/sessions/testdata` 的文件数与占用、各表行数、推理服务状态（含真实 GPU 名）、最近操作与告警。

实测：

```
/log    : 累计 196 条（ERROR 1 / WARNING 31 / INFO 164），14 个真实模块，
          按模块/级别筛选可用；limit=-1 -> 0 条（旧版会返回 99 条，已修）
/system : 含真实存储用量、各表行数、推理服务 NVIDIA GeForce RTX 5070 Laptop GPU
```

## 数据划分与统计严谨性

**为什么要做**：实测本机所有数据集的 `info.json` 里 `splits` 都是 `{"train": "0:N"}`
—— 训练与评估没有隔离，**评估结果无法复现**。

- **`random`**：归属由 `sha256(f"{salt}:{episode_index}")` 决定，**只靠 salt 就能复现**；
  写 `meta/splits.json`（LeRobot 的 splits 字段只能表达连续区间，装不下随机划分）。
- **`ordered`**：按顺序切连续区间，同时写进 `info.json["splits"]` —— **LeRobot 原生格式**。
- **按划分导出**：`/api/jobs/export` 支持 `mode=split`，只导出某个划分的 episode（产 zip + 登记带
  `split` 参数的版本）。
- **只读数据源**：数据集可能位于只读位置或工作区之外（例如 HuggingFace 缓存），
  元数据写入是“尽力而为”，**划分的权威记录在平台库**，写不进去会如实告警且不影响导出。

**划分绑定到实验**：登记实验时可指定用哪个划分（默认 `train`），平台把 split / salt / episode 数写进实验台账，并把它写进复现命令；划分名不存在会直接报 400。`/api/training/compare/{id}` 还会判断**同一数据集上的多个实验能不能直接比指标**（只有内容哈希与 split+salt 都相同才可比）。

**Kappa 置信区间**：`/reviews` 的一致性指标现在同时给出 **bootstrap 95% CI**（固定种子 →
同一份数据每次得到完全相同的区间），并标出样本量 < 30 时仅作参考。

实测：

```
划分（206 条）: salt=v1 -> train 162 / val 22 / test 22（80/10/10 目标）
          同 salt 两次划分完全一致；换 salt 后 64/206 条改变归属
ordered（14 条）: info.json splits = {'train': '0:9', 'val': '9:11', 'test': '11:14'}（LeRobot 原生）
按划分导出: mode=split / split=test -> 22 条 / 2599 帧 / zip 677KB + 版本 params 含 split
Kappa CI: 完全一致 40 对 -> [1.0, 1.0]；两组独立 60 对 -> 0.0455 [-0.1977, 0.3033]（区间跨 0）
          真实对比数据集 #2（配对 6）-> 0.3333 [0.0, 1.0]，reliable=False（如实标不可靠）
```

## 跨数据集重复检测

同一批 episode 在多个数据集里各存一份是数据平台很容易踩的坑：
有的是**有意的派生**（preprocess / repair / export 的产物，血缘里写得清楚），
有的是**意外重复**（同一批数据被接入两次）。

- **按内容而非文件名**：`content_sha` 哈希 state/action/timestamp 的**取值**（浮点先归一，
  绕开 parquet 写入差异）；`near_sha` 均匀抽 16 行取 3 位小数，用于**近似**重复
  （例如 repair 掉了两帧，长度 161→159 但采样行一致）。
- **区分「内容相同」与「字节相同」**：实测这些重复都是**内容相同、文件字节不同**，
  所以按文件哈希会全部漏报。
- **按血缘族（连通分量）判定**：先把 `parent_dataset_id` 连成的图求连通分量，
  同族内的重复 = 可解释的派生；**跨族共享内容** 才是真问题。
  （最初写成“组内只要有一对有血缘就算可解释”，结果把真问题藏进了“派生重复”里；
  后来又改成逐对判，但传递性关系导致一组冒出 7 条假告警——最终按族判才对。）
- **平台只报告，不自动删**：删数据不可逆，保留哪份必须人来定。指纹缓存在 `episode_hashes`，重扫不重算。

实测（655 条 episode / 9 个数据集，扫描 5.7s ≈ 9ms/条；第二次 0.03s 命中缓存）：

```
精确重复 208 组 / 近似重复 1 组
判定: 同数据集内 1 组 · 派生（同一血缘族） 203 组 · 跨血缘族 5 组
血缘族: {#1,#21,#23} · {#2,#22,#26} · {#5} · {#24} · {#25}

真实发现：族[#1 pusht-real、#21 ver_pre_1、#23 pusht-real_fixed]
           与 族[#2 pusht-broken、#22 pusht-broken_fixed、#26 pusht-broken_hardcases]
           共享 5~6 条完全相同的 episode（长度 118/131/141/159/161），但血缘上互不相关
           —— 同时用于训练/评估会造成数据泄漏
```

> 说明：本机的 `pusht-broken` 族是早期为测试而从 `pusht-real` 拷贝出来的夹具（部分 episode 未改动），
> 所以“共享内容”是夹具造成的。但检测器把它抓出来正是它该做的事：真实数据里同样形态的重复
> 一旦发生，就是训练/评估泄漏。

## 数据契约：LeRobot v2.1

实测的 `meta/info.json`：

```
meta/info.json       codebase_version / fps / total_episodes / total_frames / splits
                     data_path  = data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
                     video_path = videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
                     features: observation.image(video 96x96x3) / observation.state[2] / action[2]
                               episode_index / frame_index / timestamp / next.reward / next.done
                               / next.success / index / task_index
meta/tasks.jsonl     {"task_index":0,"task":"Push the T-shaped block onto the T-shaped target."}
meta/episodes.jsonl  {"episode_index":0,"tasks":[...],"length":161}
```

这与「数据定义」页（`/semantics`）的 8 个核心字段一一对应：任务描述→`tasks.jsonl`、图像/视频→
`videos/*.mp4`、末端3D轨迹→`action`/`observation.state`、成功/失败→`next.success`、时间戳→`timestamp`、
动作阶段→`next.done`、坐标系→`robot_type`+`features.names`、物体信息→需标注补充。

## 规范文档（全链路第 1、3 项）

| 文档 | 内容 |
| --- | --- |
| [docs/01-采集方案设计.md](docs/01-采集方案设计.md) | 数据类型与规格、采集频率与时长、时间同步方案、场景覆盖矩阵、Episode 结构 |
| [docs/02-采集SOP与判定标准.md](docs/02-采集SOP与判定标准.md) | 角色职责、采前检查清单、单条 episode 采集步骤、成功/失败判定表、10 条边界案例、打回重采流程 |
| [docs/03-标注规范与验收标准.md](docs/03-标注规范与验收标准.md) | 指令撰写模板与用词表、成功位标注规范、抽检与复核机制、Kappa 验收线 |
| [docs/README.md](docs/README.md) | 文档索引 + 「规范要求 ↔ 平台检查点」映射表 + 库表字段对照 |

文档里的每个规则名、阈值、实测数字、接口路径、库表字段都经脚本核对过（**81 项全部命中**），
所以规范不是写在纸上，而是跑在流水线里。

## 快速开始

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

打开 http://127.0.0.1:8000 ，Demo 账号 **admin / admin123**。

第一次使用：进「采集接入」→ ① 填扫描根目录 → 扫描磁盘 → 接入已有 LeRobot 数据集；
② 或用下面的「转换并接入」把机器人自己采的关节轨迹转成 VLA 数据集。之后各环节就有真实数据了。

### 自检

```bash
python smoke_test.py     # 18 个页面 + 平台接口 + 版本/血缘 + 日志接口 + 登录/注销/改密（不需要起服务）
python flow_test.py      # 端到端：采集→质检→标注→导出→版本闭环（需要先起服务，端口用 FLOW_BASE 覆盖）
python vlm_selftest.py   # VLM 质检 41 项断言（自带 mock，不需要模型、不碰真实数据）
```

## 页面

| 路径 | 说明 |
| --- | --- |
| `/dashboard` | 真实统计（数据集/条数/帧数/问题条/已标注/体积）+ 最近任务轮询 |
| `/collect` | **扫描磁盘 + 接入 LeRobot 数据集 + 原始关节轨迹转换**（采集环节） |
| `/data` | 数据集与 episode 明细（行数/视频帧数/时长/成功位/问题数）+ 视频预览 + **版本与血缘** |
| `/preprocess` | **按规则生成子集数据集**（真产物，登记为新数据集并建立血缘） |
| `/qc` | **执行质检**（13 条结构规则 + 可选 **VLM 内容质检**）+ 记录 + 问题明细 + 报告 |
| `/annotation` | **语言指令与成功位标注** + 视频预览 |
| `/reviews` | **标注抽检**：抽样批次 + 盲标复核 + 一致性指标 + 争议裁决 |
| `/export` | **导出 LeRobot 数据集** + zip 下载（标注写进导出物，登记为新版本） |
| `/semantics` | 统一语义层与 Episode 结构（设计说明） |
| `/models` `/training` `/interfaces` `/robots` `/scenes` `/system` `/log` | 见下方「仍是壳的部分」 |

## 目录结构

```
main.py              入口：建库、注册路由、注入当前用户
core/db.py           SQLite 持久化（datasets/episodes/jobs/qc_*/annotations/artifacts/dataset_versions）
core/lerobot.py      LeRobot v2.1 读写 + 质检规则 + 内容指纹（只读原始数据）
core/raw_hkc.py      机器人自有原始关节轨迹 → LeRobot v2.1 适配器
core/repair.py       修正类预处理（改帧：删帧 / 时间戳对齐 / 重建视频 / 跨 episode 去重）
core/reviews.py      标注质量抽检（抽样批次 / 交叉复核 / Kappa / 争议裁决）
core/session.py      实时采集会话 → LeRobot episode（含 mp4 编码）
core/training.py     训练产物解析 / 实验台账 / 离线评估 / bad case 回灌
core/audit.py        审计日志（真实操作记录 + 历史回填）
core/splits.py       可复现的 train/val/test 划分（random / ordered）
core/storage.py      存储清理（计划/执行分离 + 安全边界）
core/spatial.py      画布类标注（框/关键点/阶段）+ 入库校验
core/dedup.py        跨数据集重复检测（内容指纹 + 血缘族判定）
core/annotators.py   标注账号花名册 + 工时（操作数取自审计）
tools/collect_client.py 采集客户端（真机 realtime / 回放 replay）
core/jobs.py         后台任务引擎（线程 + 进度 + 结果落库）
core/pipeline.py     六个真实处理器：ingest / convert / preprocess / qc / repair / export（含版本与血缘登记）
routers/platform.py  平台自身 JSON 接口（/api/*）
routers/reviews.py   标注抽检页 + /api/reviews/*
routers/collect_api.py 采集会话接口（/api/collect/*）
routers/training_api.py 训练与评估接口（/api/training/*、/api/badcases/*）
routers/models_api.py  真实推理代理（/api/inference/*）
tools/infer_service.py 推理服务（纯标准库，在 yolov8n 环境跑）
routers/collect.py   采集接入页
static/js/platform.js 前端公共工具（API/任务轮询/格式化）
smoke_test.py        冒烟测试
flow_test.py         端到端活体测试
docs/                采集方案设计 / 采集SOP与判定标准 / 标注规范与验收标准
storage/             platform.db、datasets/、reports/、exports/、testdata/（全部为运行产物，可删）
```

## 本机实测的数据集

| 数据集 | 来源 | 内容 |
| --- | --- | --- |
| `pusht-real` | 本机 LeRobot 缓存 | 206 条 / 25650 帧 / av1 96x96 视频 + 2 维 state/action，质检全部通过 |
| `pusht-broken` | **故意做坏的测试夹具** | 5 条，3 条被打回重采（缺视频 / 截短 parquet / 视频损坏） |
| `hkc-joint-7dof` | **用户自己采的关节轨迹转换而来** | 14 条 / 3088 帧 / fps 9.936 / 7 维 state+action，质检通过（提示缺视觉观测） |
| `ver_pre_1` | 由 `pusht-real` 经 `preprocess:qc_pass` 派生 | 血缘：`pusht-real → ver_pre_1` |

## 剩下的限制（如实说明）

写死的壳页面**已经没有了**（`/robots` 与 `/scenes` 本轮接真，19 个页面全部是真实数据）。
仍然存在的限制是**能力/环境性**的，不是“没实现”：

- **真机硬件不在机**：`/robots` 的设备探测实测为「相机 0 个 / 串口 0 个 / RealSense SDK 可用（2.58.2）但设备 0 个」。
  采集的软件链路已用真实视频帧回放验证，接上设备即可用；机械臂侧 `read_joints()` 仍是留好的钩子。
- **推理服务需单独启动**（另一个 Python 环境，torch 在 base 是坏的）。
- **策略在环评估未做**：`/training` 的离线指标读的是 LeRobot 自己产出的 `eval_info.json`（真实文件），
  平台不会自己跑策略 rollout。
- **画布类标注已提供**（目标框/关键点/阶段），但只支持**矩形框**（无淹模/多边形），且没有逐帧追踪；标注账号与工时已有，但**不是认证系统**（登录仍是 demo 级）。
- **Kappa 只有点估计**，没有置信区间；样本量 <30 时只作参考。
- **任务归一是人工显式的**（近文本提示 + 一键合并），跨语言（英文元数据 vs 中文标注）不自动配对。
- **重复检测只到内容/采样级**：做不到语义级（“两条轨迹很像但不一样”）的相似度；且**不自动删**，只报告。
- **存储清理需人工触发**：`/system` 可算账并清理（实测释放 54.3MB / 2588 文件），但**没有定时策略或自动阈值** —— 什么时候清由人决定。

## 本机免安装运行（可选）

工作区里的 `.vendor/` 是这套依赖的免安装副本，由 `_fetch_deps.py` 生成
（`python _fetch_deps.py [额外包名...]` 可重建）。正常 `pip install -r requirements.txt` 时用不到它：

```powershell
$env:PYTHONPATH = "$PWD\.vendor"
python smoke_test.py
```
