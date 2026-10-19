# X-VLA 接入调研报告（含 zhiyan 数据接入改动清单）

> 调研时间：2026-10-14　调研方式：**只读**（未在服务器安装/修改任何东西）
> 服务器：`192.168.123.105`（主机名 `z2g9`，公司服务器，**有生产服务在跑，改动需审批**）
> 本地：`C:\Users\ll\Desktop\HKC_Data_Platform`（VLA 数采平台，Windows + RTX 5070 Laptop 8GB）

> ## ⚠️ 重要：请先读 §十~§十四（同日补充调研）
> **推荐方案已变更**：从"改 X-VLA 自有代码（§五）"改为"**用服务器 `lerobot` 0.4.4 自带的 xvla policy**"。
> 原因：服务器 `lerobot` 环境（torch 2.10+cu128，GPU 可用）**原生集成了 xvla**，且它读的就是 LeRobot v2.1 数据集
> —— **我们的数据格式天生匹配，改动清单从"改 3 个文件"简化为"调训练参数"**。
> 另：服务器 `XVLA` 环境因 torch cu130 与驱动 570 不匹配，**GPU 不可用**（§十）。

---

## 一、三个关键结论（先说最重要的）

| # | 结论 | 影响 |
|---|---|---|
| **1** | **X-VLA 原生支持 LeRobot v2.1 数据集** | 我们平台产出的 `zhiyan_pick_place` **就是 v2.1** → **数据格式不用转换** ✅ |
| **2** | **X-VLA 内置 `joint`（关节空间）动作模式** | 我们的数据是 14 维关节角 → **不需要做正运动学转 ee6d** ✅ |
| **3** | **服务器上已有完整 X-VLA 环境 + 代码 + 权重** | 不用重装、不用重下；但**那套环境当前 GPU 不可用**（见 §7）⚠️ |

**原以为的最大障碍（"必须转成 ee6d 末端位姿"）实际上不存在。**

---

## 二、服务器资源清单（实测）

### 2.1 硬件与系统

| 项 | 值 |
|---|---|
| 主机 / 系统 | `z2g9` / **Ubuntu 22.04.5 LTS** / 内核 6.8.0-90 |
| GPU | **NVIDIA RTX 4090 D，24564 MiB (24GB)**，驱动 570.211.01 |
| Python（系统） | 3.10.12；`pip 22.0.2`；**无 conda（系统级）**|
| 编译器 | gcc 11.4.0，cmake 3.22.1 |
| 磁盘 | `/`（ext4）938G，**已用 98%，仅剩 25G** ⚠️<br>`/externalDisk/data`（**exFAT**）1.9T，**可用 1.8T** ⭐ |
| 外网 | ✅ 通（GitHub / PyPI / hf-mirror 均可访问） |

### 2.2 已有的 X-VLA 相关资产 ⭐

| 资产 | 路径 | 说明 |
|---|---|---|
| **XVLA conda 环境** | `/home/z123/miniconda3/envs/XVLA` | Python 3.10.20、torch 2.13.0+cu130、transformers 4.51.3、accelerate 1.2.1、einops、timm、torchvision 0.28.0 |
| **X-VLA 代码仓库** | `/home/z123/rsg_ws/X-VLA` | `train.py`、`peft_train.py`（LoRA）、`deploy.py`、`models/`、`datasets/`、`evaluation/` |
| **X-VLA 权重（WidowX 版）** | `/externalDisk/data/model/X-VLA-WidowX` | **6.9G**：`model.safetensors`（3356MB，903 张量，0.88B 参数，FP32）+ 代码 + `demo.mp4` |
| 其他模型 | `/externalDisk/data/model/` | OpenVLA-7B（15G）、EmbodiedGen（52G） |
| HF 缓存 | `/home/z123/.cache/huggingface` | **51G** |
| **lerobot 训练管线** | `/home/z123/rsg_train_model/lerobot`<br>`/home/z123/rsg_train_model/lerobot_split_merge_prcessor` | 含数据集拆分/合并处理器 |
| 宇树机器人集成 | `/home/z123/rsg_ws/unitree_lerobot`、`/home/z123/G1/` | G1 人形 + lerobot |

### 2.3 正在运行的服务（**不要碰**）

```
z123  ml_backend.main                      已运行 43 天
z123  uvicorn main:app --port 8005         已运行 43 天
z123  inference_service.main               已运行 43 天（占 510MB 显存）
admin /label-studio/.venv/bin/label-studio 已运行 14 天
Docker: colmap / label_studio / gma_infer
```

### 2.4 本地已有的 X-VLA（已下载并验证）

```
storage\xvla\X-VLA-WidowX\      3,358.6 MB   ✅ 完整（局域网从服务器拷贝，65.6 MB/s）
   model.safetensors  3356.04 MB  903 张量 / 0.880 B 参数 / FP32
   config.json  preprocessor_config.json  tokenizer.json  vocab.json
   modeling_xvla.py  configuration_xvla.py  action_hub.py  transformer.py  modeling_florence2.py ...
验证方式：读取 safetensors 头部 + JSON 解析 config（903 张量、architectures=['XVLA']）
```
> FP32 权重 3.36GB → **本地 8GB 显存可推理**（转 bf16 仅 1.8GB）；训练仍需服务器。

---

## 三、X-VLA 的数据接入机制（读代码得出）

### 3.1 链路

```
train.py / peft_train.py
   └── create_dataloader(metas_path=..., num_actions, action_mode, ...)
         └── datasets/dataset.py :: InfiniteDataReader
               ├── 读元数据 JSON（**可以是目录，递归找 .json**）
               │     ├─ 自有格式： {"dataset_name": ..., "datalist": [...]}
               │     └─ LeRobot v2.1： {"codebase_version": "v2.1", ...}   ← ★ 我们走这条
               │           · root_path 自动推导 = 该 json 往上两级
               │           · 同级读取 episodes.jsonl → 填入 meta["datalist"]
               └── 按 meta["robot_type"] 查注册表 → Handler
                     └── datasets/domain_handler/registry.py :: _REGISTRY
                           └── Handler.iter_episode(...)
                                 → 产出样本 {"language_instruction", "image_input"[V,C,H,W],
                                            "image_mask"[V], "action"[T, dim]}
```

### 3.2 注册表（`datasets/domain_handler/registry.py`）

```python
_REGISTRY = {
    "lift2": LeRobotV21Handler,     # ← "Lerobot (v2.1 - sim)"，现成的 v2.1 处理器 ★
    "libero": LiberoHandler,
    "AGIBOT": AGIBOTLeRobotHandler,
    "robomind-franka": RobomindHandler,
    ... （共 30 个左右）
}
def get_handler_cls(dataset_name):
    return _REGISTRY[dataset_name]      # 严格查表，未注册直接 KeyError
```
**接入点：把我们的 `robot_type`（`zhiyan`）注册到这张表。**（一行）

### 3.3 `LeRobotV21Handler` 的关键实现（`datasets/domain_handler/lerobotv21.py`）

它是一份**需要按自己数据集改的模板**，文件里明确写着 `# adjust this hyper-parameters according to your need`：

```python
CAMERA_VIEW  = ["video.top_camera_view", "video.left_camera_view", "video.right_camera_view"]
ACTION_KEY   = ["action.joints", "action.gripper", "action.base_delta"]   # 12 + 2 + 3
idx_for_delta        = [0..11]
idx_for_mask_proprio = [12..16]

# iter_episode 里：
item         = self.meta["datalist"][traj_idx]      # 来自 episodes.jsonl
episode_index= item["episode_index"]
chunk        = episode_index // meta["chunks_size"]
data_path    = meta["root_path"] + meta["data_path"].format(episode_chunk=chunk, episode_index=...)
images[v]    = read_video_to_frames(meta["root_path"] + meta["video_path"].format(..., video_key=vkey))
data         = read_parquet(data_path)
all_action   = concat([data[k] for k in ACTION_KEY])      # ← 动作列名写死在这里
ins          = item["tasks"][0]                            # ← 语言指令
freq = 30.0                                                # ⚠ 帧率写死 30
```

**注意三个写死项**：相机键名、动作列名、`freq=30`。我们的数据是 **20fps、14 维关节角、相机叫 `observation.images.head` / `hand_right`** → **必须改**。

### 3.4 动作空间（`models/action_hub.py`，注册表式）

```python
@register_action("ee6d")  class EE6DActionSpace   # xyz + 6D旋转 + 夹爪；gripper_idx=(9,19)，ROT_SCALE=10
@register_action("joint") class JointActionSpace  # ★ 关节角 + 夹爪；gripper_idx=(6,13)
                          class AGIBOTEE6DActionSpace
                          class AutoActionSpace    # 自动 padding 到模型维度
```
**`joint` 模式现成可用** → 配置 `action_mode="joint"` 即可（我们的 14 维 → 7 关节 + 手部通道 ≈ 与 `gripper_idx=(6,13)` 的两夹爪布局需要对齐，见 §5）。

### 3.5 本体（domain）与 soft prompt 机制

```python
# config.json: num_domains = 30, len_soft_prompts = 32
models/transformer.py:
    DomainAwareLinear(num_domains)                 # nn.Embedding(num_domains, ...) 用于 fc/bias
    self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
    forward(..., domain_id)                        # 按本体取 soft prompt
datasets/domain_config.py:
    DATA_WEIGHTS   = {...}      # 各数据集采样权重
    DATA_DOMAIN_ID = {...}      # 数据集名 → domain id
```
**X-VLA 的设计就是"新本体 = 新 soft prompt"**（论文卖点）。给 zhiyan 加新本体 = 在 `DATA_DOMAIN_ID` 里给 `zhiyan` 一个 id，并把 `num_domains` 从 30 扩到 31（新增那一行 embedding 从头初始化，其余不动）。

---

## 四、我们的数据 vs X-VLA 要求（差距表）

| 要求 | X-VLA 期望（`LeRobotV21Handler`） | 我们的 `zhiyan_pick_place` | 差距 |
|---|---|---|---|
| 元数据格式 | `codebase_version == "v2.1"` | ✅ v2.1 | **无** ✅ |
| `episodes.jsonl` | 每项含 `episode_index`、`tasks` | ✅ 有 | **无** ✅ |
| `info.json` 模板字段 | `data_path` / `video_path` / `chunks_size` / `total_episodes` | ✅ 全部有 | **无** ✅ |
| 视频 | 每 episode 每相机一个 mp4，按 `video_path` 模板 | ✅ `videos/chunk-000/<key>/episode_XXXXXX.mp4` | **无** ✅ |
| 图像键名 | 写死 `video.top_camera_view` 等 | `observation.images.head` / `hand_right` | ⚠️ **要改** |
| 动作列名 | 写死 `action.joints` / `action.gripper` / `action.base_delta` | `action`（14 维）+ `observation.state`（14 维） | ⚠️ **要改** |
| 帧率 | 写死 `freq = 30.0` | **20 fps** | ⚠️ **要改** |
| 语言指令 | `item["tasks"][0]` | ✅ `"pick_and_place"` | **可用**（但只有 1 条，建议丰富化）|
| 本体注册 | `_REGISTRY["<robot_type>"]` | `robot_type = "zhiyan"` **未注册** | ⚠️ **要加一行** |
| 动作模式 | `action_mode` | 需设 `"joint"` | ⚙️ 配置项 |
| 域 id / soft prompt | `DATA_DOMAIN_ID["zhiyan"]` + `num_domains=31` | 未定义 | ⚠️ **要加** |

---

## 五、改动清单（具体到文件和行）

> 原则：**在服务器上复制一份自己的副本改**（如 `/externalDisk/data/vla_ws/X-VLA-zhiyan`），**不直接改团队的 `rsg_ws/X-VLA`**。

### 改动 1：注册 zhiyan 本体（`datasets/domain_handler/registry.py`）
```python
from .zhiyan import ZhiyanHandler          # 新增

_REGISTRY = {
    ...
    "zhiyan": ZhiyanHandler,               # ← 加这一行
}
```

### 改动 2：写一个 zhiyan 专用 Handler（新建 `datasets/domain_handler/zhiyan.py`）
以 `lerobotv21.py` 为模板，改三处：
```python
class ZhiyanHandler(DomainHandler):
    CAMERA_VIEW = ["observation.images.head", "observation.images.hand_right"]   # 2 路相机
    ACTION_KEY  = ["action"]                    # 我们的 action 是单列 14 维
    idx_for_delta        = []                   # 我们的动作是绝对关节角，不做 delta
    idx_for_mask_proprio = list(range(7, 14))   # 手部 7 维做 proprio mask（可选）
    # 帧率：把 freq=30 改成从 meta["fps"] 读（我们 20）
    # 动作布局：前 7 维 = 右臂关节角(度)，后 7 维 = 手部通道[0,1]
```
> 注意：`JointActionSpace.gripper_idx=(6,13)` 是**两路夹爪**的布局，我们的 14 维是"7 关节 + 7 手部通道"，**需要对齐**（要么按它的布局重排动作列，要么写一个 `@register_action("zhiyan_joint")` 的新 ActionSpace）。
> 另：我们数据里关节角是**度**，模型通常期望弧度/归一化 → **要确认归一化约定**（见 §7 待确认项）。

### 改动 3：注册域 id（`datasets/domain_config.py`）
```python
DATA_DOMAIN_ID = {
    ...
    "zhiyan": 30,          # ← 新增（0~29 已被占用，故用 30）
}
DATA_WEIGHTS = {
    ...
    "zhiyan": 1.0,         # 采样权重
}
```
同时把模型配置 `num_domains` 由 **30 → 31**（`models/configuration_xvla.py` 默认值 + 加载时的 config.json 覆盖）。新增的第 31 行 soft prompt 从头初始化，**这正是 X-VLA 官方设计的"接入新本体"方式**。

### 改动 4：准备训练元数据（**不用改代码**）
`--train_metas_path` 可以指向一个**目录**，递归扫描 `.json`。由于读取器认 `codebase_version == "v2.1"`，**直接指向我们数据集的 `meta/info.json` 即可**：
```
--train_metas_path /externalDisk/data/vla/datasets/zhiyan_pick_place/meta/info.json
# 读完后 root_path 自动推导为 .../zhiyan_pick_place，并从 meta/episodes.jsonl 读 traj 列表
```

### 改动 5：训练命令（服务器）
```bash
cd <你的 X-VLA 副本>
conda activate XVLA     # 或自建环境（该环境 GPU 不可用，见 §7）

# 方案 A：LoRA 微调（4090 D 24GB 首选，省显存）
python peft_train.py \
  --models /externalDisk/data/model/X-VLA-WidowX \
  --train_metas_path /externalDisk/data/vla/datasets/zhiyan_pick_place/meta/info.json \
  --output_dir /externalDisk/data/vla/runs/zhiyan_lora \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --iters 3000 \
  --save_interval 500 \
  --log_interval 20

# 方案 B：全量微调（显存吃紧，谨慎）
python train.py --models ... --train_metas_path ... --batch_size 1 --iters 3000 ...
```

### 改动 6：数据同步（本地 → 服务器）
平台已具备该能力：
```powershell
python tools\remote_check.py --sync     # 或直接用 core/remote.py 的 sync_dataset()
```
数据量：`zhiyan_pick_place` 约 300MB 级（7 条 / 2647 帧 / 14 个 mp4），局域网秒级。

---

## 六、训练后如何回平台验证

```
服务器训练产物（checkpoint/pytorch_model.bin）
   └── core/remote.py :: download_dir()  ← 局域网 tar 流回传（已验证 65.6 MB/s）
         └── 本地模型库（storage/models/）→ 平台"模型训练"页可见
               └── 「🎬 仿真直播」页 → 模式选「模型仿真（zhiyan + 我的模型）」
```
> 注意：平台现用的 lerobot 版本**不含 xvla policy**（策略目录里只有 act/diffusion/pi0/pi0fast/sac/smolvma/tdmpc/vqbet）。X-VLA 已在 **LeRobot 上游原生集成**（官方 README: <https://huggingface.co/docs/lerobot/xvla>），所以"平台一键训练 X-VLA"需要**升级本地 lerobot 版本**；在那之前，X-VLA 走**服务器的 `train.py` / `peft_train.py`**，平台只负责"数据准备 + 产物回收入库 + 仿真验证"。

---

## 七、风险与待确认项

| # | 事项 | 说明 | 建议 |
|---|---|---|---|
| **1** | **现有 XVLA 环境 GPU 不可用** ⚠️ | `torch 2.13.0+cu130` 而驱动是 **570.211.01**（最高 CUDA 12.8）→ `torch.cuda.is_available() == False` | 找维护同事确认：是否有其他可用环境 / 是否要把 torch 降到 cu121~cu128 |
| **2** | **公司服务器，有生产服务在跑** | GPU 已被 inference_service 占 510MB；`uvicorn:8005` / `ml_backend` 已跑 43 天 | **训练前先协调**，建议用 `CUDA_VISIBLE_DEVICES` + 低优先级，且不要改团队的环境 |
| **3** | 系统盘只剩 25G | torch + 依赖 + checkpoint 会很快吃满 | 环境/数据/产物全部放 `/externalDisk/data`（1.8T，但 **exFAT 不支持符号链接**，故 venv/conda 需放 `/home`）|
| **4** | 关节角单位与归一化 | 我们数据是**度**（如 -49.7~28.8）；X-VLA 的 proprio/action 归一化约定未确认 | 读 `datasets/utils.py :: action_slice` 与 `JointActionSpace.preprocess` 后再定 |
| **5** | 语言指令单一 | 只有 `"pick_and_place"` 一条 → VLA 的语言能力用不上 | 采集新数据时把指令写多样（这是 VLA 与 ACT 的核心差别）|
| **6** | 关节角是否要转 ee6d | **结论：不必**（有 `joint` 模式）；但若想复用官方 ee6d 预训练能力，也可用我们已有的 zhiyan URDF 做正运动学 | 优先走 `joint` 模式，成本最低 |
| **7** | zhiyan 只有右臂 7 轴数据 | 左臂/升降/底盘/头部无数据 | 仿真与训练都按"只动右臂"处理（URDF 已可锁死其余关节）|

---

## 八、已完成的配套工作（本次会话）

| 项 | 状态 |
|---|---|
| zhiyan 数据 v3.0 → v2.1 转换（含 action 自复制修正 + 图像统计 + `names` 修复） | ✅ |
| 平台 lerobot 训练链路（ACT 200 步，模型 #14） | ✅ |
| 仿真直播三种模式（LIBERO 策略仿真 / zhiyan 轨迹回放 / **zhiyan 模型驱动仿真**） | ✅ |
| zhiyan 机器人 URDF → MuJoCo（右臂 7 自由度 + 简易手，可被数据/模型驱动） | ✅ |
| 模型评估可视化（专家 vs 模型曲线，手臂 MAE 7.7°） | ✅ |
| 训练页策略下拉 + VLA 走 `--policy.path`（act/smolvla/pi0/pi0fast/diffusion） | ✅ 代码就绪（需重启平台生效）|
| **远程训练后端**（`core/remote.py`：连接/探测/同步/后台启动/日志/产物回收；零依赖 ssh.exe 后端） | ✅ 已验证 |
| SSH 免密（密钥 `storage/ssh/platform_key`，已装入服务器） | ✅ |
| 本地 X-VLA 部署（WidowX 版，3358.6MB，已验证） | ✅ |

## 九、关键路径速查

```
服务器 SSH        z123@192.168.123.105:22（密钥 storage/ssh/platform_key）
服务器 X-VLA 代码 /home/z123/rsg_ws/X-VLA
服务器 XVLA 环境  /home/z123/miniconda3/envs/XVLA
服务器 权重       /externalDisk/data/model/X-VLA-WidowX
服务器 大盘       /externalDisk/data（1.8T 可用，exFAT 不支持软链）
本地 权重         storage/xvla/X-VLA-WidowX
本地 远程配置     storage/remote.json
本地 平台代码     C:\Users\ll\Desktop\HKC_Data_Platform
```

---

---

# 🆕 重大更新（同日补充调研）—— 推荐方案改为「用服务器 lerobot 0.4.4 的 xvla policy」

> **本节结论取代 §五 的改动清单**（那套"改 X-VLA 自有代码"的方案**不再需要**）。

## 十、服务器可用环境总表（28 个 conda 环境实测）

驱动实测：**570.211.01 / CUDA 12.8** → **凡 torch 为 cu129/cu130 的环境一律不可用**。

| 环境 | torch | GPU | 关键依赖 | 能否训 VLA |
|---|---|---|---|---|
| **`lerobot`** ⭐⭐⭐ | 2.10.0+cu128 | ✅ | **lerobot 0.4.4**（含 **xvla**、groot、pi0、pi05、pi0_fast、smolvla、wall_x、sarm、rtc）、accelerate 1.14、av 15.1、diffusers | ✅ **首选** |
| `gr00t` ⭐⭐ | 2.7.0+cu126 | ✅ | **peft 0.17.1 + deepspeed 0.17.6 + flash_attn 2.7.4** + accelerate 1.12 | ✅ 依赖最全（NVIDIA GR00T 栈）|
| `unitree_lerobot` | 2.7.1+cu126 | ✅ | lerobot 0.4.1、transformers 4.57.1 | ✅（lerobot 0.4.1 无 xvla）|
| `unitree_sim_env` | 2.7.0+cu128 | ✅ | lerobot 0.4.1、transformers 4.56.1 | ✅ |
| `rsg` / `rsg_yy` | 2.8.0+cu129 / 2.10.0+cu128 | ✅ | transformers 4.42 / **5.3** | ✅ 通用 |
| `embodiedgen` / `foundationpose` | 2.8.0+cu126 | ✅ | transformers 4.42.4 + diffusers | ✅ 偏生成 |
| `tv` / `tv2` | 2.3.0+cu121 | ✅ | 较老、无 transformers | ⚠️ 有限 |
| `robomimic` | 2.3.1+cu121 (py3.8) | ✅ | 模仿学习 | ⚠️ 非 VLA |
| `RobotVision` | 2.5.1+cu121 (py3.12) | ✅ | **生产服务在用** | ⚠️ 勿动 |
| **`XVLA`** | **2.13.0+cu130** | ❌ | — | ❌ **torch 要求驱动≥580，实际 570** |
| `gmr` | 2.11.0+cu130 | ❌ | — | ❌ 同上 |
| `lerobot-dp` / `unifolm-wma` / `retarget` / `realman` | 无 torch | ❌ | — | ❌ |

**实测报错原文（`XVLA` 环境）**：
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080).
```
→ 这不是我们造成的，是**团队该环境 torch 升级到 cu130 后与驱动 570 不匹配**，建议向维护同事反馈。

## 十一、为什么改用 lerobot 的 xvla（关键发现）

```
/home/z123/miniconda3/envs/lerobot/lib/python3.10/site-packages/lerobot/policies/xvla/
   configuration_xvla.py  (203 行)   modeling_xvla.py (548 行)
   processor_xvla.py      (556 行)   action_hub.py    (588 行)
   soft_transformer.py    (415 行)   modeling_florence2.py (2757 行)
factory.py: elif name == "xvla": from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
```
**LeRobot 上游已把 X-VLA 集成为标准 policy**（`@PreTrainedConfig.register_subclass("xvla")`），因此：

| 原方案（改 X-VLA 自有代码，§五） | **新方案（用 lerobot 的 xvla）** |
|---|---|
| 改 `_REGISTRY` 注册 zhiyan | ❌ 不需要 |
| 新写 `zhiyan.py` Handler（改相机键/动作列/fps） | ❌ 不需要 |
| 改 `domain_config.py` + `num_domains` 30→31 | ❌ 不需要 |
| 用 X-VLA 的 `train.py` / `peft_train.py` | ✅ 用 **lerobot 标准训练命令** |
| 数据要按 Handler 约定再造 | ❌ **不需要**，lerobot 直接读 LeRobot v2.1 |

> 我们平台这次会话已加好 VLA 通路（`--policy.path` + `device=cuda` + batch 自动降 1），**服务器上用同一套机制即可**。

## 十二、`XVLAConfig` 训练参数（从服务器源码读出，可直接用）

```python
n_obs_steps=1          chunk_size=32        n_action_steps=32      dtype="float32"  # 或 bfloat16
normalization_mapping: VISUAL/STATE/ACTION = IDENTITY   ★★ 不做 mean/std 归一化！
                                                     → 我们的数据(度/原始值)可直接喂，不需要统计归一化
tokenizer_name="facebook/bart-large"   tokenizer_max_length=64
hidden_size=1024  depth=24  num_heads=16  mlp_ratio=4.0
num_domains=30    len_soft_prompts=32     dim_time=32   max_len_seq=512
action_mode="ee6d"   ← 可改为 "joint"（我们的数据是关节角）
num_denoising_steps=10    use_proprio=True
max_state_dim=32   max_action_dim=20     # 动作会被 padding 到 20 维
domain_feature_key: str|None = None      ★ 可用数据集某个 feature 当 domain 标识（比硬编码 id 灵活）
resize_imgs_with_padding / num_image_views / empty_cameras=0

# 微调省显存开关 ★
freeze_vision_encoder=False    freeze_language_encoder=False
train_policy_transformer=True  train_soft_prompts=True

# 优化器
optimizer_lr=1e-4   betas=(0.9,0.99)   weight_decay=0.0   grad_clip=10.0
optimizer_soft_prompt_lr_scale=1.0
scheduler_warmup_steps=1000   decay_steps=30000   decay_lr=2.5e-6
```

**对 4090 D 24GB 的关键提示**：可开 `freeze_vision_encoder=True`（冻 VLM 视觉编码器）+ `dtype=bfloat16` + `batch_size=1`，显存压力大幅下降。

## 十三、修订后的执行步骤（服务器，需你授权后执行）

```bash
# 0) 环境（GPU 已验证可用）
conda activate lerobot          # lerobot 0.4.4 + torch 2.10.0+cu128，CUDA True ✓

# 1) 同步数据（本地平台 → 服务器）
#    本地: python tools\remote_check.py --sync
#    目标: /externalDisk/data/vla/datasets/zhiyan_pick_place   (LeRobot v2.1 ✓)

# 2) 试跑（先用 200 步验证链路，lr 小、batch 1、单进程）
python -m lerobot.scripts.train \
  --policy.type=xvla \
  --policy.action_mode=joint \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --dataset.repo_id=local/zhiyan_pick_place \
  --dataset.root=/externalDisk/data/vla/datasets/zhiyan_pick_place \
  --batch_size=1 --steps=200 --save_freq=200 --num_workers=0 \
  --output_dir=/externalDisk/data/vla/runs/zhiyan_xvla_test \
  --wandb.enable=false

# 3) 微调（若要复用官方 X-VLA 权重）
#    --policy.path=/externalDisk/data/model/X-VLA-WidowX   （或 X-VLA-Pt）
```

**验证点**：能加载数据集（数据集统计/特征对齐）→ 能建 policy → 开始迭代（不 OOM）→ 有 loss 下降。

## 十四、仍需确认（比原清单少了很多）

| # | 事项 | 现状 |
|---|---|---|
| 1 | 训练占用 GPU 需协调（有 43 天的生产服务） | ❓ 建议问维护同事 |
| 2 | `action_mode` 用 `joint` 还是 `ee6d` | `joint` 最省事（数据就是关节角）；`ee6d` 需正运动学转换 |
| 3 | 14 维动作如何映射到 `max_action_dim=20` | 大概率走 `auto`/padding，需实测确认 |
| 4 | 语言指令单一（仅 `pick_and_place`） | 建议采集时丰富化 |
| 5 | 本地 lerobot 是 0.3.x（无 xvla） | 训练在服务器（0.4.4）→ **不影响**；本地仅做推理/仿真 |

---

*本报告基于对服务器代码的只读阅读与实测数据，未在服务器做任何安装或修改。*
*（2026-10-14 补充：新增 §十~§十四，推荐方案由"改 X-VLA 自有代码"改为"用服务器 lerobot 0.4.4 的 xvla policy"。）*
