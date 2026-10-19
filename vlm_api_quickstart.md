# Qwen3-VL-4B 服务速查（x86 / RTX 4090 D）

## 1. 环境

- **运行方式**：非 Docker（外置盘是 exFAT，Docker/venv 都要软链，建不起来）——直接在 venv 里跑 vLLM，由 **systemd 托管、开机自启**。
- **运行环境**：`z123@192.168.123.105`（主机名 `z2g9`），x86_64 / Ubuntu 22.04 / Python 3.10.12；**vLLM 0.19.1 + torch 2.10.0+cu128**（CUDA 12.8，驱动 570.211.01）；RTX 4090 D 24 GB，`--gpu-memory-utilization 0.5 --max-model-len 16384`。
  > 不能用更高版本 vLLM：0.20+ 拉的是 torch cu13，需要驱动 ≥ 580。
- **模型路径**：`/externalDisk/data/models/Qwen3-VL-4B-Instruct-AWQ-4bit`；venv 在 `/externalDisk/data/vllm-mnt/venv`（ext4 loop 镜像 `vllm-disk.img`，写进 `/etc/fstab` 自动挂载）。
- **服务名**：`qwen3vl.service`

```bash
sudo systemctl start   qwen3vl     # 启动（约 40 s 就绪）
sudo systemctl stop    qwen3vl     # 关闭
sudo systemctl restart qwen3vl     # 重启
systemctl status qwen3vl           # 状态（不需要 sudo）
```

排障：`journalctl -u qwen3vl -n 50`，或看 `/externalDisk/data/vllm-mnt/logs/qwen3vl.log`。

## 2. 请求方式

OpenAI 兼容：**`http://192.168.123.105:8000/v1`**，模型名 **`qwen3-vl-4b`**，图片/视频都用 base64 data URL。

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://192.168.123.105:8000/v1", api_key="EMPTY")  # key 随便填

img = base64.b64encode(open("帧.jpg", "rb").read()).decode()
r = client.chat.completions.create(
    model="qwen3-vl-4b", temperature=0, max_tokens=64,
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
        {"type": "text", "text": '读出图中呼梯面板两列楼层数字，只输出 JSON：{"left":"","right":""}'}]}])
print(r.choices[0].message.content)      # 实测 → {"left":"4","right":"7"}
```

> ⚠️ 本机若设了 `HTTP_PROXY` / `NO_PROXY`（尤其含 `[::1]`），openai SDK 走 httpx 会直接抛
> `Unknown scheme for proxy URL` / `Invalid port`。内网直连不需要代理，跑之前清掉即可：
> `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy -u NO_PROXY -u no_proxy python 你的脚本.py`

## 3. 视频

整段视频**直接附在请求里**即可，无需抽帧，服务端自己解码并全片均匀采样：

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://192.168.123.105:8000/v1", api_key="EMPTY")

vb64 = base64.b64encode(open("采集.mp4", "rb").read()).decode()
r = client.chat.completions.create(
    model="qwen3-vl-4b", temperature=0, max_tokens=256,
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + vb64}},
        {"type": "text", "text": "这段视频里在干什么？用中文简要描述。"}]}])
print(r.choices[0].message.content)
# 实测（1920x1080 / 8.8 s / 44 MB）→ 5.1 s，"一个白色的机械臂正在操作一个触摸屏设备……"
```

判断数采成功/失败，只把最后那句 prompt 换掉（模型只回 JSON，退出码可自行按 `success` 决定）：

```python
{"type": "text", "text": '判断这次数据采集是否成功（机器人是否完成预期操作、没有卡死/掉落/明显失败）。'
                         '只输出 JSON：{"success": true|false, "reason": "不超过20字"}'}
```

**成本与限制（实测）**

| 输入 | prompt_tokens | 结果 |
|---|---:|---|
| 640×480 / 4 s，直传 | 4954 | ✅ ~1 s |
| 1280×720 / 4 s，直传 | 11674 | ✅ ~2 s |
| 1920×1080 / 8.8 s，直传 | 11668 | ✅ 5.1 s |
| 8 张 1080p 帧（客户端抽帧） | 16369 | ⚠️ 逼近上限，**输出被截断** |

- **token 数与视频时长无关**（4 s 和 20 s 几乎一样）——服务端是固定帧预算的全片均匀采样，所以**能看到结尾**，长视频不会更贵。
- **直传比客户端抽帧更省**：服务端会把分辨率归一到内部像素上限（720p 与 1080p 的 token 数几乎相同），而客户端抽帧按原分辨率全算。
- 想要更高分辨率或更多帧，相应调大服务端 `--max-model-len`；`cap_pixels_per_frame` 在 vLLM 0.19.1 上**尚不生效**（实测无变化）。
