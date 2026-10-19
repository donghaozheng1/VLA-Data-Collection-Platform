#!/bin/bash
set -e
ENV=/home/z123/vla_env
ROOT=/externalDisk/data/vla
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com
export PIP_CACHE_DIR=/home/z123/tmp/pipcache
export TMPDIR=/home/z123/tmp
mkdir -p $TMPDIR $PIP_CACHE_DIR
echo "== 0/5 装 virtualenv（system pip，无需 sudo）=="
python3 -m pip install -q --user virtualenv
echo "== 1/5 建环境（--copies）=="
rm -rf $ENV
python3 -m virtualenv --copies -q $ENV 2>&1 | tail -3
source $ENV/bin/activate
python -V
echo "== 2/5 torch cu121 =="
pip install -q -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch;print('TORCH_OK',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo "== 3/5 X-VLA 仓库 =="
[ -d $ROOT/X-VLA/.git ] || git clone --depth 1 https://github.com/2toINF/X-VLA.git $ROOT/X-VLA
ls $ROOT/X-VLA | head
echo "== 4/5 0.9B 权重（3.3GB）=="
pip install -q huggingface_hub
python -c "from huggingface_hub import snapshot_download;print(snapshot_download('2toINF/X-VLA-Pt',local_dir='/externalDisk/data/vla/X-VLA-Pt'))"
echo "== 5/5 依赖 =="
pip install -q transformers accelerate einops timm
[ -f $ROOT/X-VLA/requirements.txt ] && pip install -q -r $ROOT/X-VLA/requirements.txt || echo 'no requirements'
echo "ENV_SETUP_DONE"
du -sh $ENV $PIP_CACHE_DIR $ROOT 2>/dev/null
df -h / /externalDisk/data | tail -2
