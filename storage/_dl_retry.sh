
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/externalDisk/data/vla/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0
ok=0
for i in $(seq 1 12); do
  echo "=== attempt $i ==="
  /home/z123/miniconda3/envs/lerobot/bin/python -c "from huggingface_hub import snapshot_download as s; s('lerobot/xvla-base', local_dir='/externalDisk/data/model/xvla-base', max_workers=2)" && ok=1 && break
  echo "attempt $i failed, retry in 5s"
  sleep 5
done
if [ $ok -eq 1 ]; then echo DOWNLOAD_OK; else echo DOWNLOAD_FAILED; fi
ls -la /externalDisk/data/model/xvla-base
