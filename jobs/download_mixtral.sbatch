#!/bin/bash
#SBATCH --job-name=download-mixtral
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/download_mixtral_%j.out
#SBATCH --error=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/download_mixtral_%j.err

export HF_HOME=/dcs/large/u5754610
export TRANSFORMERS_CACHE=/dcs/large/u5754610/hub

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

conda activate torch_env

echo "[download_mixtral] Downloading to $HF_HOME"

python - <<'EOF'
import os
from huggingface_hub import snapshot_download

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
cache_dir = os.environ["HF_HOME"]

print(f"Downloading {model_id} -> {cache_dir}")
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    local_files_only=False,
)
print("Done.")
EOF
