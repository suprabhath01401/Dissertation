#!/bin/bash
#SBATCH --job-name=mixtral-infer
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-02:00:00
#SBATCH --output=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/mixtral_%j.out
#SBATCH --error=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/mixtral_%j.err

export HF_HOME=/dcs/large/u5754610

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

conda activate torch_env

cd /dcs/pg25/u5754610/Desktop/test/legal_rag

python backend/routing/cluster_inference.py \
  --input-file "$INFER_INPUT" \
  --output-file "$INFER_OUTPUT" \
  --model "mistralai/Mixtral-8x7B-Instruct-v0.1"
