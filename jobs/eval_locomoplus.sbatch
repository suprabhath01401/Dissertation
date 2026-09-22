#!/bin/bash
#SBATCH --job-name=eval-locomoplus
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/eval_locomoplus_%j.out
#SBATCH --error=logs/eval_locomoplus_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

cd /dcs/pg25/u5754610/Desktop/test/legal_rag

python backend/evaluation/benchmarks.py --dataset locomoplus
