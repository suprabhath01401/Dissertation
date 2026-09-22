#!/bin/bash
#SBATCH --job-name=llama-ollama-infer
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-02:00:00
#SBATCH --output=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/llama_ollama_%j.out
#SBATCH --error=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/llama_ollama_%j.err

# Starts a local ollama server on this compute node, ensures llama3.1:8b is
# cached (no-op if warmup_llama_ollama.sbatch already pulled it), then runs
# one inference through cluster_inference.py — same model + serving engine
# as the local full_system config, just running on cluster GPU hardware.

export OLLAMA_MODELS=/dcs/large/u5754610/ollama_models
export OLLAMA_HOST=127.0.0.1:11435
export PATH="/dcs/large/u5754610/ollama/bin:$PATH"
mkdir -p "$OLLAMA_MODELS"

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
conda activate torch_env

echo "[llama_ollama] Starting ollama serve on $OLLAMA_HOST"
ollama serve &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

ready=0
for i in $(seq 1 30); do
  if curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  echo "[llama_ollama] ollama serve did not become ready in time" >&2
  exit 1
fi

echo "[llama_ollama] Ensuring llama3.1:8b is pulled (no-op if already cached)"
ollama pull llama3.1:8b

cd /dcs/pg25/u5754610/Desktop/test/legal_rag

export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
python backend/routing/cluster_inference.py \
  --input-file "$INFER_INPUT" \
  --output-file "$INFER_OUTPUT" \
  --model "llama3.1:8b"
