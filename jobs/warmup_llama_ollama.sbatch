#!/bin/bash
#SBATCH --job-name=warmup-llama-ollama
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --output=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/warmup_llama_ollama_%j.out
#SBATCH --error=/dcs/pg25/u5754610/Desktop/test/legal_rag/logs/warmup_llama_ollama_%j.err

# One-time job: starts a local ollama server on the compute node and pulls
# llama3.1:8b into shared storage so later inference jobs don't pay for it.
# No GPU requested — pulling doesn't need one.

export OLLAMA_MODELS=/dcs/large/u5754610/ollama_models
export OLLAMA_HOST=127.0.0.1:11435
export PATH="/dcs/large/u5754610/ollama/bin:$PATH"
mkdir -p "$OLLAMA_MODELS"

echo "[warmup] Starting ollama serve on $OLLAMA_HOST (models -> $OLLAMA_MODELS)"
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
  echo "[warmup] ollama serve did not become ready in time" >&2
  exit 1
fi

echo "[warmup] Pulling llama3.1:8b"
ollama pull llama3.1:8b
echo "[warmup] Done."
