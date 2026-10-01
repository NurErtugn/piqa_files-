#!/bin/bash
#SBATCH --job-name=recall-rem3904
#SBATCH --account=a-infra01
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/capstor/scratch/cscs/nertug/fineweb-gems/logs/recall-pilot2k-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/nertug/fineweb-gems/logs/recall-pilot2k-%A_%a.err

set -euo pipefail

export PIPELINE_ROOT=/capstor/scratch/cscs/nertug/fineweb-gems
export HF_HOME="$PIPELINE_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"

SHARD="$SLURM_ARRAY_TASK_ID"
INPUT="$PIPELINE_ROOT/judge-remaining3904/recall-input/part-${SHARD}.jsonl"
OUTPUT="$PIPELINE_ROOT/judge-remaining3904/recall/part-${SHARD}.jsonl"

PORT=$((20000 + SLURM_JOB_ID % 20000))
BASE_URL="http:""//127.0.0.1:${PORT}"
ENDPOINT="${BASE_URL}/v1/chat/completions"

mkdir -p \
  "$PIPELINE_ROOT/logs" \
  "$PIPELINE_ROOT/judge-remaining3904/recall"

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

srun \
  --nodes=1 \
  --ntasks=1 \
  --overlap \
  --environment=vllm-rocm \
  vllm serve Qwen/Qwen3-32B \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$PIPELINE_ROOT/logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

echo "Shard: $SHARD"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Waiting for Qwen server..."

READY=0

for attempt in $(seq 1 540); do
    if curl -fsS \
        "${BASE_URL}/v1/models" \
        >/dev/null 2>&1; then
        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Qwen server stopped before becoming ready."
        tail -n 100 \
          "$PIPELINE_ROOT/logs/vllm-${SLURM_JOB_ID}.log"
        exit 1
    fi

    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "Qwen server did not become ready in 90 minutes."
    tail -n 100 \
      "$PIPELINE_ROOT/logs/vllm-${SLURM_JOB_ID}.log"
    exit 1
fi

echo "Qwen server is ready."

source /users/nertug/miniconda3_x86/bin/activate
conda activate fineweb-gems

cd /users/nertug/projects/fineweb-gems/pipeline

fineweb-gems judge \
  --mode recall \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --endpoint "$ENDPOINT" \
  --model Qwen/Qwen3-32B \
  --resume

echo "Recall shard $SHARD completed."
