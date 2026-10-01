#!/bin/bash
#SBATCH --job-name=mistral-10m-v3pilot
#SBATCH --account=a-infra01
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0
#SBATCH --output=/capstor/scratch/cscs/nertug/fineweb-gems/logs/mistral-10m-v3pilot-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/nertug/fineweb-gems/logs/mistral-10m-v3pilot-%A_%a.err

set -euo pipefail

export PIPELINE_ROOT=/capstor/scratch/cscs/nertug/fineweb-gems
export HF_HOME="$PIPELINE_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"

MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
SHARD="$SLURM_ARRAY_TASK_ID"

AUDIT_ROOT="$PIPELINE_ROOT/production10m"
INPUT="$AUDIT_ROOT/independent-v3-pilot/input/part-${SHARD}.jsonl"
OUTPUT="$AUDIT_ROOT/independent-v3-pilot/output/part-${SHARD}.jsonl"

PORT=$((24000 + SLURM_JOB_ID % 15000))
BASE_URL="http:""//127.0.0.1:${PORT}"
ENDPOINT="${BASE_URL}/v1/chat/completions"

mkdir -p "$PIPELINE_ROOT/logs" "$AUDIT_ROOT/independent-v3-pilot/output"

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
  vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --generation-config vllm \
    > "$PIPELINE_ROOT/logs/mistral-vllm-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

echo "Shard: $SHARD"
echo "Waiting for Mistral server and model download..."

READY=0

for attempt in $(seq 1 720); do
    if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Mistral server stopped before becoming ready."
        tail -n 150 \
          "$PIPELINE_ROOT/logs/mistral-vllm-${SLURM_JOB_ID}.log"
        exit 1
    fi

    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "Mistral server was not ready within two hours."
    tail -n 150 \
      "$PIPELINE_ROOT/logs/mistral-vllm-${SLURM_JOB_ID}.log"
    exit 1
fi

echo "Mistral server is ready."

source /users/nertug/miniconda3_x86/bin/activate
conda activate fineweb-gems

cd /users/nertug/projects/fineweb-gems/pipeline

python run_independent_judge_production_v3.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --resume

echo "Independent audit shard $SHARD completed."
