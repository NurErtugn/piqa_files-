#!/bin/bash
#SBATCH --job-name=propella-10m
#SBATCH --account=a-infra01
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-622%4
#SBATCH --output=/capstor/scratch/cscs/nertug/fineweb-gems/logs/propella-10m-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/nertug/fineweb-gems/logs/propella-10m-%A_%a.err

set -euo pipefail

export PIPELINE_ROOT=/capstor/scratch/cscs/nertug/fineweb-gems
export HF_HOME="$PIPELINE_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"

SHARD="$SLURM_ARRAY_TASK_ID"
INPUT="$PIPELINE_ROOT/production10m/propella-input/part-${SHARD}.jsonl"
OUTPUT="$PIPELINE_ROOT/production10m/propella/part-${SHARD}.jsonl"
ERRORS="$PIPELINE_ROOT/production10m/propella-errors/part-${SHARD}.jsonl"

# Use a job-specific port in case multiple tasks share a node.
PORT=$((20000 + SLURM_JOB_ID % 20000))
ENDPOINT="http:""//127.0.0.1:${PORT}/v1/chat/completions"

mkdir -p \
  "$PIPELINE_ROOT/logs" \
  "$PIPELINE_ROOT/production10m/propella" \
  "$PIPELINE_ROOT/production10m/propella-errors"

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
  vllm serve ellamind/propella-1-4b \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.60 \
    --generation-config vllm \
    > "$PIPELINE_ROOT/logs/propella-vllm-${SLURM_JOB_ID}-${SHARD}.log" 2>&1 &

SERVER_PID=$!

echo "Shard: $SHARD"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Port: $PORT"
echo "Waiting for Propella server..."

READY=0

for attempt in $(seq 1 180); do
    if curl -fsS \
        "http://127.0.0.1:${PORT}/v1/models" \
        >/dev/null 2>&1; then
        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Propella server stopped before becoming ready."
        tail -n 100 \
          "$PIPELINE_ROOT/logs/propella-vllm-${SLURM_JOB_ID}-${SHARD}.log"
        exit 1
    fi

    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "Propella server did not become ready."
    tail -n 100 \
      "$PIPELINE_ROOT/logs/propella-vllm-${SLURM_JOB_ID}-${SHARD}.log"
    exit 1
fi

echo "Propella server is ready."

source /users/nertug/miniconda3_x86/bin/activate
conda activate fineweb-gems

cd /users/nertug/projects/fineweb-gems/pipeline

python run_propella_calibration.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --errors "$ERRORS" \
  --endpoint "$ENDPOINT"

echo "Shard $SHARD completed."
