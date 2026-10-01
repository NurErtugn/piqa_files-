#!/bin/bash
#SBATCH --job-name=judge-smoke
#SBATCH --account=a-infra01
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/nertug/fineweb-gems/logs/judge-smoke-%j.out
#SBATCH --error=/capstor/scratch/cscs/nertug/fineweb-gems/logs/judge-smoke-%j.err

set -euo pipefail

export PIPELINE_ROOT=/capstor/scratch/cscs/nertug/fineweb-gems
export HF_HOME="$PIPELINE_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$PIPELINE_ROOT/precision-smoke"

srun \
  --nodes=1 \
  --ntasks=1 \
  --overlap \
  --environment=vllm-rocm \
  vllm serve Qwen/Qwen3-32B \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$PIPELINE_ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

SERVER_PID=$!

echo "Waiting for Qwen server..."

READY=0
for attempt in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM stopped before becoming ready."
        tail -n 100 "$PIPELINE_ROOT/logs/vllm-$SLURM_JOB_ID.log"
        exit 1
    fi

    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "vLLM did not become ready in 30 minutes."
    tail -n 100 "$PIPELINE_ROOT/logs/vllm-$SLURM_JOB_ID.log"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo "Qwen server is ready."

source /users/nertug/miniconda3_x86/bin/activate
conda activate fineweb-gems

cd /users/nertug/projects/fineweb-gems/pipeline

export JUDGE_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export JUDGE_MODEL=Qwen/Qwen3-32B
export JUDGE_API_KEY=EMPTY

fineweb-gems judge \
  --mode precision \
  --input "$PIPELINE_ROOT/candidates-pilot/part-0.jsonl" \
  --output "$PIPELINE_ROOT/precision-smoke/part-0.jsonl" \
  --limit 10

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "Judge smoke test completed."
