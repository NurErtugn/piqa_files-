#!/bin/bash
#SBATCH --job-name=propella-53
#SBATCH --account=a-infra01
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/nertug/fineweb-gems/logs/propella-53-%j.out
#SBATCH --error=/capstor/scratch/cscs/nertug/fineweb-gems/logs/propella-53-%j.err

set -euo pipefail

export PIPELINE_ROOT=/capstor/scratch/cscs/nertug/fineweb-gems
export HF_HOME="$PIPELINE_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p \
  "$PIPELINE_ROOT/logs" \
  "$PIPELINE_ROOT/propella-calibration"

srun \
  --nodes=1 \
  --ntasks=1 \
  --overlap \
  --environment=vllm-rocm \
  vllm serve ellamind/propella-1-4b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.60 \
    --generation-config vllm \
    > "$PIPELINE_ROOT/logs/propella-vllm-$SLURM_JOB_ID.log" 2>&1 &

SERVER_PID=$!

echo "Waiting for Propella server..."

READY=0

for attempt in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Propella server stopped before becoming ready."
        tail -n 100 \
          "$PIPELINE_ROOT/logs/propella-vllm-$SLURM_JOB_ID.log"
        exit 1
    fi

    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "Propella server did not become ready."
    tail -n 100 \
      "$PIPELINE_ROOT/logs/propella-vllm-$SLURM_JOB_ID.log"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo "Propella server is ready."

source /users/nertug/miniconda3_x86/bin/activate
conda activate fineweb-gems

cd /users/nertug/projects/fineweb-gems/pipeline

python run_propella_calibration.py \
  --input "$PIPELINE_ROOT/propella-calibration/audit53.jsonl" \
  --output "$PIPELINE_ROOT/propella-calibration/audit53_propella.jsonl"

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "Propella calibration completed."
