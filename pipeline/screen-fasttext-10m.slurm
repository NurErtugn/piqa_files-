#!/bin/bash
#SBATCH --job-name=fw-ft-10m
#SBATCH --account=a-infra01
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/nertug/fineweb-gems/logs/screen-ft-10m-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/nertug/fineweb-gems/logs/screen-ft-10m-%A_%a.err

set -euo pipefail

source /users/nertug/miniconda3_x86/bin/activate
conda activate fineweb-gems

cd /users/nertug/projects/fineweb-gems/pipeline

export PIPELINE_ROOT=/capstor/scratch/cscs/nertug/fineweb-gems
export HF_HOME="$PIPELINE_ROOT/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p \
  "$PIPELINE_ROOT/logs" \
  "$PIPELINE_ROOT/fasttext-source-production10m/candidates" \
  "$PIPELINE_ROOT/fasttext-source-production10m/stats"

python screen_fasttext_frozen.py \
  --config /users/nertug/projects/fineweb-gems/pipeline/config.json \
  --piqa /users/nertug/projects/fineweb-gems/pipeline/work/piqa-signatures.json \
  --thresholds "$PIPELINE_ROOT/production-config/fasttext-thresholds-frozen.json" \
  --worker "$SLURM_ARRAY_TASK_ID" \
  --workers 4 \
  --max-seen 2500000 \
  --output "$PIPELINE_ROOT/fasttext-source-production10m/candidates/part-${SLURM_ARRAY_TASK_ID}.jsonl" \
  --stats-output "$PIPELINE_ROOT/fasttext-source-production10m/stats/part-${SLURM_ARRAY_TASK_ID}.json"
