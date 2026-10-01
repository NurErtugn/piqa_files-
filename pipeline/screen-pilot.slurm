#!/bin/bash
#SBATCH --job-name=fw-pilot
#SBATCH --account=a-infra01
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/nertug/fineweb-gems/logs/screen-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/nertug/fineweb-gems/logs/screen-%A_%a.err

set -euo pipefail

source /users/nertug/miniconda3_x86/bin/activate
conda activate fineweb-gems

cd /users/nertug/projects/fineweb-gems/pipeline

export PIPELINE_ROOT=/capstor/scratch/cscs/nertug/fineweb-gems
export HF_HOME="$PIPELINE_ROOT/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

fineweb-gems screen \
  --config "$PIPELINE_ROOT/config.json" \
  --piqa "$PIPELINE_ROOT/piqa-signatures.json" \
  --worker "$SLURM_ARRAY_TASK_ID" \
  --workers "$SLURM_ARRAY_TASK_COUNT" \
  --max-candidates 500 \
  --output "$PIPELINE_ROOT/candidates-pilot/part-$SLURM_ARRAY_TASK_ID.jsonl"
