#!/bin/bash
#SBATCH --job-name=extract-v3-passages
#SBATCH --account=a-infra01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/nertug/projects/fineweb-gems/pipeline/extract-v3-passages-%j.out
#SBATCH --error=/users/nertug/projects/fineweb-gems/pipeline/extract-v3-passages-%j.err
set -euo pipefail
cd /users/nertug/projects/fineweb-gems/pipeline
/users/nertug/miniconda3_x86/envs/fineweb-gems/bin/python prepare_selected_passages_v3.py
