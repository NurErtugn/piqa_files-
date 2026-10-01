#!/bin/bash
#SBATCH --job-name=assemble-common-v3
#SBATCH --account=a-infra01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/users/nertug/projects/fineweb-gems/pipeline/assemble-common-v3-%j.out
#SBATCH --error=/users/nertug/projects/fineweb-gems/pipeline/assemble-common-v3-%j.err
set -euo pipefail
cd /users/nertug/projects/fineweb-gems/pipeline
/users/nertug/miniconda3_x86/envs/fineweb-gems/bin/python assemble_commonsense_v3.py
