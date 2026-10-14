#!/bin/bash

#SBATCH --job-name=gpu_job
#SBATCH --account=cis260266p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate intent311

cd "$SLURM_SUBMIT_DIR"

python "$@"