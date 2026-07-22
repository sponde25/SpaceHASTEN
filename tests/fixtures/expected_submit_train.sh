#!/bin/bash
#SBATCH -J train
#SBATCH -o /WORKDIR/logs/slurm/train/task-%a.out
#SBATCH -e /WORKDIR/logs/slurm/train/task-%a.err
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --array=1-1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

set -euo pipefail
export TASK_ID="${SLURM_ARRAY_TASK_ID}"
source /data/programs/oce/actoce
conda activate chemprop-2.1.2

python3 -m spacehasten.remote.train data.csv model_v1
