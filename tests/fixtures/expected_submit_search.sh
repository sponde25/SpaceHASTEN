#!/bin/bash
#SBATCH -J search
#SBATCH -o /WORKDIR/logs/slurm/search/task-%a.out
#SBATCH -e /WORKDIR/logs/slurm/search/task-%a.err
#SBATCH -p jobs
#SBATCH --cpus-per-task=2
#SBATCH --array=1-4%2

set -euo pipefail
export TASK_ID="${SLURM_ARRAY_TASK_ID}"
source /data/programs/oce/actoce
conda activate chemprop-2.1.2

echo "task ${SLURM_ARRAY_TASK_ID}" > out_${SLURM_ARRAY_TASK_ID}.txt
