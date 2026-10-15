#!/bin/bash -l
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 00:45:00
#SBATCH -J agg_eval
#SBATCH -o results/logs/%x_%j.out
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
echo "== eval 1990 labels"; python -u scripts/38_eval_1990_labels.py 2>&1 | grep -v -i warn
source ~/bw_setup.sh > /dev/null 2>&1
echo "== aggregate"; python -u scripts/20_aggregate_canopy.py 2>&1 | grep -v Warning; echo "aggregate exit code ${PIPESTATUS[0]}"
