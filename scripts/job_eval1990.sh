#!/bin/bash -l
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH -J eval1990
#SBATCH -o results/logs/%x_%j.out
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
module load GPU/Miniforge/26.3.2-2-eb
eval "$(conda shell.bash hook)"
conda activate gis
python scripts/38_eval_1990_labels.py
