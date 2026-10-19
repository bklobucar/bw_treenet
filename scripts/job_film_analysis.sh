#!/bin/bash -l
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 01:00:00
#SBATCH -J film_analysis
#SBATCH -o results/logs/%x_%j.out
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
echo "== common footprint + film footprints"
python -u scripts/39_common_footprint.py --extra 1990 2005cmp 2005deg --own-footprint pan1999 pan2000 --no-diag 2>&1 | grep -v -i warn
echo "== 1990s labels, predicted pixels only"
python -u scripts/38_eval_1990_labels.py 2>&1 | grep -v -i warn
