#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -J label_stats
#SBATCH -o results/logs/%x_%j.out
# 57: hole and patch sizes in the historical training labels (data/processed/*/tiles_19*/labels) and the label
# vectors (clips_for_labelling, */validation) -> results/label_stats/. Read-only on data/. gis env: GH200 only.
# Submit from the repo root:  sbatch scripts/job_label_stats.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -eo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
echo "START label_stats $(date)  on $(hostname)"
echo "== label vector files under data/ (for the record)"
find data/raw data/processed -maxdepth 3 \( -name "*.shp" -o -name "*.gpkg" \) | grep -iv -e tatort -e hydro | sort
echo "== label sets"
for d in data/processed/*/tiles_19*; do echo "$d: $(ls $d/labels/*.tif 2>/dev/null | wc -l) label tiles"; done
python -u scripts/57_label_stats.py "$@"
echo "END label_stats $(date)"
