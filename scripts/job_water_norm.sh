#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J waternorm
#SBATCH -o results/logs/%x_%j.out
# Water hypothesis: (50) patch failure and window drops by water fraction (gtb, sth 2005/2010), then
# (49) variants a (production), d (water-masked norms), e (shifted-grid ensemble) on the 49 test sheets
# and labelled sets. Each step is skipped if its output exists.
# Submit from the repo root:  sbatch scripts/job_water_norm.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate bwtreenet
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START waternorm $(date)  on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ -f results/water_patch_check.csv ]]; then
  echo "results/water_patch_check.csv exists, skipping 50"
else
  python scripts/50_water_patch_check.py --cities gtb sth --eras 2005 2010
fi
echo "50 done $(date)"
if [[ -f results/norm_fix_test/summary_ade.csv ]]; then
  echo "results/norm_fix_test/summary_ade.csv exists, skipping 49"
else
  python scripts/49_norm_fix_test.py --variants a d e
fi
echo "END waternorm $(date)"
