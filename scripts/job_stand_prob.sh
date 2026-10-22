#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J standprob
#SBATCH -o results/logs/%x_%j.out
# Probability panels and canopy/non-canopy histograms (~2005, 2010) for gtb_pan2000 strip rows 3, 6, 7.
# Writes results/stand_prob_check/gtb_pan2000_row<N>.png; existing PNGs are not replaced.
# Submit from the repo root:  sbatch scripts/job_stand_prob.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START standprob $(date)  on $(hostname)"
python scripts/45_stand_prob_check.py --log results/logs/film_fill_4233698.out --key gtb_pan2000 --rows 3 6 7
echo "END standprob $(date)"
