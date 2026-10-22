#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J dropedges
#SBATCH -o results/logs/%x_%j.out
# Do the ~2005 -> ~2010 window drops (46) cluster at sheet / frame edges? (48; stdout only)
# Needs results/digital_flight_drop_windows.csv from job_flight_drop.sh.
# Submit from the repo root:  sbatch scripts/job_drop_edges.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START dropedges $(date)  on $(hostname)"
python scripts/48_drop_edge_check.py --cities gtb sth malmo
echo "END dropedges $(date)"
