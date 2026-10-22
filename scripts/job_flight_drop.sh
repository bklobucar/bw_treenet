#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J flightdrop
#SBATCH -o results/logs/%x_%j.out
# ~2005 vs ~2010 canopy per 250 m window (gtb, sth, malmo): change distribution, >20 pp drops, top 20.
# Writes results/digital_flight_drop_windows.csv (not replaced if it exists).
# Submit from the repo root:  sbatch scripts/job_flight_drop.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START flightdrop $(date)  on $(hostname)"
python scripts/46_digital_flight_drop.py --cities gtb sth malmo
echo "END flightdrop $(date)"
