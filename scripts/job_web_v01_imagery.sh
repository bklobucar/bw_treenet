#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH -J web_v01_imagery
#SBATCH -o results/logs/%x_%j.out
# v0.1 web map: historical imagery COGs (flight years <= 2005 only) + manifest (43 --what imagery).
# Submit from the repo root:  sbatch scripts/job_web_v01_imagery.sh
# Existing COGs are kept, so a rerun resumes.

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START web_v01_imagery $(date)  on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
python scripts/43_export_web_layers.py --what imagery
echo "43 imagery done $(date)"
du -sh results/web_v01/imagery
echo "END web_v01_imagery $(date)"
