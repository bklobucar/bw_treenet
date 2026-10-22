#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH -J web_v01
#SBATCH -o results/logs/%x_%j.out
# v0.1 web map: DeSO aggregation (41), then canopy COGs + manifest (43 --what canopy).
# Imagery runs separately: scripts/job_web_v01_imagery.sh.
# Submit from the repo root:  sbatch scripts/job_web_v01.sh
# 41 skips if results/deso/ outputs exist; 43 keeps existing COGs, so a rerun resumes.

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
DESO=data/ext/DeSO_2025.gpkg
[[ -f $DESO ]] || { echo "ERROR: $DESO not found"; exit 1; }

echo "START web_v01 $(date)  on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
python scripts/41_deso_aggregate.py --deso $DESO
echo "41 done $(date)"
python scripts/43_export_web_layers.py --what canopy
echo "43 canopy done $(date)"
du -sh results/deso results/web_v01/canopy
echo "END web_v01 $(date)"
