#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J rank_windows
#SBATCH -o results/logs/%x_%j.out
# Candidate windows for the empty label_windows.csv rows (55): checks the CSV rows with coordinates first
# (centre in sheet, flight year(s), mosaic cover), then ranks categories A, B (2005, 2010) and C into
# results/window_candidates/ (CSV + PNG strip each). Maps: results/inference_v19_hydro2 (51); NMD:
# data/ext/nmd2023/nmd2023.vrt. gis env: GH200 only, hence the gpu partition.
# Submit from the repo root:  sbatch scripts/job_rank_windows.sh      (--check-only: just the CSV check)

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -eo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START rank_windows $(date)  on $(hostname)"
for f in label_windows.csv data/Tatorter_2023.gpkg data/ext/nmd2023/nmd2023.vrt data/hydro_v2/malmo_water.gpkg \
         data/hydro_v2/gtb_water.gpkg data/hydro_v2/sth_water.gpkg; do
  [[ -f $f ]] || { echo "ERROR: $f not found"; exit 1; }
done
# extra arguments go to 55, e.g. sbatch scripts/job_rank_windows.sh --overwrite
python -u scripts/55_rank_candidate_windows.py "$@"
echo "END rank_windows $(date)"
