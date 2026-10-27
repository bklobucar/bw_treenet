#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J label_queue
#SBATCH -o results/logs/%x_%j.out
# Labelling package for the research assistant (53): windows from label_windows.csv and the stratified
# test points -> results/anna_labelling/ and results/anna_labelling.zip; strata keys -> results/anna_labelling_key/
# (not in the zip). Existing window/set folders are kept; nothing is overwritten without --overwrite.
# NMD2023 basskikt v2.1: the per-city clips + data/ext/nmd2023/nmd2023.vrt (copied from the PC; not in git).
# Without it, 53 builds the windows and skips the test points. gis env: GH200 only, hence the gpu partition.
# Submit from the repo root:  sbatch scripts/job_label_queue.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -eo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
NMD=data/ext/nmd2023/nmd2023.vrt

echo "START label_queue $(date)  on $(hostname)"
for f in label_windows.csv data/Tatorter_2023.gpkg data/hydro_v2/malmo_water.gpkg data/hydro_v2/gtb_water.gpkg \
         data/hydro_v2/sth_water.gpkg styles/extent.qml styles/canopy.qml styles/ignore.qml styles/points.qml; do
  [[ -f $f ]] || { echo "ERROR: $f not found"; exit 1; }
done
[[ -f $NMD ]] || echo "WARNING: NMD not found at $NMD; 53 will skip the test points"
python -u scripts/53_build_label_queue.py --nmd "$NMD"
du -sh results/anna_labelling results/anna_labelling.zip results/anna_labelling_key
echo "END label_queue $(date)"
