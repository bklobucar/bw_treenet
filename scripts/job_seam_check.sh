#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J seamcheck
#SBATCH -o results/logs/%x_%j.out
# Seam diagnostic (47): gtb 639_31_55 E317875 N6395875, 2012 flight (era 2010) -- ramp vs sheet edges and
# inference patch grid, per-patch input/normalisation stats, rerun of the right patch with production and
# left-patch norms; then a scan of gtb/sth 2005 and 2010 prob maps for near-zero patches over texture.
# Writes results/seam_check/ (existing outputs are not replaced).
# Submit from the repo root:  sbatch scripts/job_seam_check.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate bwtreenet
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START seamcheck $(date)  on $(hostname)"
python scripts/47_seam_check.py --city gtb --era 2010 --sheet 639_31_55 --e 317875 --n 6395875 \
    --scan --scan-cities gtb sth --scan-eras 2005 2010
echo "END seamcheck $(date)"
