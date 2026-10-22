#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -J film_fill
#SBATCH -o results/logs/%x_%j.out
# Filled image strips for the three late film flights (40 --fill --top 12 --strips-only).
# Writes results/film_windows_{gtb_pan2000,sth_pan1999,sth_pan2000}_fill.png; existing PNGs are not replaced.
# Submit from the repo root:  sbatch scripts/job_film_fill.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START film_fill $(date)  on $(hostname)"
python scripts/40_film_digital_windows.py --city gtb --film pan2000 --digital 2005 --show 1990:1990r05 pan2000 2005 2010 --fill --top 12 --strips-only
python scripts/40_film_digital_windows.py --city sth --film pan1999 --digital 2005 --show 1990:1990r05 pan1999 2005 2010 --fill --top 12 --strips-only
python scripts/40_film_digital_windows.py --city sth --film pan2000 --digital 2005 --show 1990:1990r05 pan2000 2005 2010 --fill --top 12 --strips-only
echo "END film_fill $(date)"
