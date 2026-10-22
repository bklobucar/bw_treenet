#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J normfix
#SBATCH -o results/logs/%x_%j.out
# Norm-fix test (49): production 1000/200 vs 2000/400 patches vs fixed sheet-level norms, on the test
# sheets, the 1990s label clips, v19's held-out validation tiles and a modern tile sample.
# Writes results/norm_fix_test/ (summary.csv and 639_31_55 PNGs; not replaced if they exist).
# Submit from the repo root:  sbatch scripts/job_norm_fix.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate bwtreenet
set -euo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

echo "START normfix $(date)  on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/49_norm_fix_test.py
echo "END normfix $(date)"
