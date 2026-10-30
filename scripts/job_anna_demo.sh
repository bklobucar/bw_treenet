#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 2
#SBATCH -t 00:20:00
#SBATCH -J anna_demo
#SBATCH -o results/logs/%x_%j.out
# Demo windows for the labelling walkthrough (53, windows only): demo_windows.csv -> results/anna_demo/ and
# results/anna_demo.zip. No test points, keys, log.csv or styles; results/anna_labelling(_key) are not touched
# (53 refuses an --out-dir there). Existing demo windows are kept; pass --overwrite to rebuild them.
# Submit from the repo root:  sbatch scripts/job_anna_demo.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
set -eo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
echo "START anna_demo $(date)  on $(hostname)"
python -u scripts/53_build_label_queue.py --windows-csv demo_windows.csv --out-dir results/anna_demo "$@"
du -sh results/anna_demo results/anna_demo.zip
echo "END anna_demo $(date)"
