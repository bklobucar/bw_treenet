#!/bin/bash -l
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 01:00:00
#SBATCH -J film_windows
#SBATCH -o results/logs/%x_%j.out
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
python -u scripts/40_film_digital_windows.py --city gtb --film pan2000 --digital 2005 --show 1990:1990r05 pan2000 2005 2010 2>&1 | grep -v -i warn | grep -v ": done"
python -u scripts/40_film_digital_windows.py --city sth --film pan1999 --digital 2005 --show 1990:1990r05 pan1999 pan2000 2005 2010 2>&1 | grep -v -i warn | grep -v ": done"
python -u scripts/40_film_digital_windows.py --city sth --film pan2000 --digital 2005 --show 1990:1990r05 pan1999 pan2000 2005 2010 2>&1 | grep -v -i warn | grep -v ": done"
