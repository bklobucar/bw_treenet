#!/bin/bash -l
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J era_infer
#SBATCH -o results/logs/%x_%j.out
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
source ~/bw_setup.sh
for ce in malmo:2005 gtb:2005 sth:2005 gtb:2005cmp malmo:2010 gtb:2010 sth:2010; do
  python scripts/19_production_inference.py --city ${ce%%:*} --era ${ce##*:} 2>&1 | grep -v Warning
done
