#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -J v18a
#SBATCH -o /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_v18a_%j.out
#SBATCH -e /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_v18a_%j.err

module load GPU/Miniforge/26.3.2-2-eb
eval "$(conda shell.bash hook)"
conda activate bwtreenet

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
echo "START v18a $(date)"
python scripts/07_train_v18.py --batch 8 --tag v18a
echo "END v18a $(date)"
