#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH -t 02:00:00
#SBATCH -J bwtreenet_inference
#SBATCH -o /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_v13_inference_%j.out
#SBATCH -e /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_v13_inference_%j.err

module load GPU/Miniforge/26.3.2-2-eb
eval "$(conda shell.bash hook)"
conda activate bwtreenet

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

python scripts/08_inference.py
