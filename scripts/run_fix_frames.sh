#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH -t 02:00:00
#SBATCH -J bw_fix_frames
#SBATCH -o /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_fix_frames_%j.out
#SBATCH -e /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_fix_frames_%j.err

module load GPU/Miniforge/26.3.2-2-eb
eval "$(conda shell.bash hook)"
conda activate bwtreenet
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet

python scripts/19_production_inference.py --city gtb   --era 1960
python scripts/19_production_inference.py --city sth   --era 1960
python scripts/19_production_inference.py --city sth   --era 1970
python scripts/19_production_inference.py --city malmo --era 1990
python scripts/20_aggregate_canopy.py
