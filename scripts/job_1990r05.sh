#!/bin/bash -l
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J res1990
#SBATCH -o results/logs/%x_%j.out
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
module load GPU/Miniforge/26.3.2-2-eb
eval "$(conda shell.bash hook)"
conda activate gis
echo "== training tile and clip resolution"
for c in malmo gtb sth; do f=$(find data/processed/$c/tiles_1990 -name "*.tif" | head -1); echo "$c tiles_1990: $(find data/processed/$c/tiles_1990 -name '*.tif' | wc -l) files, $(gdalinfo $f | grep 'Pixel Size')"; done
for f in data/raw/clips_for_labelling/*_1990_clip.tif; do echo "$f: $(gdalinfo $f | grep 'Pixel Size')"; done
echo "== output naming"; ls results/inference_v19/malmo_1990r05 | head -6
echo "== build"; python scripts/37_res_test_1990s.py build --per-city 999 2>&1 | grep -v -i warn
source ~/bw_setup.sh
echo "== inference"
for c in malmo gtb sth; do python scripts/19_production_inference.py --city $c --era 1990r05 2>&1 | grep -v Warning; done
