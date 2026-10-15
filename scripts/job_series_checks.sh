#!/bin/bash -l
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -J series_chk
#SBATCH -o results/logs/%x_%j.out
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate gis
echo "== build 2005deg (0.5 m -> 1 m average -> 0.5 m bilinear)"
for p in malmo:mmo gtb:gtb sth:sth; do c=${p%%:*}; s=${p##*:}; mkdir -p data/raw/$c/2005deg_OF_gray_$s
  for f in data/raw/$c/2005_OF_gray_$s/*.tif; do o=data/raw/$c/2005deg_OF_gray_$s/$(basename $f); [ -f $o ] && continue
    gdalwarp -q -overwrite -tr 1 1 -r average -srcnodata 0 -dstnodata 0 $f /tmp/deg_$$.tif && gdalwarp -q -overwrite -tr 0.5 0.5 -r bilinear -srcnodata 0 -dstnodata 0 -ot Byte -co COMPRESS=LZW -co TILED=YES /tmp/deg_$$.tif $o
  done; echo "$c: $(ls data/raw/$c/2005deg_OF_gray_$s | wc -l) sheets"; done
rm -f /tmp/deg_$$.tif
source ~/bw_setup.sh > /dev/null 2>&1
echo "== inference 2005deg"
for c in malmo gtb sth; do python -u scripts/19_production_inference.py --city $c --era 2005deg 2>&1 | grep "canopy cover\|Error\|Traceback"; done
conda activate gis
echo "== common footprint"
python -u scripts/39_common_footprint.py 2>&1 | grep -v -i warn
