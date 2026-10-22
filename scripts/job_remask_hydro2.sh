#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J remask_hydro2
#SBATCH -o results/logs/%x_%j.out
# hydro_v2 re-masking without new inference (51), then web v0.1 from the re-masked maps:
# DeSO aggregation (41) -> results/deso_hydro2/, canopy COGs + manifest (43) -> results/web_v01_hydro2/.
# The original maps (results/inference_v19/) and v0.1 outputs (results/deso/, results/web_v01/) are not touched.
# Imagery COGs do not depend on the water mask: reuse results/web_v01/imagery/.
# Submit from the repo root:  sbatch scripts/job_remask_hydro2.sh

module load GPU/Miniforge/26.3.2-2-eb > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate bwtreenet
set -eo pipefail

cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
INF2=results/inference_v19_hydro2
DESO=data/ext/DeSO_2025.gpkg

echo "START remask_hydro2 $(date)  on $(hostname), $SLURM_CPUS_PER_TASK CPUs"
for f in data/hydro_v2/malmo_water.gpkg data/hydro_v2/gtb_water.gpkg data/hydro_v2/sth_water.gpkg $DESO; do
  [[ -f $f ]] || { echo "ERROR: $f not found"; exit 1; }
done
python scripts/51_remask_water.py --workers 12
echo "51 done $(date)"

conda deactivate
conda activate gis
python scripts/41_deso_aggregate.py --deso $DESO --inf $INF2 --out-dir results/deso_hydro2
echo "41 done $(date)"
python scripts/43_export_web_layers.py --what canopy --inf $INF2 --out results/web_v01_hydro2
echo "43 done $(date)"
du -sh $INF2 results/deso_hydro2 results/web_v01_hydro2
echo "END remask_hydro2 $(date)"
