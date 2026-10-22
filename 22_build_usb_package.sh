#!/bin/bash
# 22_build_usb_package.sh
# Build a portable QGIS package from bw_treenet v19 production outputs.
# Run from the repo root on a compute node with the `gis` conda env active:
#   nohup bash 22_build_usb_package.sh > usb_build.log 2>&1 &
#
# Output layout (all EPSG:3006, COG):
#   $OUT/imagery/<city>_<era>.tif      JPEG q80, source aerial frames
#   $OUT/canopy/<city>_<era>.tif       0/1 canopy, nodata 0
#   $OUT/change/<city>_<a>_<b>.tif     1 gain, 2 loss, 3 stable, nodata 0
#   $OUT/vectors/coverage.gpkg         frame footprints per city-era
#   $OUT/vectors/hydro.gpkg            Lantmateriet hydrography per city
#   $OUT/vectors/Tatorter_2023.gpkg    SCB tatort boundary
#   $OUT/canopy_change_v19.csv

set -euo pipefail

OUT=/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bwtreenet_usb
TMP=/tmp/usb_build_$$
QUALITY=80

# ---------------------------------------------------------------------------
# sanity checks
for tool in gdalbuildvrt gdal_translate gdaltindex ogr2ogr; do
  command -v $tool >/dev/null || { echo "ERROR: $tool not found; activate the gis env"; exit 1; }
done
GDALCALC=$(command -v gdal_calc.py || command -v gdal_calc || true)
[[ -n $GDALCALC ]] || { echo "ERROR: gdal_calc not found"; exit 1; }
[[ -d results/inference_v19 && -d data/raw ]] || { echo "ERROR: run from the bw_treenet repo root"; exit 1; }

mkdir -p $OUT/{imagery,canopy,change,vectors} $TMP
export GDAL_NUM_THREADS=ALL_CPUS
export GDAL_CACHEMAX=4096
trap 'rm -rf $TMP' EXIT

declare -A IMG=(
  [malmo_1960]=data/raw/malmo/1960_OF_gray_mmo
  [malmo_1970]=data/raw/malmo/1970_OF_gray_mmo
  [malmo_1990]=data/raw/malmo/1990_OF_gray_mmo
  [gtb_1960]=data/raw/gtb/1960_OF_gray_gtb
  [gtb_1970]=data/raw/gtb/1970_OF_gray_gtb
  [gtb_1990]=data/raw/gtb/1990_OF_rgb_gtb
  [sth_1960]=data/raw/sth/1960_OF_gray_sth
  [sth_1970]=data/raw/sth/1970_OF_gray_sth
  [sth_1990]=data/raw/sth/1990_OF_gray_sth
)
ORDER="malmo_1960 malmo_1970 malmo_1990 gtb_1960 gtb_1970 gtb_1990 sth_1960 sth_1970 sth_1990"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# per city-era: imagery, canopy, coverage footprint
for ce in $ORDER; do
  src=${IMG[$ce]}
  can=results/inference_v19/$ce
  log "== $ce"

  if [[ ! -d $src ]]; then log "   WARNING: imagery dir $src missing, skipping imagery"; else
    if [[ -f $OUT/imagery/$ce.tif ]]; then log "   imagery exists, skipping"; else
      log "   imagery ($(ls $src/*.tif | wc -l) frames)"
      gdalbuildvrt -q -srcnodata 0 -vrtnodata 0 $TMP/${ce}_img.vrt $src/*.tif
      photo=""; [[ $ce == gtb_1990 ]] && photo="-co PHOTOMETRIC=YCBCR"
      gdal_translate -q $TMP/${ce}_img.vrt $OUT/imagery/$ce.tif -of COG \
        -co COMPRESS=JPEG -co QUALITY=$QUALITY $photo -co BLOCKSIZE=512 \
        -co NUM_THREADS=ALL_CPUS -co BIGTIFF=IF_SAFER
    fi
  fi

  if [[ ! -d $can ]]; then log "   WARNING: canopy dir $can missing, skipping"; continue; fi
  if [[ -f $OUT/canopy/$ce.tif ]]; then log "   canopy exists, skipping"; else
    log "   canopy ($(ls $can/*_canopy.tif | wc -l) frames)"
    gdalbuildvrt -q -srcnodata 0 -vrtnodata 0 $TMP/${ce}_can.vrt $can/*_canopy.tif
    gdal_translate -q $TMP/${ce}_can.vrt $OUT/canopy/$ce.tif -of COG \
      -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKSIZE=512 \
      -co NUM_THREADS=ALL_CPUS -co BIGTIFF=IF_SAFER
  fi

  log "   coverage footprint"
  gdaltindex -f GPKG -overwrite -lyr_name coverage_$ce -t_srs EPSG:3006 \
    $OUT/vectors/coverage.gpkg $can/*_canopy.tif >/dev/null
done

# ---------------------------------------------------------------------------
# change rasters: 1 gain, 2 loss, 3 stable, 0 nodata
# built from VRTs with nodata disabled so 0 = "no canopy" inside coverage
change() {
  local city=$1 a=$2 b=$3
  local out=$OUT/change/${city}_${a}_${b}.tif
  [[ -f $out ]] && { log "   change $city $a->$b exists, skipping"; return; }
  [[ -d results/inference_v19/${city}_$a && -d results/inference_v19/${city}_$b ]] || return 0
  log "   change $city $a->$b"
  gdalbuildvrt -q -srcnodata None -tr 0.5 0.5 -r nearest $TMP/${city}_${a}.vrt results/inference_v19/${city}_$a/*_canopy.tif
  gdalbuildvrt -q -srcnodata None -tr 0.5 0.5 -r nearest $TMP/${city}_${b}.vrt results/inference_v19/${city}_$b/*_canopy.tif
  $GDALCALC -A $TMP/${city}_${a}.vrt -B $TMP/${city}_${b}.vrt \
    --outfile=$TMP/${city}_${a}_${b}.tif --type=Byte --NoDataValue=0 --extent=intersect \
    --calc="1*(A==0)*(B==1) + 2*(A==1)*(B==0) + 3*(A==1)*(B==1)" \
    --co COMPRESS=DEFLATE --co TILED=YES --co BIGTIFF=IF_SAFER --quiet
  gdal_translate -q $TMP/${city}_${a}_${b}.tif $out -of COG \
    -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKSIZE=512 -co NUM_THREADS=ALL_CPUS
  rm -f $TMP/${city}_${a}_${b}.tif
}

log "== change rasters"
for city in malmo gtb sth; do
  change $city 1960 1990
  change $city 1960 1970
  change $city 1970 1990
done

# ---------------------------------------------------------------------------
# vectors and tables
log "== vectors"
cp -n data/Tatorter_2023.gpkg $OUT/vectors/ 2>/dev/null || true
rm -f $OUT/vectors/hydro.gpkg
ogr2ogr -f GPKG         $OUT/vectors/hydro.gpkg data/hydro/1/mv_south.shp -nln hydro_sth   -t_srs EPSG:3006
ogr2ogr -f GPKG -update $OUT/vectors/hydro.gpkg data/hydro/2/mv_south.shp -nln hydro_malmo -t_srs EPSG:3006
ogr2ogr -f GPKG -update $OUT/vectors/hydro.gpkg data/hydro/3/mv_south.shp -nln hydro_gtb   -t_srs EPSG:3006
cp results/canopy_change_v19.csv $OUT/

# ---------------------------------------------------------------------------
log "== done"
du -sh $OUT/* $OUT
