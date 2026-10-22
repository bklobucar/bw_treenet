#!/usr/bin/env python3
"""43_export_web_layers.py -- v0.1 public web map layers (ArcGIS Online): canopy and imagery COGs.

Per city and era (1960 1970 1990r05 2005 2010, plus 2020 where the inference exists), in results/web_v01/:
  canopy/<city>_<era>_canopy_v19.tif   Byte, EPSG:3006, 0.5 m. 1 = canopy, 0 = measured non-canopy,
                                        255 = nodata (prob < 0: outside the 2023 tätort or water; or no
                                        sheet). DEFLATE COG, overviews resampled NEAREST.
  imagery/<city>_<era>_imagery.tif     source images of the sheets flown <= 2005 only, JPEG COG (q80).
                                        Imagery flown 2006 or later is NEVER exported.
  manifest.csv                         one row per file: flight years, STAC collection (where known),
                                        licence, licence_status, model, inference resolution, date.

Flight years come from the file names (<sheet>_<year>_canopy.tif, <sheet>_<year>.tif).
licence_status: "public" if every flight year <= 2005 (Lantmäteriet historical orthophotos, CC0),
"pending" otherwise (Ortofoto Nedladdning, CC BY 4.0, awaiting Lantmäteriet's answer).
COG options follow 22_build_usb_package.sh. Existing outputs are kept (a rerun resumes) unless --overwrite.
The manifest always lists every canopy/imagery file present, whichever layer --what built.

    python scripts/43_export_web_layers.py                  # canopy only (default)
    python scripts/43_export_web_layers.py --what imagery   # or: --what all
gis env (GDAL Python bindings).
"""
import argparse, csv, datetime, importlib, os, shutil, sys, tempfile
import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m41 = importlib.import_module("41_deso_aggregate")   # index(): sheet -> (year, canopy, prob); 39 helpers
cf = m41.cf

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("GDAL_CACHEMAX", "4096")
ERAS = ["1960", "1970", "1990r05", "2005", "2010"]
OPTIONAL = ["2020"]
PUBLIC_MAX_YEAR = m41.PUBLIC_MAX_YEAR
MODEL, INF_RES = "v19", 0.5
OUT = "results/web_v01"
QUALITY = 80
LICENCE = {"public": "CC0 1.0 (Lantmäteriet historiska ortofoton)",
           "pending": "CC BY 4.0 (Lantmäteriet Ortofoto Nedladdning); awaiting Lantmäteriet's answer"}
# STAC collections of the post-2000 picks (notes/paper2_notes_2026-10.md §2.1), keyed by flight year
STAC = {("malmo", 2004): "orto-historiska-2000-2004", ("gtb", 2008): "orto-goteborg-lerum-2008",
        ("sth", 2005): "orto-historiska-2005-2009", ("malmo", 2010): "orto-a-2010",
        ("gtb", 2012): "orto-storgoteborg-2012", ("sth", 2011): "orto-o2-2011",
        ("malmo", 2020): "orto-a2-2020", ("gtb", 2022): "orto-g2-2022", ("sth", 2019): "orto-o2-2019"}
COG_CANOPY = ["COMPRESS=DEFLATE", "PREDICTOR=2", "BLOCKSIZE=512", "NUM_THREADS=ALL_CPUS",
              "BIGTIFF=IF_SAFER", "OVERVIEW_RESAMPLING=NEAREST"]
COG_IMAGERY = ["COMPRESS=JPEG", f"QUALITY={QUALITY}", "BLOCKSIZE=512", "NUM_THREADS=ALL_CPUS", "BIGTIFF=IF_SAFER"]


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def status(years):
    return "public" if max(years) <= PUBLIC_MAX_YEAR else "pending"


def imagery_index(city, era):
    """sheet -> (flight year, source image) from data/raw/<city>/<era>_OF_gray_<sfx>/."""
    out = {}
    d = cf.src_dir(city, era)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            m = cf.SHEET.search(f)
            if m and f.endswith(".tif"):
                out[m.group(1)] = (int(m.group(2)), os.path.join(d, f))
    return out


def to_cog(srcs, out, src_nodata, options):
    """Mosaic srcs into one COG (EPSG:3006), written to <out>.part first so a crash leaves no partial file."""
    vrt = gdal.BuildVRT("", srcs, srcNodata=src_nodata, VRTNodata=src_nodata, resolution="highest")
    part = out + ".part"
    gdal.Translate(part, vrt, format="COG", outputSRS="EPSG:3006", creationOptions=options)
    vrt = None
    os.replace(part, out)


def recode_sheet(can_path, prob_path, tmp):
    """1 = canopy, 0 = measured non-canopy, 255 = prob < 0, on the canopy map's own grid."""
    c, p = gdal.Open(can_path), gdal.Open(prob_path)
    if (c.RasterXSize, c.RasterYSize, c.GetGeoTransform()) != (p.RasterXSize, p.RasterYSize, p.GetGeoTransform()):
        sys.exit(f"[error] {can_path} and {prob_path} are on different grids")
    out = np.where(p.GetRasterBand(1).ReadAsArray() < 0, 255,
                   c.GetRasterBand(1).ReadAsArray() > 0).astype(np.uint8)
    path = os.path.join(tmp, os.path.basename(can_path))
    d = gdal.GetDriverByName("GTiff").Create(path, c.RasterXSize, c.RasterYSize, 1, gdal.GDT_Byte,
                                             ["COMPRESS=DEFLATE", "TILED=YES"])
    d.SetGeoTransform(c.GetGeoTransform())
    d.SetProjection(c.GetProjection())
    b = d.GetRasterBand(1); b.WriteArray(out); b.SetNoDataValue(255)
    d = None
    return path


def row(layer, city, era, path, years, n_sheets):
    years = sorted(set(years))
    st = status(years)
    coll = sorted({STAC[(city, y)] for y in years if (city, y) in STAC})
    return dict(layer=layer, city=city, era=era, file=os.path.relpath(path, OUT),
                flight_years=";".join(map(str, years)), stac_collection=";".join(coll),
                licence=LICENCE[st], licence_status=st,
                model=MODEL if layer == "canopy" else "", inference_resolution_m=INF_RES if layer == "canopy" else "",
                sheets=n_sheets, size_mb=round(os.path.getsize(path) / 1e6, 1),
                date=datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat())


def main(a):
    os.makedirs(os.path.join(OUT, "canopy"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "imagery"), exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="web_v01_", dir=os.environ.get("TMPDIR"))
    rows = []
    try:
        for city in a.cities:
            eras = ERAS + [e for e in OPTIONAL if m41.index(city, e)]
            for era in eras:
                idx = m41.index(city, era)
                if not idx:
                    sys.exit(f"[error] {city}: no inference for era {era}")
                log(f"== {city} {era}: {len(idx)} sheets, flight years {sorted({v[0] for v in idx.values()})}")

                out = os.path.join(OUT, "canopy", f"{city}_{era}_canopy_{MODEL}.tif")
                if a.what in ("canopy", "all"):
                    if os.path.exists(out) and not a.overwrite:
                        log("   canopy exists, kept")
                    else:
                        tmp = tempfile.mkdtemp(dir=tmp_root)
                        srcs = [recode_sheet(can, prob, tmp) for _, can, prob in idx.values()]
                        to_cog(srcs, out, 255, COG_CANOPY)
                        shutil.rmtree(tmp)
                        log(f"   canopy -> {out}")
                if os.path.exists(out):
                    rows.append(row("canopy", city, era, out, [v[0] for v in idx.values()], len(idx)))

                img = imagery_index(city, era)
                pub = {sh: v for sh, v in img.items() if v[0] <= PUBLIC_MAX_YEAR}
                out = os.path.join(OUT, "imagery", f"{city}_{era}_imagery.tif")
                if a.what in ("imagery", "all"):
                    if len(pub) < len(img):
                        log(f"   imagery: {len(img) - len(pub)} sheet(s) flown >= {PUBLIC_MAX_YEAR + 1} not exported")
                    if not pub:
                        log("   imagery: nothing <= 2005 to export")
                    elif os.path.exists(out) and not a.overwrite:
                        log("   imagery exists, kept")
                    else:
                        srcs = [p for _, p in pub.values()]
                        opts = COG_IMAGERY + (["PHOTOMETRIC=YCBCR"] if gdal.Open(srcs[0]).RasterCount == 3 else [])
                        to_cog(srcs, out, 0, opts)
                        log(f"   imagery -> {out}")
                if pub and os.path.exists(out):
                    years = [v[0] for v in pub.values()]
                    assert max(years) <= PUBLIC_MAX_YEAR      # never export imagery flown 2006 or later
                    rows.append(row("imagery", city, era, out, years, len(pub)))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if not rows:
        sys.exit("[error] no web layers present; manifest not written")
    man = os.path.join(OUT, "manifest.csv")
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    log(f"wrote {man} ({len(rows)} files, {sum(r['size_mb'] for r in rows) / 1e3:.2f} GB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["canopy", "imagery", "all"], default="canopy")
    ap.add_argument("--cities", nargs="+", default=cf.CITIES)
    ap.add_argument("--overwrite", action="store_true", help="rebuild COGs that already exist")
    main(ap.parse_args())
