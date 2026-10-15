#!/usr/bin/env python3
"""37_res_test_1990s.py -- does running v19 on the 1990s at native 1 m (as production does) under-call canopy
compared with the same imagery resampled to 0.5 m, the resolution v19 was trained at?

Step 1 (build): copies a spread of 1990s sheets per city, resampled 1 m -> 0.5 m, into
        data/raw/<city>/1990r05_OF_gray_<suffix>/   (same file names, Byte, nodata 0)
Step 2 (you):   add ("<city>", "1990r05") entries to SOURCES in 19_production_inference.py and run it
Step 3 (compare): canopy % per sheet, 1 m run vs 0.5 m run, from the binary outputs in results/inference_v19/

    python scripts/37_res_test_1990s.py build   [--per-city 4] [--resampling bilinear]
    python scripts/37_res_test_1990s.py compare
Run in the gis env (GDAL CLI + Python bindings).
"""
import argparse, csv, glob, os, re, subprocess, sys
import numpy as np
from osgeo import gdal

SRC = {"malmo": "data/raw/malmo/1990_OF_gray_mmo", "gtb": "data/raw/gtb/1990_OF_rgb_gtb", "sth": "data/raw/sth/1990_OF_gray_sth"}
SUFFIX = {"malmo": "mmo", "gtb": "gtb", "sth": "sth"}
SHEET = re.compile(r"(\d{3}_\d{2}_\d{2})")
ERA_T = "1990r05"


def test_dir(city):
    return f"data/raw/{city}/{ERA_T}_OF_gray_{SUFFIX[city]}"


def build(a):
    for city, src in SRC.items():
        files = sorted(glob.glob(os.path.join(src, "*.tif")))
        if not files:
            print(f"{city}: no 1990s sheets in {src}"); continue
        step = max(1, len(files) // a.per_city)
        pick = files[::step][:a.per_city]
        os.makedirs(test_dir(city), exist_ok=True)
        for f in pick:
            out = os.path.join(test_dir(city), os.path.basename(f))
            ds = gdal.Open(f); res = ds.GetGeoTransform()[1]; ds = None
            print(f"{city} {os.path.basename(f)}  {res:.2f} m -> 0.5 m")
            subprocess.run(["gdalwarp", "-overwrite", "-tr", "0.5", "0.5", "-tap", "-r", a.resampling, "-srcnodata", "0",
                            "-dstnodata", "0", "-ot", "Byte", "-co", "COMPRESS=LZW", "-co", "TILED=YES", f, out], check=True,
                           capture_output=True)
    print(f"\nnext: add ('<city>', '{ERA_T}') to SOURCES in scripts/19_production_inference.py and run it for each city")


def frac(binary_path, source_path):
    b = gdal.Open(binary_path); s = gdal.Open(source_path)
    bb = b.GetRasterBand(1).ReadAsArray()
    # source may be at a different resolution than the binary (1 m vs 0.5 m) -> read it on the binary's grid
    sa = s.GetRasterBand(1).ReadAsArray(buf_xsize=b.RasterXSize, buf_ysize=b.RasterYSize)
    valid = sa > 0
    return float(bb[valid].astype(bool).mean()) if valid.any() else float("nan"), int(valid.sum())


def find_out(city, era, sheet):
    hits = [p for p in glob.glob(f"results/inference_v19/{city}_{era}/**/*.tif", recursive=True) if sheet in os.path.basename(p)]
    return hits[0] if hits else None


def compare(a):
    rows = []
    for city in SRC:
        for f in sorted(glob.glob(os.path.join(test_dir(city), "*.tif"))):
            sheet = SHEET.search(os.path.basename(f)).group(1)
            o1, o2 = find_out(city, "1990", sheet), find_out(city, ERA_T, sheet)
            if not (o1 and o2):
                print(f"{city} {sheet}: missing output ({'1 m' if not o1 else '0.5 m'} run)"); continue
            src1 = os.path.join(SRC[city], os.path.basename(f))
            f1, _ = frac(o1, src1); f2, _ = frac(o2, f)
            rows.append((city, sheet, 100 * f1, 100 * f2))
    if not rows:
        sys.exit("nothing to compare yet")
    print(f"\n{'city':6s} {'sheet':10s} {'1 m %':>7s} {'0.5 m %':>8s} {'diff pp':>8s} {'ratio':>6s}")
    for c, s, x, y in rows:
        print(f"{c:6s} {s:10s} {x:7.1f} {y:8.1f} {y - x:+8.1f} {y / x if x else float('nan'):6.2f}")
    for c in SRC:
        r = [x for x in rows if x[0] == c]
        if r:
            d = np.array([y - x for _, _, x, y in r])
            print(f"{c:6s} mean difference {d.mean():+.1f} pp over {len(r)} sheets (min {d.min():+.1f}, max {d.max():+.1f})")
    with open("results/res_test_1990s.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["city", "sheet", "cover_1m_pct", "cover_05m_pct"]); w.writerows(rows)
    print("\nwrote results/res_test_1990s.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "compare"])
    ap.add_argument("--per-city", type=int, default=4)
    ap.add_argument("--resampling", default="bilinear", help="1 m -> 0.5 m upsampling (default bilinear)")
    a = ap.parse_args()
    build(a) if a.step == "build" else compare(a)
