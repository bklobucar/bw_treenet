#!/usr/bin/env python3
"""39_common_footprint.py -- canopy cover per city and era on ONE common footprint.

Why: eras cover different areas (e.g. Stockholm 36 frames in 1960-1990, 47 from 2005), so the per-era
totals of 20_aggregate_canopy.py mix change with coverage. Here, per city, the study area is the
intersection of (2023 tätort) and (valid imagery in EVERY series era). Every era -- series and extra --
is measured on that same area.

Grid: 1 m per 5 x 5 km sheet. Canopy maps (0/1, any resolution) are averaged onto it, so partial cells
count fractionally; imagery validity = source pixel > 0 (nearest). Map value 0 is NOT treated as nodata.

Also prints a short diagnosis of the 1990s label clips (why a clip can score zero): tätort share of the
clip, and mean source brightness / probability / canopy inside it for the 1990 and 1990r05 runs.

    python scripts/39_common_footprint.py                       # default eras
    python scripts/39_common_footprint.py --series 1960 1970 1990r05 2005 2010 --extra 1990 2005cmp 2005deg
Writes results/canopy_common_footprint.csv. gis env (GDAL Python bindings).
"""
import argparse, csv, glob, os, re
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
CITIES = ["malmo", "gtb", "sth"]
SFX = {"malmo": "mmo", "gtb": "gtb", "sth": "sth"}
TAT = "data/Tatorter_2023.gpkg"
INF = "results/inference_v19"
SHEET = re.compile(r"(\d{3}_\d{2}_\d{2})_(\d{4})")


def src_dir(city, era):
    if city == "gtb" and era == "1990":
        return "data/raw/gtb/1990_OF_rgb_gtb"
    return f"data/raw/{city}/{era}_OF_gray_{SFX[city]}"


def sheet_box(code):
    nnn, ee, ab = code.split("_")
    n0 = int(nnn) * 10_000 + int(ab[0]) * 1000
    e0 = int(ee) * 10_000 + int(ab[1]) * 1000
    return e0, n0, e0 + 5000, n0 + 5000


def index(city, era):
    """sheet -> (source path, canopy path) for one city-era (only sheets that have both)."""
    src = {}
    for f in glob.glob(os.path.join(src_dir(city, era), "*.tif")):
        m = SHEET.search(os.path.basename(f))
        if m:
            src[m.group(1)] = f
    out = {}
    for f in glob.glob(os.path.join(INF, f"{city}_{era}", "*_canopy.tif")):
        m = SHEET.search(os.path.basename(f))
        if m and m.group(1) in src:
            out[m.group(1)] = (src[m.group(1)], f)
    return out


def warp(path, box, res, alg, ftype=gdal.GDT_Byte):
    a = gdal.Warp("", path, format="MEM", outputBounds=box, xRes=res, yRes=res, resampleAlg=alg,
                  srcNodata="None", outputType=ftype).ReadAsArray()
    return a[0] if a.ndim == 3 else a


def tatort(box, res, tat_ds):
    nx, ny = int(round((box[2] - box[0]) / res)), int(round((box[3] - box[1]) / res))
    m = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    m.SetGeoTransform((box[0], res, 0, box[3], 0, -res))
    m.SetProjection(tat_ds.GetLayer(0).GetSpatialRef().ExportToWkt())
    gdal.Rasterize(m, tat_ds, burnValues=[1], layers=[tat_ds.GetLayer(0).GetName()])
    return m.ReadAsArray().astype(bool)


def common_footprint(a, tat_ds):
    rows = []
    for city in CITIES:
        idx = {e: index(city, e) for e in a.series + a.extra}
        missing = [e for e in a.series if not idx[e]]
        if missing:
            print(f"\n{city}: no outputs for series era(s) {missing} -- skipping city"); continue
        sheets = sorted(set.intersection(*[set(idx[e]) for e in a.series]))
        area = 0.0
        can = {e: 0.0 for e in a.series + a.extra}
        n_extra = {e: 0 for e in a.extra}
        for sh in sheets:
            box = sheet_box(sh)
            common = tatort(box, a.res, tat_ds)
            for e in a.series:
                common &= warp(idx[e][sh][0], box, a.res, "near") > 0
            if not common.any():
                continue
            area += common.sum()
            for e in a.series + a.extra:
                if sh not in idx[e]:
                    continue
                c = warp(idx[e][sh][1], box, a.res, "average", gdal.GDT_Float32)
                can[e] += float(c[common].sum())
                if e in n_extra:
                    n_extra[e] += 1
        km2 = area * a.res ** 2 / 1e6
        print(f"\n{city}: {len(sheets)} sheets in every series era, common area {km2:.1f} km2")
        for e in a.series + a.extra:
            pct = 100 * can[e] / area if area else float("nan")
            note = "" if e in a.series else f"  (extra; {n_extra[e]}/{len(sheets)} sheets present)"
            if e in a.extra and n_extra[e] == 0:
                continue
            print(f"  {e:8s} {pct:6.2f} %{note}")
            rows.append(dict(city=city, era=e, role="series" if e in a.series else "extra",
                             common_km2=round(km2, 2), canopy_pct=round(pct, 3),
                             sheets=len(sheets) if e in a.series else n_extra[e]))
    if rows:
        with open("results/canopy_common_footprint.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
        print("\nwrote results/canopy_common_footprint.csv")
        print("\n" + f"{'':8s}" + "".join(f"{e:>9s}" for e in a.series + a.extra))
        for city in CITIES:
            vals = {r["era"]: r["canopy_pct"] for r in rows if r["city"] == city}
            if vals:
                print(f"{city:8s}" + "".join(f"{vals[e]:9.2f}" if e in vals else f"{'-':>9s}" for e in a.series + a.extra))


def diagnose_clips(tat_ds):
    print("\n== 1990s label clips: why a clip can score zero")
    for city in CITIES:
        clip = f"data/raw/clips_for_labelling/{city}_1990_clip.tif"
        if not os.path.exists(clip):
            continue
        d = gdal.Open(clip); g = d.GetGeoTransform()
        box = (g[0], g[3] + g[5] * d.RasterYSize, g[0] + g[1] * d.RasterXSize, g[3])
        t = tatort(box, 1.0, tat_ds)
        print(f"{city}: clip E{box[0]:.0f}-{box[2]:.0f} N{box[1]:.0f}-{box[3]:.0f}, inside tätort {100*t.mean():.0f}%")
        for era in ("1990", "1990r05"):
            for kind in ("canopy", "prob"):
                hits = []
                for f in glob.glob(os.path.join(INF, f"{city}_{era}", f"*_{kind}.tif")):
                    fd = gdal.Open(f); fg = fd.GetGeoTransform()
                    fx0, fy1 = fg[0], fg[3]; fx1 = fx0 + fg[1] * fd.RasterXSize; fy0 = fy1 + fg[5] * fd.RasterYSize
                    if fx0 < box[2] and fx1 > box[0] and fy0 < box[3] and fy1 > box[1]:
                        hits.append(f)
                if not hits:
                    print(f"   {era:8s} {kind:6s} no map overlaps"); continue
                v = warp(hits, box, 1.0, "near", gdal.GDT_Float32)
                print(f"   {era:8s} {kind:6s} {len(hits)} map(s) {os.path.basename(hits[0])}: "
                      f"mean {v.mean():.3f}  max {v.max():.3f}  share>0 {100*(v > 0).mean():.1f}%")
        s = warp(clip, box, 1.0, "near", gdal.GDT_Float32)
        print(f"   clip image: mean {s.mean():.1f}, share>0 {100*(s > 0).mean():.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", nargs="+", default=["1960", "1970", "1990r05", "2005", "2010"])
    ap.add_argument("--extra", nargs="+", default=["1990", "2005cmp", "2005deg"])
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--no-diag", action="store_true")
    a = ap.parse_args()
    tat_ds = gdal.OpenEx(TAT, gdal.OF_VECTOR)
    common_footprint(a, tat_ds)
    if not a.no_diag:
        diagnose_clips(tat_ds)
