#!/usr/bin/env python3
"""48_drop_edge_check.py -- do the ~2005 -> ~2010 window drops (46) cluster at sheet / frame edges?

Reads results/digital_flight_drop_windows.csv (46) and adds, per window centre:
  d_sheet    distance (m) to the nearest edge of its 5 x 5 km sheet;
  d_frame_a, d_frame_b   distance to the nearest edge of the inference frame (the source raster that 19
             ran on; its _prob.tif has the frame's grid) of era a and era b;
  d_patch_b  distance to the nearest inference patch edge in era b's frame, using 19's tiling
             (PATCH 1000, OVERLAP 200, stride 800, last patch snapped to the frame edge) -- a control:
             with seams every 400 m most windows have one within 200 m;
  img_zero_b % zero (nodata) pixels of era b's image inside the window.
Per city it prints: the share of windows dropping more than --drop pp by distance band (sheet and frame
edge); the drops vs all windows by patch-seam distance; the eastings / northings and the within-sheet
offsets that recur among the drops; every era-b image frame with zero fill along an edge or drops near
an edge (zero share and contiguous zero rows/columns per edge, drops within 500 m of that edge); and the
drop windows themselves with all distances. Stdout only.

    python scripts/48_drop_edge_check.py
gis env (GDAL Python bindings).
"""
import argparse, csv, importlib, os, sys
from collections import Counter, defaultdict
import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m43 = importlib.import_module("43_export_web_layers")   # imagery_index; via 41: index(); 39: sheet_box
m41, cf = m43.m41, m43.cf

gdal.UseExceptions()
CSV = "results/digital_flight_drop_windows.csv"
PATCH, OVERLAP = 1000, 200           # 19_production_inference.py
STRIDE = PATCH - OVERLAP
BANDS = [(0, 250), (250, 500), (500, 1000), (1000, float("inf"))]
EDGE_STRIP = 250.0                   # m, edge strip for the zero-fill check
NEAR_EDGE = 500.0                    # m, a drop "near" an edge


def positions(total, patch, stride):
    """As 19_production_inference.positions."""
    if total <= patch:
        return [0]
    pos = list(range(0, total - patch + 1, stride))
    if pos[-1] != total - patch:
        pos.append(total - patch)
    return pos


def frame_geom(path):
    d = gdal.Open(path); g = d.GetGeoTransform()
    x0, y1, W, H, res = g[0], g[3], d.RasterXSize, d.RasterYSize, g[1]
    xe = sorted({x0 + p * res for p in positions(W, PATCH, STRIDE)} | {x0 + (p + PATCH) * res for p in positions(W, PATCH, STRIDE)})
    ye = sorted({y1 - p * res for p in positions(H, PATCH, STRIDE)} | {y1 - (p + PATCH) * res for p in positions(H, PATCH, STRIDE)})
    return dict(box=(x0, y1 - H * res, x0 + W * res, y1), xe=np.array(xe), ye=np.array(ye))


def d_box(e, n, box):
    return min(e - box[0], box[2] - e, n - box[1], box[3] - n)


def band_table(rows, key, label):
    print(f"  share dropping by {label}:")
    print(f"    {'band (m)':>12s} {'windows':>8s} {'drops':>6s} {'share':>7s}")
    for lo, hi in BANDS:
        sel = [r for r in rows if lo <= r[key] < hi]
        nd = sum(r["drop"] for r in sel)
        hs = "inf" if hi == float("inf") else f"{hi:.0f}"
        print(f"    {f'{lo:.0f}-{hs}':>12s} {len(sel):8d} {nd:6d} {100 * nd / max(len(sel), 1):6.1f}%")


def edge_zero(arr, res):
    """-> {edge: (zero share in the edge strip %, contiguous all-zero rows/cols from the edge)}."""
    k = int(round(EDGE_STRIP / res))
    z = arr == 0
    strips = dict(left=z[:, :k], right=z[:, -k:], top=z[:k, :], bottom=z[-k:, :])
    full_cols, full_rows = z.all(axis=0), z.all(axis=1)

    def run(v):
        return int(np.argmin(v)) if not v.all() else len(v)
    runs = dict(left=run(full_cols), right=run(full_cols[::-1]), top=run(full_rows), bottom=run(full_rows[::-1]))
    return {e: (100 * strips[e].mean(), runs[e]) for e in strips}


def city_report(city, rows, a):
    ea, eb = a.era_a, a.era_b
    ia, ib = m41.index(city, ea), m41.index(city, eb)
    imgs = m43.imagery_index(city, eb)
    geo_a = {sh: frame_geom(v[2]) for sh, v in ia.items()}
    geo_b = {sh: frame_geom(v[2]) for sh, v in ib.items()}
    half = a.win / 2
    by_sheet = defaultdict(list)
    for r in rows:
        e, n, sh = r["centre_e"], r["centre_n"], r["sheet"]
        r["drop"] = r["change_pp"] < -a.drop
        r["d_sheet"] = d_box(e, n, cf.sheet_box(sh))
        r["d_frame_a"] = d_box(e, n, geo_a[sh]["box"]) if sh in geo_a else float("nan")
        r["d_frame_b"] = d_box(e, n, geo_b[sh]["box"]) if sh in geo_b else float("nan")
        if sh in geo_b:
            g = geo_b[sh]
            r["d_patch_b"] = min(np.abs(g["xe"] - e).min(), np.abs(g["ye"] - n).min())
        else:
            r["d_patch_b"] = float("nan")
        r["img_zero_b"] = float("nan")
        by_sheet[sh].append(r)

    drops = [r for r in rows if r["drop"]]
    print(f"\n== {city}: {len(rows)} windows, {len(drops)} drop more than {a.drop:.0f} pp "
          f"({100 * len(drops) / max(len(rows), 1):.1f}%)")
    mism = [abs(r["d_frame_b"] - r["d_sheet"]) for r in rows if r["d_frame_b"] == r["d_frame_b"]]
    if mism:
        print(f"  frame vs sheet edge distance, largest difference: {max(mism):.1f} m "
              f"({'frames are on the sheet grid' if max(mism) < 1 else 'some frames are NOT on the sheet grid'})")
    band_table(rows, "d_sheet", "distance to the sheet edge")
    band_table(rows, "d_frame_b", f"distance to the {eb} frame edge")
    print(f"  control, distance to the nearest {eb} inference patch edge:")
    for lo, hi in ((0, 50), (50, 125), (125, 1e9)):
        sel = [r for r in rows if lo <= r["d_patch_b"] < hi]
        nd = sum(r["drop"] for r in sel)
        print(f"    {lo:4.0f}-{'' if hi > 1e8 else f'{hi:.0f}':>4s} m  windows {len(sel):6d}  drops {nd:5d}  share {100 * nd / max(len(sel), 1):5.1f}%")

    if drops:
        print("  recurring centre coordinates among the drops (count >= 3):")
        for lab, key in (("easting", "centre_e"), ("northing", "centre_n")):
            rec = [(v, c) for v, c in Counter(r[key] for r in drops).most_common() if c >= 3]
            print(f"    {lab}: " + (", ".join(f"{v} x{c}" for v, c in rec[:12]) if rec else "none"))
        print(f"  within-sheet offset of the window centre (m from the sheet's west / south edge), drops vs all:")
        for lab, ax in (("E offset", 0), ("N offset", 1)):
            off_all = Counter(round((r["centre_e"], r["centre_n"])[ax] - cf.sheet_box(r["sheet"])[ax]) for r in rows)
            off_drop = Counter(round((r["centre_e"], r["centre_n"])[ax] - cf.sheet_box(r["sheet"])[ax]) for r in drops)
            exp = len(drops) / max(len(rows), 1)
            top = sorted(off_all, key=lambda o: -(off_drop[o] - exp * off_all[o]))[:6]
            print(f"    {lab}: " + ", ".join(f"{o} m: {off_drop[o]}/{off_all[o]} ({100 * off_drop[o] / off_all[o]:.0f}%)" for o in top)
                  + f"   (overall {100 * exp:.1f}%)")

    print(f"  {eb} image frames: zero share in the {EDGE_STRIP:.0f} m edge strip / contiguous all-zero rows or cols, "
          f"drops within {NEAR_EDGE:.0f} m of that edge (frames with zero fill or edge drops only)")
    shown = 0
    for sh in sorted(set(by_sheet) & set(imgs)):
        year, path = imgs[sh]
        d = gdal.Open(path)
        arr = d.GetRasterBand(1).ReadAsArray()
        g = d.GetGeoTransform(); res = g[1]
        x0, y1 = g[0], g[3]
        for r in by_sheet[sh]:
            c0 = int((r["centre_e"] - half - x0) / res); r0 = int((y1 - r["centre_n"] - half) / res)
            k = int(round(a.win / res))
            sub = arr[max(r0, 0):max(r0 + k, 0), max(c0, 0):max(c0 + k, 0)]
            r["img_zero_b"] = 100 * (sub == 0).mean() if sub.size else float("nan")
        ez = edge_zero(arr, res)
        box = (x0, y1 - d.RasterYSize * res, x0 + d.RasterXSize * res, y1)
        near = dict(left=lambda r: r["centre_e"] - box[0], right=lambda r: box[2] - r["centre_e"],
                    bottom=lambda r: r["centre_n"] - box[1], top=lambda r: box[3] - r["centre_n"])
        nd = {e: sum(1 for r in by_sheet[sh] if r["drop"] and near[e](r) < NEAR_EDGE) for e in near}
        if any(z > 0.5 or run > 0 for z, run in ez.values()) or any(nd.values()):
            shown += 1
            print(f"    {os.path.basename(path):22s} " + "  ".join(
                f"{e} {ez[e][0]:5.1f}%/{ez[e][1]:4d}px d{nd[e]}" for e in ("left", "right", "top", "bottom"))
                + f"   whole frame zero {100 * (arr == 0).mean():.1f}%")
    if not shown:
        print("    none")

    if drops:
        print(f"  drop windows ({len(drops)}):")
        print(f"    {'sheet':10s} {'E':>7s} {'N':>8s} {ea:>6s} {eb:>6s} {'change':>7s} {'d_sheet':>8s} {'d_frm_a':>8s} "
              f"{'d_frm_b':>8s} {'d_patch':>8s} {'img0_b':>7s}")
        for r in sorted(drops, key=lambda r: r["d_sheet"]):
            print(f"    {r['sheet']:10s} {r['centre_e']:7d} {r['centre_n']:8d} {r['cover_a']:6.1f} {r['cover_b']:6.1f} "
                  f"{r['change_pp']:+7.1f} {r['d_sheet']:8.0f} {r['d_frame_a']:8.0f} {r['d_frame_b']:8.0f} "
                  f"{r['d_patch_b']:8.0f} {r['img_zero_b']:6.1f}%")


def main(a):
    if not os.path.exists(CSV):
        sys.exit(f"[error] {CSV} not found; run 46_digital_flight_drop.py first")
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in ("centre_e", "centre_n"):
            r[k] = int(r[k])
        for k in ("cover_a", "cover_b", "change_pp", "valid_pct"):
            r[k] = float(r[k])
    for city in a.cities:
        cr = [r for r in rows if r["city"] == city]
        if cr:
            city_report(city, cr, a)
        else:
            print(f"\n== {city}: no windows in {CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="+", default=["gtb", "sth", "malmo"])
    ap.add_argument("--era-a", default="2005")
    ap.add_argument("--era-b", default="2010")
    ap.add_argument("--win", type=float, default=250.0, help="window size used by 46 (m)")
    ap.add_argument("--drop", type=float, default=20.0, help="a drop is a change below -this (pp)")
    main(ap.parse_args())
