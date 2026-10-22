#!/usr/bin/env python3
"""45_stand_prob_check.py -- why does a digital era call one closed stand canopy and miss the next?

Takes the windows from the row labels that 40_film_digital_windows.py printed to its log, for one strip
(--key, e.g. gtb_pan2000). For each window and each digital era (--eras, default 2005 2010):
  * prints the probability histogram (0.1 bins) of measured pixels (prob >= 0) inside the window,
    split into predicted canopy (canopy map = 1) and predicted non-canopy (= 0), with mean, median and
    the share of non-canopy pixels just under the threshold (0.4-0.6) -- near misses vs confident misses;
  * writes results/stand_prob_check/<key>_row<N>.png: one row per era, columns image | probability |
    mask, and a colour bar under the probability column.
    Probability: viridis, 0 (dark purple) -> 1 (yellow); white ticks on the bar at 0.5 and at the
    production threshold 0.6; grey = prob < 0 (outside tätort or water); dark red = outside the map.
    Mask: white canopy, black non-canopy, grey prob < 0, dark red outside the map.

    python scripts/45_stand_prob_check.py --log results/logs/film_fill_4233698.out --key gtb_pan2000 --rows 3 6 7
gis env (GDAL Python bindings).
"""
import argparse, glob, importlib, os, sys
import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cf = importlib.import_module("39_common_footprint")      # warp (srcNodata="None"), src_dir, INF
m44 = importlib.import_module("44_strip_line_check")     # parse_log, covered

gdal.UseExceptions()
RES = 0.5
THRESHOLD = 0.6                    # 19's production threshold
OUT_DIR = "results/stand_prob_check"
BINS = np.round(np.arange(0, 1.01, 0.1), 2)
# viridis, 5 stops, linearly interpolated
VIRIDIS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float32)
GREY, DARK_RED = np.array([128, 128, 128]), np.array([120, 0, 0])


def viridis(p):
    """(k, k) values in [0, 1] -> (3, k, k) uint8."""
    x = np.clip(p, 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(x.astype(int), len(VIRIDIS) - 2)
    f = (x - i)[..., None]
    rgb = VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f
    return np.moveaxis(rgb.astype(np.uint8), -1, 0)


def load(city, era, sh, wb, k):
    """-> dict(img, img_cov, can, prob, cov, year) for one sheet and era, or None if there is no map."""
    can_p = next(iter(glob.glob(os.path.join(cf.INF, f"{city}_{era}", f"{sh}_*_canopy.tif"))), None)
    if not can_p:
        return None
    prob_p = can_p.replace("_canopy.tif", "_prob.tif")
    img_p = next(iter(glob.glob(os.path.join(cf.src_dir(city, era), f"{sh}_*.tif"))), None)
    cov = m44.covered(can_p, wb, k)
    d = dict(can=cf.warp(can_p, wb, RES, "near")[:k, :k], cov=cov, year=os.path.basename(can_p).split("_")[3])
    d["prob"] = cf.warp(prob_p, wb, RES, "near", gdal.GDT_Float32)[:k, :k] if os.path.exists(prob_p) else None
    if img_p:
        d["img"], d["img_cov"] = cf.warp(img_p, wb, RES, "near", gdal.GDT_Float32)[:k, :k], m44.covered(img_p, wb, k)
    else:
        d["img"], d["img_cov"] = np.zeros((k, k), np.float32), np.zeros((k, k), bool)
    return d


def histogram(d):
    p, c, cov = d["prob"], d["can"] > 0, d["cov"]
    meas = cov & (p >= 0)
    n = cov.sum()
    print(f"     measured {100 * meas.sum() / max(n, 1):.0f}% of the window, "
          f"prob < 0 {100 * (cov & (p < 0)).sum() / max(n, 1):.0f}%, outside the map {100 * (~cov).mean():.0f}%")
    groups = (("canopy", p[meas & c]), ("non-canopy", p[meas & ~c]))
    hist = {g: np.histogram(v, bins=BINS)[0] for g, v in groups}
    print(f"     {'prob bin':10s}" + "".join(f"{g:>36s}" for g, _ in groups))
    for b in range(len(BINS) - 1):
        cells = []
        for g, v in groups:
            share = 100 * hist[g][b] / max(v.size, 1)
            cells.append(f"{hist[g][b]:9d} {share:5.1f}% {'#' * int(round(share / 5)):<20s}")
        print(f"     {BINS[b]:.1f}-{BINS[b + 1]:.1f}   " + " ".join(cells))
    for g, v in groups:
        if v.size:
            print(f"     {g:10s} n {v.size:7d}  mean {v.mean():.3f}  median {np.median(v):.3f}")
    nc = groups[1][1]
    if nc.size:
        print(f"     non-canopy near misses (0.4 <= prob < {THRESHOLD}): {100 * ((nc >= 0.4) & (nc < THRESHOLD)).mean():.1f}%  "
              f"confident (< 0.2): {100 * (nc < 0.2).mean():.1f}%")


def panels(d, k):
    v = d["img"][d["img_cov"]]
    lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 1)
    g8 = np.clip((d["img"] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    g8[~d["img_cov"]] = 0
    img = np.stack([g8] * 3)
    if d["prob"] is not None:
        pr = viridis(d["prob"])
        pr[:, d["prob"] < 0] = GREY[:, None]
    else:
        pr = np.full((3, k, k), 60, np.uint8)
    pr[:, ~d["cov"]] = DARK_RED[:, None]
    m = np.where(d["can"] > 0, 255, 0).astype(np.uint8)
    mk = np.stack([m] * 3)
    if d["prob"] is not None:
        mk[:, (d["prob"] < 0) & (d["can"] == 0)] = GREY[:, None]
    mk[:, ~d["cov"]] = DARK_RED[:, None]
    return [img, pr, mk]


def save_png(path, rows, k):
    gap, bar_h = 6, 24
    H = len(rows) * (k + gap) + bar_h
    W = 3 * k + 2 * gap
    rgb = np.full((3, H, W), 255, np.uint8)
    for r, ps in enumerate(rows):
        for c, p in enumerate(ps):
            if p is not None:
                rgb[:, r * (k + gap):r * (k + gap) + k, c * (k + gap):c * (k + gap) + k] = p
    y, x = len(rows) * (k + gap), k + gap                    # colour bar under the probability column
    rgb[:, y:y + bar_h - 8, x:x + k] = viridis(np.tile(np.linspace(0, 1, k), (bar_h - 8, 1)))
    for t in (0.5, THRESHOLD):
        tx = x + int(round(t * (k - 1)))
        rgb[:, y:y + bar_h, tx - 1:tx + 1] = 255 if t == THRESHOLD else 200
    rgb[:, y + bar_h - 8:y + bar_h, x:x + k] = 255
    mem = gdal.GetDriverByName("MEM").Create("", W, H, 3, gdal.GDT_Byte)
    for b in range(3):
        mem.GetRasterBand(b + 1).WriteArray(rgb[b])
    gdal.GetDriverByName("PNG").CreateCopy(path, mem)


def main(a):
    city = a.key.split("_")[0]
    rows = m44.parse_log(a.log, a.key)
    os.makedirs(OUT_DIR, exist_ok=True)
    k = int(round(a.win / RES))
    print(f"strip {a.key}, eras {a.eras}; PNG rows = eras in that order, columns = image | probability | mask")
    for r in a.rows:
        if r not in rows:
            print(f"\n== row {r}: not in the log"); continue
        sh, ce, cn, rest = rows[r]
        wb = (ce - a.win / 2, cn - a.win / 2, ce + a.win / 2, cn + a.win / 2)
        print(f"\n== row {r}: {sh}  centre E{ce:.0f} N{cn:.0f}  ({rest})")
        png_rows = []
        for era in a.eras:
            d = load(city, era, sh, wb, k)
            if d is None:
                print(f"   {era}: no canopy map for {sh}")
                png_rows.append([None, None, None]); continue
            print(f"   {era} (flight {d['year']}): canopy {100 * ((d['can'] > 0) & d['cov']).sum() / max(d['cov'].sum(), 1):.0f}% of the window")
            if d["prob"] is None:
                print("     no _prob.tif; histogram skipped")
            else:
                histogram(d)
            png_rows.append(panels(d, k))
        out = os.path.join(OUT_DIR, f"{a.key}_row{r}.png")
        if os.path.exists(out) and not a.overwrite:
            print(f"   {out} exists, not replaced (--overwrite)")
        else:
            save_png(out, png_rows, k)
            print(f"   wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="log of the 40 run with the row labels")
    ap.add_argument("--key", default="gtb_pan2000", help="strip key <city>_<film>, as in film_windows_<key>_fill.png")
    ap.add_argument("--rows", type=int, nargs="+", default=[3, 6, 7])
    ap.add_argument("--eras", nargs="+", default=["2005", "2010"], help="digital eras, one PNG row each")
    ap.add_argument("--win", type=float, default=250.0, help="window size in m (40's --win)")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
