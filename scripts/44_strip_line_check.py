#!/usr/bin/env python3
"""44_strip_line_check.py -- diagnose straight vertical lines in one era's column of the film strips.
Note: tests straight-line (seam/tiling) artefacts; not needed for the gtb conifer case (see 45).

Takes the windows from the row labels that 40_film_digital_windows.py printed to its log
("row N. <sheet>  E<centre> N<centre> ..."), for one strip (--key, e.g. gtb_pan2000). For each window:
  1. Finds the line: the column with the largest left/right jump in the era's canopy map (same map
     the strip drew: the window's own sheet), and checks whether it coincides with
       - the 5 km sheet grid, and the 2.5 km / 1 km grids,
       - an edge of any canopy map or source image of the era near the window,
       - an inference patch edge of the sheet's frame (19: 1000 px patches, 800 px stride),
       - an edge of a Lantmäteriet source tile of the sheet (STAC footprints, from the tile list
         that 36 built the era from; skipped if the list or the network is unavailable).
  2. Reports on each side of the line (50 m strips): image mean / std / % zero / % outside the image,
     canopy values (1 canopy, 0 measured non-canopy, 255 = prob < 0 or outside the map) and
     probability stats where the _prob.tif exists.
  3. Writes results/strip_line_check/<key>_row<N>.png: image | probability | mask, line marked yellow.
     Probability: black 0 -> white 1, blue = prob < 0. Mask: white canopy, black non-canopy,
     grey = prob < 0. Dark red in both = outside the map.

    python scripts/44_strip_line_check.py --log results/logs/film_fill_4233698.out --key gtb_pan2000 --rows 3 6 7
gis env (GDAL Python bindings); the STAC check needs network access.
"""
import argparse, glob, importlib, os, re, sys
import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cf = importlib.import_module("39_common_footprint")     # warp (srcNodata="None"), sheet_box, src_dir, SHEET, INF
m32 = importlib.import_module("32_leafon_scan")         # item_polygon_sweref
m36 = importlib.import_module("36_fetch_build_era")     # STAC base URL, read_list, http_json

gdal.UseExceptions()
RES = 0.5
PATCH, STRIDE = 1000, 800          # 19_production_inference.py: PATCH 1000, OVERLAP 200
THRESHOLD = 0.6                    # 19's production threshold
TOL = 2.0                          # m; "coincides" if the line is this close
OUT_DIR = "results/strip_line_check"
ROW = re.compile(r"row (\d+)\. (\d{3}_\d{2}_\d{2})\s+E(\d+) N(\d+)\s*(.*)")
WROTE = re.compile(r"wrote results/film_windows_(\S+?)(_fill)?\.png")


def parse_log(path, key):
    """-> {row number: (sheet, E centre, N centre, rest of label)} for the strip written as <key>."""
    block, out = {}, None
    for ln in open(path, errors="replace"):
        m = ROW.search(ln)
        if m:
            block[int(m.group(1))] = (m.group(2), float(m.group(3)), float(m.group(4)), m.group(5).strip())
            continue
        w = WROTE.search(ln)
        if w:
            if w.group(1) == key:
                out = block
            block = {}
    if out is None:
        sys.exit(f"[error] no strip 'film_windows_{key}' with row labels in {path}")
    return out


def bounds(path):
    d = gdal.Open(path); g = d.GetGeoTransform()
    return g[0], g[3] + g[5] * d.RasterYSize, g[0] + g[1] * d.RasterXSize, g[3]


def covered(path, wb, k):
    """Bool (k, k): window cells inside the raster's extent."""
    x0, y0, x1, y1 = bounds(path)
    xs = wb[0] + (np.arange(k) + 0.5) * RES
    ys = wb[3] - (np.arange(k) + 0.5) * RES
    return np.outer((ys > y0) & (ys < y1), (xs > x0) & (xs < x1))


def positions(total, patch, stride):
    """Patch origins along one axis, as in 19_production_inference.py."""
    pos = list(range(0, max(total - patch, 0) + 1, stride))
    if pos[-1] != total - patch:
        pos.append(total - patch)
    return pos


def near_grid(e, step):
    return abs(e - round(e / step) * step)


def files_near(pattern, wb, pad=50.0):
    out = []
    for p in sorted(glob.glob(pattern)):
        x0, y0, x1, y1 = bounds(p)
        if x0 < wb[2] + pad and x1 > wb[0] - pad and y0 < wb[3] + pad and y1 > wb[1] - pad:
            out.append((p, x0, x1))
    return out


def stac_tiles(city, sheet, tiles_file):
    """-> [(tile, x0, x1, date)] for the source tiles 36 used for this sheet, or a reason string."""
    if not tiles_file or not os.path.exists(tiles_file):
        return "no tile list (pass --tiles results/leafon_scan/<list>.txt)"
    tl = m36.read_list(tiles_file).get((city, sheet))
    if not tl:
        return f"sheet not in {tiles_file}"
    out = []
    for coll, tile in tl:
        try:
            it = m36.http_json(f"{m36.STAC}/collections/{coll}/items/{tile}")
        except Exception as e:
            return f"STAC unavailable ({type(e).__name__})"
        xs = [p[0] for p in m32.item_polygon_sweref(it)]
        pr = it.get("properties", {})
        out.append((tile, min(xs), max(xs), pr.get("datetime") or pr.get("start_datetime") or pr.get("flygar")))
    return out


def find_line(can, w=40):
    """Column with the largest jump in canopy share between the w columns left and right of it."""
    p = can.mean(axis=0)
    c = np.cumsum(np.r_[0, p])
    best, bc = 0.0, None
    for x in range(w, len(p) - w):
        d = (c[x] - c[x - w]) / w - (c[x + w] - c[x]) / w
        if abs(d) > abs(best):
            best, bc = d, x
    return bc, best


def side_stats(sl, img, img_cov, can, can_cov, prob):
    n = img[:, sl].size
    iv = img[:, sl][img_cov[:, sl]]
    s = dict(img_mean=iv.mean() if iv.size else np.nan, img_std=iv.std() if iv.size else np.nan,
             img_zero=100 * (iv == 0).sum() / n, img_outside=100 * (~img_cov[:, sl]).sum() / n)
    cc, cv = can[:, sl], can_cov[:, sl]
    masked = (prob[:, sl] < 0) & cv if prob is not None else np.zeros_like(cv)
    s.update(can_1=100 * ((cc > 0) & cv).sum() / n, can_0=100 * ((cc == 0) & cv & ~masked).sum() / n,
             can_255_masked=100 * masked.sum() / n, can_255_outside=100 * (~cv).sum() / n)
    if prob is not None:
        pv = prob[:, sl][cv & (prob[:, sl] >= 0)]
        s.update(prob_mean=pv.mean() if pv.size else np.nan, prob_median=np.median(pv) if pv.size else np.nan,
                 prob_ge_thr=100 * (pv >= THRESHOLD).sum() / max(pv.size, 1))
    return s


def save_png(path, img, img_cov, prob, can, can_cov, line):
    k, gap = img.shape[0], 6
    rgb = np.full((3, k, 3 * k + 2 * gap), 255, np.uint8)
    v = img[img_cov]
    lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 1)
    g8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    g8[~img_cov] = 0
    panels = [np.stack([g8] * 3)]
    if prob is not None:
        p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        pp = np.stack([p8] * 3)
        pp[:, prob < 0] = np.array([0, 0, 160])[:, None]
    else:
        pp = np.full((3, k, k), 60, np.uint8)
    pp[:, ~can_cov] = np.array([120, 0, 0])[:, None]
    panels.append(pp)
    m = np.zeros((k, k), np.uint8)
    m[can > 0] = 255
    if prob is not None:
        m[(prob < 0) & (can == 0)] = 128
    mm = np.stack([m] * 3)
    mm[:, ~can_cov] = np.array([120, 0, 0])[:, None]
    panels.append(mm)
    for i, p in enumerate(panels):
        x = i * (k + gap)
        rgb[:, :, x:x + k] = p
        if line is not None:
            rgb[:, ::8, x + line] = np.array([255, 220, 0])[:, None]      # dashed yellow line
            rgb[:, 1::8, x + line] = np.array([255, 220, 0])[:, None]
    mem = gdal.GetDriverByName("MEM").Create("", rgb.shape[2], k, 3, gdal.GDT_Byte)
    for b in range(3):
        mem.GetRasterBand(b + 1).WriteArray(rgb[b])
    gdal.GetDriverByName("PNG").CreateCopy(path, mem)


def main(a):
    city = a.key.split("_")[0]
    rows = parse_log(a.log, a.key)
    tiles_file = a.tiles or next(iter(sorted(glob.glob(f"results/leafon_scan/final_era{a.era}*.txt"))), None)
    os.makedirs(OUT_DIR, exist_ok=True)
    k = int(round(a.win / RES))
    inf_dir = os.path.join(cf.INF, f"{city}_{a.era}")
    img_dir = cf.src_dir(city, a.era)
    print(f"strip {a.key}, era {a.era}: canopy {inf_dir}/, imagery {img_dir}/, tile list {tiles_file}")

    for r in a.rows:
        if r not in rows:
            print(f"\n== row {r}: not in the log"); continue
        sh, ce, cn, rest = rows[r]
        wb = (ce - a.win / 2, cn - a.win / 2, ce + a.win / 2, cn + a.win / 2)
        print(f"\n== row {r}: {sh}  centre E{ce:.0f} N{cn:.0f}  ({rest})")
        print(f"   window E{wb[0]:.0f}-{wb[2]:.0f} N{wb[1]:.0f}-{wb[3]:.0f}, sheet box {cf.sheet_box(sh)}")
        can_p = next(iter(glob.glob(os.path.join(inf_dir, f"{sh}_*_canopy.tif"))), None)
        img_p = next(iter(glob.glob(os.path.join(img_dir, f"{sh}_*.tif"))), None)
        if not can_p:
            print(f"   no {a.era} canopy map for {sh}"); continue
        prob_p = can_p.replace("_canopy.tif", "_prob.tif")
        prob_p = prob_p if os.path.exists(prob_p) else None

        can = cf.warp(can_p, wb, RES, "near")[:k, :k]
        can_cov = covered(can_p, wb, k)
        prob = cf.warp(prob_p, wb, RES, "near", gdal.GDT_Float32)[:k, :k] if prob_p else None
        if prob is not None:
            prob[~can_cov] = np.nan
        if img_p:
            img = cf.warp(img_p, wb, RES, "near", gdal.GDT_Float32)[:k, :k]
            img_cov = covered(img_p, wb, k)
        else:
            print(f"   no {a.era} image for {sh}")
            img, img_cov = np.zeros((k, k), np.float32), np.zeros((k, k), bool)

        c, jump = find_line((can > 0) & can_cov)
        if c is None:
            print("   no line found"); continue
        e = wb[0] + c * RES
        print(f"   line at column {c} -> E{e:.1f}  (canopy share left - right {100 * jump:+.0f} pp)")

        # (1) coincidences
        checks = [(f"{s / 1000:g} km grid", near_grid(e, s)) for s in (5000, 2500, 1000)]
        for label, pat in (("canopy map", os.path.join(inf_dir, "*_canopy.tif")),
                           ("source image", os.path.join(img_dir, "*.tif"))):
            for p, x0, x1 in files_near(pat, wb):
                checks.append((f"{label} edge {os.path.basename(p)} (E{x0:.0f}-{x1:.0f})", min(abs(e - x0), abs(e - x1))))
        fx0, _, fx1, _ = bounds(can_p)
        nx = int(round((fx1 - fx0) / RES))
        pe = sorted({fx0 + x * RES for x in positions(nx, PATCH, STRIDE)} |
                    {fx0 + (x + PATCH) * RES for x in positions(nx, PATCH, STRIDE)})
        checks.append((f"inference patch edge (nearest E{min(pe, key=lambda v: abs(v - e)):.0f})",
                       min(abs(v - e) for v in pe)))
        st = stac_tiles(city, sh, tiles_file)
        if isinstance(st, str):
            print(f"   source tiles: {st}")
        else:
            for tile, x0, x1, dt in st:
                checks.append((f"source tile {tile} edge (E{x0:.0f}-{x1:.0f}, {dt})", min(abs(e - x0), abs(e - x1))))
        for label, d in sorted(checks, key=lambda t: t[1]):
            print(f"   {'COINCIDES' if d <= TOL else '         '} {d:8.1f} m  {label}")

        # (2) both sides of the line, 50 m strips, 1 m gap at the line
        sw, g = int(round(50 / RES)), int(round(1 / RES))
        sides = (("left", slice(max(c - sw, 0), max(c - g, 0))), ("right", slice(min(c + g, k), min(c + sw, k))))
        for name, sl in sides:
            s = side_stats(sl, img, img_cov, can, can_cov, prob)
            line = (f"   {name:5s} image mean {s['img_mean']:6.1f} std {s['img_std']:5.1f} zero {s['img_zero']:5.1f}% "
                    f"outside {s['img_outside']:5.1f}% | canopy 1 {s['can_1']:5.1f}% 0 {s['can_0']:5.1f}% "
                    f"255 {s['can_255_masked'] + s['can_255_outside']:5.1f}% (prob<0 {s['can_255_masked']:.1f}, "
                    f"outside map {s['can_255_outside']:.1f})")
            if prob is not None:
                line += (f" | prob mean {s['prob_mean']:.3f} median {s['prob_median']:.3f} "
                         f">={THRESHOLD} {s['prob_ge_thr']:5.1f}%")
            print(line)

        # (3) PNG
        out = os.path.join(OUT_DIR, f"{a.key}_row{r}.png")
        if os.path.exists(out) and not a.overwrite:
            print(f"   {out} exists, not replaced (--overwrite)")
        else:
            save_png(out, img, img_cov, prob, can, can_cov, c)
            print(f"   wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="log of the 40 run with the row labels")
    ap.add_argument("--key", default="gtb_pan2000", help="strip key <city>_<film>, as in film_windows_<key>_fill.png")
    ap.add_argument("--rows", type=int, nargs="+", default=[3, 6, 7])
    ap.add_argument("--era", default="2010", help="era of the column with the line")
    ap.add_argument("--win", type=float, default=250.0, help="window size in m (40's --win)")
    ap.add_argument("--tiles", default=None, help="36 tile list for the era (default results/leafon_scan/final_era<era>*.txt)")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
