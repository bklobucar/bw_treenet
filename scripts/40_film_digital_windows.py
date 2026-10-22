#!/usr/bin/env python3
"""40_film_digital_windows.py -- where does the film -> digital canopy step land?

For one city, compares a film era with a digital era on production's canopy maps:
  1. City totals: gross gain / loss, and gained pixels split by distance from film-era canopy
     (<=1 m = wider crown outlines, 1-5 m = near existing crowns, >5 m = whole new patches).
  2. The step broken down by how wooded each window is in an independent era (default 2010).
  3. A PNG strip of the windows with the largest step: imagery for each era with the
     predicted canopy outlined in red, to judge leaf state and what the model calls canopy.
     --fill draws canopy as a semi-transparent green fill instead and adds a last column with the
     film-era mask alone (canopy white, non-canopy black, outside the valid area grey); output
     results/film_windows_<city>_<film>_fill.png. Rows and columns are labelled in the PNG.
     --top N renders strictly the N largest steps, sorted by step (no per-sheet cap).
     --strips-only skips the statistics (gain/loss split, woodedness table) and the bin era.

Run from the project root in the `gis` env (GDAL + numpy only).
  python scripts/40_film_digital_windows.py --city gtb --film pan2000 --digital 2005 \
      --show 1990:1990r05 pan2000 2005 2010
  python scripts/40_film_digital_windows.py --city gtb --film pan2000 --digital 2005 \
      --show 1990:1990r05 pan2000 2005 2010 --fill --top 12 --strips-only
An existing output PNG is not replaced unless --overwrite is given.
Columns in --show are IMAGE_ERA[:CANOPY_ERA]; the canopy era defaults to the image era.
"""
import argparse
import glob
import os
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

ap = argparse.ArgumentParser()
ap.add_argument("--city", default="gtb")
ap.add_argument("--film", default="pan2000")
ap.add_argument("--digital", default="2005")
ap.add_argument("--bin-era", default="2010", help="independent era used to class windows by how wooded they are")
ap.add_argument("--show", nargs="+", default=["1990:1990r05", "pan2000", "2005", "2010"])
ap.add_argument("--win", type=float, default=250.0, help="window size in m")
ap.add_argument("--n", type=int, default=8, help="windows to render")
ap.add_argument("--per-sheet", type=int, default=2, help="max rendered windows per sheet")
ap.add_argument("--res", type=float, default=0.5)
ap.add_argument("--fill", action="store_true", help="green filled canopy + film-mask column; writes *_fill.png")
ap.add_argument("--alpha", type=float, default=0.4, help="opacity of the --fill overlay")
ap.add_argument("--top", type=int, default=0, help="render only the N largest steps, sorted by step")
ap.add_argument("--strips-only", action="store_true", help="skip the statistics; only render the strip")
ap.add_argument("--overwrite", action="store_true", help="replace an existing output PNG")
a = ap.parse_args()
OUT = f"results/film_windows_{a.city}_{a.film}{'_fill' if a.fill else ''}.png"
if os.path.exists(OUT) and not a.overwrite:
    sys.exit(f"{OUT} exists; pass --overwrite to replace it")


def sheet_of(path):
    return "_".join(os.path.basename(path).split("_")[:3])


def maps(era, kind="canopy"):
    return {sheet_of(p): p for p in glob.glob(f"results/inference_v19/{a.city}_{era}/*_{kind}.tif")}


def image(era, sheet):
    for pat in (f"data/raw/{a.city}/{era}_OF_gray_*/{sheet}_*.tif", f"data/raw/{a.city}/{era}_*/{sheet}_*.tif"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def bounds(path):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    return (gt[0], gt[3] + gt[5] * ds.RasterYSize, gt[0] + gt[1] * ds.RasterXSize, gt[3])


def warp(path, box, alg="near", ftype=gdal.GDT_Byte):
    # srcNodata=None: canopy maps carry NoData=0, which must stay 0 (not-canopy), not be dropped
    arr = gdal.Warp("", path, format="MEM", outputBounds=box, xRes=a.res, yRes=a.res,
                    resampleAlg=alg, srcNodata="None", outputType=ftype).ReadAsArray()
    return arr[0] if arr.ndim == 3 else arr


def dilate(m, r):
    """Square dilation by r pixels (separable, no scipy needed)."""
    out = m.copy()
    for s in range(1, r + 1):
        out[s:, :] |= m[:-s, :]
        out[:-s, :] |= m[s:, :]
    m2 = out.copy()
    for s in range(1, r + 1):
        out[:, s:] |= m2[:, :-s]
        out[:, :-s] |= m2[:, s:]
    return out


def outline(c):
    inner = c.copy()
    inner[1:, :] &= c[:-1, :]
    inner[:-1, :] &= c[1:, :]
    inner[:, 1:] &= c[:, :-1]
    inner[:, :-1] &= c[:, 1:]
    return c & ~inner


# 5x7 bitmap font for the PNG labels (GDAL + numpy only); unknown characters render as blanks
FONT = {ch: [int(b, 2) for b in rows.split(",")] for ch, rows in (ln.split(" ", 1) for ln in """\
0 01110,10001,10011,10101,11001,10001,01110
1 00100,01100,00100,00100,00100,00100,01110
2 01110,10001,00001,00010,00100,01000,11111
3 11110,00001,00001,01110,00001,00001,11110
4 00010,00110,01010,10010,11111,00010,00010
5 11111,10000,11110,00001,00001,10001,01110
6 00110,01000,10000,11110,10001,10001,01110
7 11111,00001,00010,00100,01000,01000,01000
8 01110,10001,10001,01110,10001,10001,01110
9 01110,10001,10001,01111,00001,00010,01100
A 01110,10001,10001,11111,10001,10001,10001
B 11110,10001,10001,11110,10001,10001,11110
C 01110,10001,10000,10000,10000,10001,01110
D 11110,10001,10001,10001,10001,10001,11110
E 11111,10000,10000,11110,10000,10000,11111
F 11111,10000,10000,11110,10000,10000,10000
G 01110,10001,10000,10111,10001,10001,01111
H 10001,10001,10001,11111,10001,10001,10001
I 01110,00100,00100,00100,00100,00100,01110
J 00111,00010,00010,00010,00010,10010,01100
K 10001,10010,10100,11000,10100,10010,10001
L 10000,10000,10000,10000,10000,10000,11111
M 10001,11011,10101,10101,10001,10001,10001
N 10001,10001,11001,10101,10011,10001,10001
O 01110,10001,10001,10001,10001,10001,01110
P 11110,10001,10001,11110,10000,10000,10000
Q 01110,10001,10001,10001,10101,10010,01101
R 11110,10001,10001,11110,10100,10010,10001
S 01111,10000,10000,01110,00001,00001,11110
T 11111,00100,00100,00100,00100,00100,00100
U 10001,10001,10001,10001,10001,10001,01110
V 10001,10001,10001,10001,10001,01010,00100
W 10001,10001,10001,10101,10101,10101,01010
X 10001,10001,01010,00100,01010,10001,10001
Y 10001,10001,01010,00100,00100,00100,00100
Z 11111,00001,00010,00100,01000,10000,11111
% 11000,11001,00010,00100,01000,10011,00011
_ 00000,00000,00000,00000,00000,00000,11111
. 00000,00000,00000,00000,00000,01100,01100
: 00000,01100,01100,00000,01100,01100,00000
- 00000,00000,00000,11111,00000,00000,00000
+ 00000,00100,00100,11111,00100,00100,00000
/ 00001,00001,00010,00100,01000,10000,10000
| 00100,00100,00100,00100,00100,00100,00100""".splitlines())}
SCALE = 2                                   # glyphs drawn at 10 x 14 px
LABEL_H = 7 * SCALE + 8                     # height of a label band


def text(rgb, s, y, x, color=(0, 0, 0)):
    """Draw s (upper-cased) into rgb (3, H, W) with its top-left corner at (y, x), clipped at the edge."""
    W = rgb.shape[2]
    for ch in s.upper():
        rows = FONT.get(ch)
        if rows:
            for ry, bits in enumerate(rows):
                for rx in range(5):
                    if bits >> (4 - rx) & 1:
                        y0, x0 = y + ry * SCALE, x + rx * SCALE
                        if x0 + SCALE <= W:
                            for bnd in range(3):
                                rgb[bnd, y0:y0 + SCALE, x0:x0 + SCALE] = color[bnd]
        x += 6 * SCALE


F, D, B = maps(a.film), maps(a.digital), maps(a.bin_era)
Fp, Dp = maps(a.film, "prob"), maps(a.digital, "prob")
for era, m in ((a.film, F), (a.digital, D)) + (() if a.strips_only else ((a.bin_era, B),)):
    if not m:
        have = sorted(os.path.basename(p) for p in glob.glob(f"results/inference_v19/{a.city}_*"))
        sys.exit(f"no canopy maps for {a.city} {era}; available: {have}")
sheets = sorted(set(F) & set(D) & set(Fp) & set(Dp) & (set(F) if a.strips_only else set(B)))
if not sheets:
    sys.exit("no sheet has film, digital and bin-era maps")

k = int(round(a.win / a.res))
r1, r5 = int(round(1.0 / a.res)), int(round(5.0 / a.res))
tot = dict(valid=0, film=0, dig=0, gain=0, loss=0, e1=0, e5=0, far=0)
recs = []  # (sheet, i, j, box, n_valid, n_film, n_dig, n_bin)

for sh in sheets:
    box = bounds(F[sh])
    valid = (warp(Fp[sh], box, ftype=gdal.GDT_Float32) >= 0) & (warp(Dp[sh], box, ftype=gdal.GDT_Float32) >= 0)
    f = (warp(F[sh], box) > 0) & valid
    d = (warp(D[sh], box) > 0) & valid
    b = None if a.strips_only else (warp(B[sh], box) > 0) & valid

    if not a.strips_only:
        gain, loss = d & ~f, f & ~d
        near1 = dilate(f, r1)
        near5 = dilate(f, r5)
        tot["valid"] += int(valid.sum()); tot["film"] += int(f.sum()); tot["dig"] += int(d.sum())
        tot["gain"] += int(gain.sum()); tot["loss"] += int(loss.sum())
        tot["e1"] += int((gain & near1).sum())
        tot["e5"] += int((gain & near5 & ~near1).sum())
        tot["far"] += int((gain & ~near5).sum())

    ny, nx = (valid.shape[0] // k) * k, (valid.shape[1] // k) * k

    def blk(x):
        return x[:ny, :nx].reshape(ny // k, k, nx // k, k).sum(axis=(1, 3)).astype(np.float64)

    nv, nf, nd = blk(valid), blk(f), blk(d)
    nb = np.zeros_like(nv) if b is None else blk(b)
    for i in range(ny // k):
        for j in range(nx // k):
            if nv[i, j] >= 0.9 * k * k:
                recs.append((sh, i, j, box, nv[i, j], nf[i, j], nd[i, j], nb[i, j]))
    print(f"  {sh}: done", flush=True)

R = np.array([r[4:] for r in recs])
nv, nf, nd, nb = R.T
cov_f, cov_d, cov_b = nf / nv, nd / nv, nb / nv

if not a.strips_only:
    px_km2 = a.res ** 2 / 1e6
    V = tot["valid"]
    print(f"\n== {a.city}: {a.film} (film) vs {a.digital} (digital), {len(sheets)} sheets, {V * px_km2:.1f} km2 scored")
    print(f"cover          film {100 * tot['film'] / V:5.2f} %   digital {100 * tot['dig'] / V:5.2f} %   "
          f"step {100 * (tot['dig'] - tot['film']) / V:+.2f} pp")
    print(f"gross change   gain {100 * tot['gain'] / V:5.2f} pp   loss {100 * tot['loss'] / V:5.2f} pp")
    g = max(1, tot["gain"])
    print("gained pixels by distance from film-era canopy:")
    print(f"  <=1 m (wider outlines)       {100 * tot['e1'] / g:5.1f} % of gain   ({100 * tot['e1'] / V:.2f} pp)")
    print(f"  1-5 m (near existing crowns) {100 * tot['e5'] / g:5.1f} % of gain   ({100 * tot['e5'] / V:.2f} pp)")
    print(f"  >5 m  (whole new patches)    {100 * tot['far'] / g:5.1f} % of gain   ({100 * tot['far'] / V:.2f} pp)")

    net = (nd - nf).sum()
    print(f"\nstep by how wooded the {a.win:.0f} m window is in {a.bin_era} ({len(recs)} windows):")
    print(f"  {a.bin_era + ' cover':14s} {'area':>6s} {'film':>7s} {'digital':>8s} {'step':>7s} {'share of net step':>18s}")
    for lo, hi in ((0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        m = (cov_b >= lo) & (cov_b < hi)
        if not m.any():
            continue
        area = nv[m].sum()
        print(f"  {int(100 * lo):3d}-{min(100, int(100 * hi)):3d} %      {100 * area / nv.sum():5.1f}% "
              f"{100 * nf[m].sum() / area:6.2f}% {100 * nd[m].sum() / area:7.2f}% "
              f"{100 * (nd[m].sum() - nf[m].sum()) / area:+6.2f}  {100 * (nd[m].sum() - nf[m].sum()) / net:16.1f}%")

# ---- strip of the windows with the largest step
order = np.argsort(-(cov_d - cov_f), kind="stable")
picked, per = [], {}
if a.top:
    picked = list(order[:a.top])
else:
    for idx in order:
        sh = recs[idx][0]
        if per.get(sh, 0) >= a.per_sheet:
            continue
        picked.append(idx)
        per[sh] = per.get(sh, 0) + 1
        if len(picked) == a.n:
            break

cols = [(s.split(":") + [s])[:2] if ":" in s else [s, s] for s in a.show]
heads = [f"image {ie} / canopy {ce}" if ie != ce else f"{ie}" for ie, ce in cols]
if a.fill:
    heads.append(f"{a.film} mask: white canopy / grey outside")
gap = 6
ncol = len(heads)
H = LABEL_H + len(picked) * (LABEL_H + k) + (len(picked) - 1) * gap
W = ncol * k + (ncol - 1) * gap
rgb = np.full((3, H, W), 255, np.uint8)
for c, h in enumerate(heads):
    text(rgb, h, 4, c * (k + gap) + 2)
print(f"\nstrip: rows = windows with the largest step; columns = " + " | ".join(heads))
GREEN = np.array([40, 220, 40], np.float32).reshape(3, 1)
for r, idx in enumerate(picked):
    sh, i, j, box = recs[idx][:4]
    wb = (box[0] + j * a.win, box[3] - (i + 1) * a.win, box[0] + (j + 1) * a.win, box[3] - i * a.win)
    cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
    step = 100 * (cov_d[idx] - cov_f[idx])
    label = (f"{r + 1}. {sh}  E{cx:.0f} N{cy:.0f}  {a.film} {100 * cov_f[idx]:.0f}%  "
             f"{a.digital} {100 * cov_d[idx]:.0f}%  step {step:+.0f} pp")
    print(f"  row {label}" + ("" if a.strips_only else f"  {a.bin_era} {100 * cov_b[idx]:.0f}%"))
    y = LABEL_H + r * (LABEL_H + k + gap)
    text(rgb, label, y + 4, 2)
    y += LABEL_H
    for c, (ie, ce) in enumerate(cols):
        x = c * (k + gap)
        tile = np.full((3, k, k), 60, np.uint8)
        ip = image(ie, sh)
        if ip:
            im = warp(ip, wb, "bilinear", gdal.GDT_Float32)
            lo, hi = np.percentile(im, (2, 98))
            g8 = np.clip((im - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            tile[:] = g8[:k, :k]
        else:
            print(f"    no imagery for {ie} on {sh}")
        cp = maps(ce).get(sh)
        if cp:
            cm = (warp(cp, wb) > 0)[:k, :k]
            if a.fill:
                t = tile[:, cm].astype(np.float32)
                tile[:, cm] = np.round((1 - a.alpha) * t + a.alpha * GREEN).astype(np.uint8)
            else:
                ol = outline(cm)
                tile[0][ol], tile[1][ol], tile[2][ol] = 255, 40, 40
        rgb[:, y:y + k, x:x + k] = tile
    if a.fill:
        x = len(cols) * (k + gap)
        v = (warp(Fp[sh], wb, ftype=gdal.GDT_Float32) >= 0)[:k, :k]
        cm = (warp(F[sh], wb) > 0)[:k, :k]
        mask = np.full((k, k), 128, np.uint8)
        mask[v] = 0
        mask[v & cm] = 255
        rgb[:, y:y + k, x:x + k] = mask

mem = gdal.GetDriverByName("MEM").Create("", W, H, 3, gdal.GDT_Byte)
for bnd in range(3):
    mem.GetRasterBand(bnd + 1).WriteArray(rgb[bnd])
gdal.GetDriverByName("PNG").CreateCopy(OUT, mem)
print(f"wrote {OUT}")
