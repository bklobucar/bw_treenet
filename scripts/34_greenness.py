#!/usr/bin/env python3
"""34_greenness.py -- put a number on "does this flight look leaf-on?" using the catalogue thumbnails.

For every thumbnail folder made by 33_fetch_thumbs.py it computes the green chromatic coordinate
GCC = G / (R + G + B) (the standard phenocam greenness index) over land pixels, per tile, and reports
the median over tiles. Flights with KNOWN dates (April = leaf-off, June = leaf-on) calibrate the scale,
so year-only flights (2004, 2005, 2008, 2009) can be placed on it.

Land pixels: not white nodata, not dark water (brightness 35..235). Tiles with < 30% land are skipped.
Caveat: different cameras and colour balancing shift GCC a little, so compare within a city and read
the result as "closer to the April or the June references", not as a date.

Run on HomePC (needs numpy + Pillow, both in the bwtreenet env):
    python ~/bw_treenet/scripts/34_greenness.py ~/thumbs_check
"""
import glob, os, re, sys
import numpy as np
from PIL import Image

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/thumbs_check")
rows = []
for d in sorted(glob.glob(os.path.join(root, "*/"))):
    name = os.path.basename(d.rstrip("/"))
    vals = []
    for f in glob.glob(os.path.join(d, "*.jpg")):
        try:
            a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        except Exception:
            continue
        s = a.sum(-1)
        bright = s / 3
        land = (bright > 35) & (bright < 235)
        if land.mean() < 0.30:
            continue
        gcc = a[..., 1][land] / s[land]
        vals.append((np.median(gcc), np.percentile(gcc, 90)))
    if not vals:
        continue
    v = np.array(vals)
    city = name.split("_")[0]
    year = re.search(r"_(\d{4})$", name)
    rows.append((city, int(year.group(1)) if year else 0, name, len(v), np.median(v[:, 0]), np.median(v[:, 1]),
                 np.percentile(v[:, 0], 25), np.percentile(v[:, 0], 75)))

print(f"{'city':6s} {'year':4s} {'flight folder':44s} {'tiles':>5s} {'GCC med':>8s} {'GCC p90':>8s}  IQR of tile medians")
for city in sorted({r[0] for r in rows}):
    for r in sorted((r for r in rows if r[0] == city), key=lambda r: r[1]):
        print(f"{r[0]:6s} {r[1]:4d} {r[2]:44s} {r[3]:5d} {r[4]:8.4f} {r[5]:8.4f}  {r[6]:.4f}-{r[7]:.4f}")
    print()
