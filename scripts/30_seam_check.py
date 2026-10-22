#!/usr/bin/env python
"""30_seam_check.py -- check that rebuilt frames line up with their neighbouring sheets.

For every rebuilt frame in data/rebuilt_frames/ and every side that has a neighbouring sheet,
the script takes the outermost pixel lines on both sides of the seam, removes the slow trend,
and cross-correlates them along the seam.  A correctly georeferenced frame peaks at a shift of
0 px; a systematic datum / reprojection error shows up as a shift of 1-3 px (one pixel is 0.5 m
or 1 m).  East-west seams measure the north-south error and north-south seams the east-west
error.  Neighbours may come from a different flight, so the correlation value is reported too:
trust the shift only where the peak correlation is clearly above the zero-shift correlation
and above ~0.3.

    python scripts/30_seam_check.py            # run from the repo root, bwtreenet env
    python scripts/30_seam_check.py --root DIR

A neighbour is looked up first among the rebuilt frames, then among the sheets in data/raw.
"""
import argparse
import glob
import os
import re

import numpy as np
import rasterio
from rasterio.windows import Window

FRAMES_RE = re.compile(r"^(\d{3})_(\d{2})_(\d{2})_(\d{4})\.tif$")
SHEET = 5000
LINES = 2          # outermost pixel lines averaged on each side of the seam (--lines)
MAXSHIFT = 8       # pixels
TREND = 51         # moving-average window removed from the signal (pixels)


def code_from_xy(x0, y0):
    return f"{int(y0) // 10000:03d}_{int(x0) // 10000:02d}_{(int(y0) % 10000) // 1000}{(int(x0) % 10000) // 1000}"


def bbox_from_code(code):
    n, e, ab = code.split("_")
    return int(e) * 10000 + int(ab[1]) * 1000, int(n) * 10000 + int(ab[0]) * 1000


def smooth(v, k):
    ker = np.ones(k) / k
    return np.convolve(np.pad(v, k // 2, mode="edge"), ker, mode="valid")


def edge_signal(path, side, LINES=LINES):
    """Mean of the outermost LINES pixel lines on `side` of a sheet, as a 1-D array along the edge."""
    with rasterio.open(path) as s:
        h, w = s.height, s.width
        if side == "E":
            a = s.read(1, window=Window(w - LINES, 0, LINES, h)).astype(float).mean(axis=1)
        elif side == "W":
            a = s.read(1, window=Window(0, 0, LINES, h)).astype(float).mean(axis=1)
        elif side == "N":
            a = s.read(1, window=Window(0, 0, w, LINES)).astype(float).mean(axis=0)
        else:
            a = s.read(1, window=Window(0, h - LINES, w, LINES)).astype(float).mean(axis=0)
    return a


def best_shift(a, b):
    a = a - smooth(a, TREND)
    b = b - smooth(b, TREND)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best, bestc, c0 = 0, -2.0, 0.0
    for s in range(-MAXSHIFT, MAXSHIFT + 1):
        x = a[max(0, s): n + min(0, s)]
        y = b[max(0, -s): n - max(0, s)]
        if x.std() == 0 or y.std() == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if s == 0:
            c0 = c
        if c > bestc:
            best, bestc = s, c
    return best, bestc, c0


def find_neighbour(stage_files, raw_files, folder, code):
    """Neighbouring sheet of the SAME era folder: rebuilt version first, then the raw sheet."""
    for pool in (stage_files, raw_files):
        for p in pool.get((folder, code), []):
            if os.path.getsize(p) > 0:
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--lines", type=int, default=LINES, help="pixel lines averaged on each side of a seam")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    stage = sorted(glob.glob(os.path.join(root, "data", "rebuilt_frames", "*", "*.tif")))
    if not stage:
        raise SystemExit("[error] no rebuilt frames; run 28_rebuild_frames.py --build first.")
    raw = {}
    for p in glob.glob(os.path.join(root, "data", "raw", "*", "*OF_*", "*.tif")):
        m = FRAMES_RE.match(os.path.basename(p))
        if m:
            raw.setdefault((os.path.basename(os.path.dirname(p)), "_".join(m.groups()[:3])), []).append(p)
    stg = {}
    for p in stage:
        m = FRAMES_RE.match(os.path.basename(p))
        stg.setdefault((os.path.basename(os.path.dirname(p)), "_".join(m.groups()[:3])), []).append(p)

    print(f"{'frame':18s} {'side':4s} {'neighbour':22s} {'shift px':>8s} {'peak r':>7s} {'r at 0':>7s}  verdict")
    for p in stage:
        name = os.path.basename(p)
        code = "_".join(FRAMES_RE.match(name).groups()[:3])
        folder = os.path.basename(os.path.dirname(p))
        x0, y0 = bbox_from_code(code)
        sides = {"E": (x0 + SHEET, y0), "W": (x0 - SHEET, y0), "N": (x0, y0 + SHEET), "S": (x0, y0 - SHEET)}
        opposite = {"E": "W", "W": "E", "N": "S", "S": "N"}
        for side, (nx, ny) in sides.items():
            ncode = code_from_xy(nx, ny)
            # a neighbouring sheet from the rebuilt set must be a different frame, not this one
            pool_stage = {k: v for k, v in stg.items() if k != (folder, code)}
            q = find_neighbour(pool_stage, raw, folder, ncode)
            if not q:
                continue
            try:
                s1, s2 = edge_signal(p, side, a.lines), edge_signal(q, opposite[side], a.lines)
                if len(s1) != len(s2):
                    print(f"{name:18s} {side:4s} {os.path.basename(q):22s}  (different pixel size, skipped)")
                    continue
                sh, pk, c0 = best_shift(s1, s2)
            except Exception as e:
                print(f"{name:18s} {side:4s} {os.path.basename(q):22s}  error: {str(e)[:60]}")
                continue
            if pk < 0.3:
                verdict = "inconclusive (low correlation)"
            elif abs(sh) <= 1:
                verdict = "OK"
            else:
                verdict = "SHIFTED -- check the reprojection"
            print(f"{name:18s} {side:4s} {os.path.basename(q):22s} {sh:8d} {pk:7.2f} {c0:7.2f}  {verdict}")


if __name__ == "__main__":
    main()
