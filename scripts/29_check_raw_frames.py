#!/usr/bin/env python
"""29_check_raw_frames.py -- decode every block of every raw orthophoto sheet and report
the ones that fail (LZW 'code not yet in table', truncated, zero-byte, unreadable).

Run in the bwtreenet env on a compute node, from the repo root:
    python scripts/29_check_raw_frames.py
    python scripts/29_check_raw_frames.py --root /path/to/repo --procs 8
Writes results/bad_frames.txt (one path per line) and exits 1 if anything is bad.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import rasterio


def chk(path):
    try:
        if os.path.getsize(path) == 0:
            return path, "zero bytes"
        with rasterio.open(path) as s:
            for _, w in s.block_windows(1):
                s.read(1, window=w)
        return path, None
    except Exception as e:
        return path, str(e).replace("\n", " ")[:100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    pat = os.path.join(a.root, "data", "raw", "*", "*OF_*", "*.tif")
    files = sorted(f for f in glob.glob(pat) if not f.endswith(".ovr"))
    print(f"checking {len(files)} sheets with {a.procs} processes ...", flush=True)
    with Pool(a.procs) as p:
        res = p.map(chk, files, chunksize=1)
    bad = [(f, e) for f, e in res if e]
    for f, e in bad:
        print("BAD", os.path.relpath(f, a.root), "--", e)
    os.makedirs(os.path.join(a.root, "results"), exist_ok=True)
    with open(os.path.join(a.root, "results", "bad_frames.txt"), "w") as fh:
        fh.write("\n".join(os.path.relpath(f, a.root) for f, _ in bad) + ("\n" if bad else ""))
    print(f"{len(files)} sheets, {len(bad)} bad")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
