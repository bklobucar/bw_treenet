#!/usr/bin/env python
"""28_rebuild_frames.py -- rebuild damaged / missing 5 x 5 km source frames from
Lantmateriet catalogue tiles that were downloaded by hand (see fetch_tiles.sh).

Why: seven frames in data/raw are unusable (six fail a full read, one Malmo 1998
sheet was never downloaded).  The catalogue tiles are 5 x 5 km in RT90 2.5 gon V
(EPSG:3021); the project frames are 5 x 5 km sheets in SWEREF 99 TM (EPSG:3006).
This script reprojects the tiles onto the exact sheet grid.

No network access and no credentials are used here.

Usage (inside the bwtreenet env, from the repo root):
    python scripts/28_rebuild_frames.py --plan      # check names, tiles, estimated coverage (fast)
    python scripts/28_rebuild_frames.py --build     # warp at full resolution into data/rebuilt_frames/
    python scripts/28_rebuild_frames.py --install   # move bad files to data/raw_corrupt/, copy new ones in
Options: --tiles DIR (default data/stac_tiles), --only CODE [CODE ...], --force, --root DIR
"""
import argparse
import csv
import glob
import os
import re
import shutil
import sys
import time

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

DST_CRS = CRS.from_epsg(3006)
MAX_NODATA = 0.05          # refuse to install a frame with more nodata than this (unless --force)
SHEET = 5000.0             # sheet size in metres

# (folder, sheet code, flight year in the file name).  A frame is rebuilt from tiles of
# its own flight year first; remaining empty pixels are filled from the nearest other year.
FRAMES = [
    ("data/raw/gtb/1960_OF_gray_gtb",  "639_32_05", 1963),
    ("data/raw/gtb/1960_OF_gray_gtb",  "639_32_50", 1960),
    ("data/raw/sth/1960_OF_gray_sth",  "657_68_00", 1958),
    ("data/raw/sth/1960_OF_gray_sth",  "657_68_05", 1958),
    ("data/raw/sth/1970_OF_gray_sth",  "658_66_55", 1971),
    ("data/raw/sth/1970_OF_gray_sth",  "658_67_00", 1971),
    ("data/raw/malmo/1990_OF_gray_mmo", "616_37_50", 1998),
]

SHEET_RE = re.compile(r"^(\d{3})_(\d{2})_(\d{2})_(\d{4})\.tif$")


# ── geometry helpers ────────────────────────────────────────────────────────
def frame_bbox(code):
    """'639_32_05' -> (E min, N min, E max, N max) in EPSG:3006.
    NNN = northing in 10 km, EE = easting in 10 km, then two digits (a, b) = northing / easting
    offset in km inside that 10 km square (0 or 5)."""
    n, e, ab = code.split("_")
    a, b = int(ab[0]), int(ab[1])
    x0 = int(e) * 10000 + b * 1000
    y0 = int(n) * 10000 + a * 1000
    return (float(x0), float(y0), float(x0 + SHEET), float(y0 + SHEET))


def read_tfw(path):
    """World-file -> (pixel size, x of top-left corner, y of top-left corner) or None."""
    try:
        v = [float(s) for s in open(path).read().split()[:6]]
        res = v[0]
        return res, v[4] - res / 2.0, v[5] + res / 2.0
    except Exception:
        return None


def year_of(name):
    m = re.search(r"_c(\d\d)(?:\.tif)?$", name)
    return 1900 + int(m.group(1)) if m else None


# ── stage 1: inspection ─────────────────────────────────────────────────────
def check_convention(root, folder):
    """Compare the naming convention with every readable surviving sheet in the folder."""
    d = os.path.join(root, folder)
    skip = {c for f, c, _ in FRAMES if f == folder}
    ok = bad = unreadable = 0
    res_seen = []
    for p in sorted(glob.glob(os.path.join(d, "*.tif"))):
        m = SHEET_RE.match(os.path.basename(p))
        if not m or os.path.getsize(p) == 0 or "_".join(m.groups()[:3]) in skip:
            continue
        try:
            with rasterio.open(p) as s:
                bb, res = s.bounds, s.res[0]
        except Exception:
            unreadable += 1
            continue
        want = frame_bbox("_".join(m.groups()[:3]))
        if all(abs(a - b) < 1.0 for a, b in zip((bb.left, bb.bottom, bb.right, bb.top), want)):
            ok += 1
            res_seen.append(res)
        else:
            bad += 1
    res = float(np.median(res_seen)) if res_seen else None
    return ok, bad, unreadable, res


def scan_tiles(tile_dir):
    tiles = []
    for p in sorted(glob.glob(os.path.join(tile_dir, "*.tif"))):
        tid = os.path.splitext(os.path.basename(p))[0]
        try:
            with rasterio.open(p) as s:
                b = transform_bounds(s.crs, DST_CRS, *s.bounds, densify_pts=41)
                tiles.append(dict(id=tid, path=p, year=year_of(tid), bounds=b, res=s.res[0], crs=str(s.crs)))
        except Exception as e:
            print(f"   [warn] cannot open {tid}: {str(e)[:80]}")
    return tiles


def intersects(b, bb):
    return min(b[2], bb[2]) - max(b[0], bb[0]) > 1.0 and min(b[3], bb[3]) - max(b[1], bb[1]) > 1.0


def pick_tiles(tiles, bbox, year):
    """Tiles whose footprint touches the frame: own flight year first, then nearest other years."""
    cand = [t for t in tiles if t["year"] is not None and intersects(t["bounds"], bbox)]
    cand.sort(key=lambda t: (abs(t["year"] - year), t["year"], t["id"]))
    return cand


# ── stage 2: warp ───────────────────────────────────────────────────────────
def warp_frame(bbox, res, cand, threads=8):
    x0, y0, x1, y1 = bbox
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    dst_t = from_origin(x0, y1, res, res)
    out = np.zeros((h, w), np.uint8)
    contrib = {}
    for t in cand:
        if out.all():
            break
        tmp = np.zeros((h, w), np.uint8)
        with rasterio.open(t["path"]) as src:
            reproject(source=rasterio.band(src, 1), destination=tmp,
                      dst_transform=dst_t, dst_crs=DST_CRS, resampling=Resampling.bilinear,
                      src_nodata=0, dst_nodata=0, num_threads=threads)
        fill = (out == 0) & (tmp > 0)
        n = int(fill.sum())
        out[fill] = tmp[fill]
        contrib[t["id"]] = n
    return out, contrib, dst_t


def write_tif(path, arr, transform):
    tmp = path + ".tmp"
    prof = dict(driver="GTiff", dtype="uint8", count=1, height=arr.shape[0], width=arr.shape[1],
                crs=DST_CRS, transform=transform, tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(arr, 1)
    os.replace(tmp, path)


def verify(path):
    """Decode every block; return (width, height, nodata fraction, mean of non-zero pixels)."""
    n0 = n = 0
    ssum = 0.0
    with rasterio.open(path) as s:
        for _, win in s.block_windows(1):
            a = s.read(1, window=win)
            n += a.size
            z = int((a == 0).sum())
            n0 += z
            ssum += float(a.sum(dtype=np.float64))
        w, h = s.width, s.height
    nz = n - n0
    return w, h, n0 / n, (ssum / nz if nz else 0.0)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="repo root (default: parent of scripts/)")
    ap.add_argument("--tiles", default=None, help="folder with the downloaded catalogue tiles")
    ap.add_argument("--only", nargs="*", default=None, help="sheet codes to process, e.g. 639_32_50")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--build", action="store_true")
    g.add_argument("--install", action="store_true")
    ap.add_argument("--force", action="store_true", help="install even if a frame has much nodata")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    tile_dir = a.tiles or os.path.join(root, "data", "stac_tiles")
    stage_dir = os.path.join(root, "data", "rebuilt_frames")
    bad_dir = os.path.join(root, "data", "raw_corrupt")
    frames = [f for f in FRAMES if not a.only or f[1] in a.only]
    if not frames:
        sys.exit("[error] no frame matches --only")

    # 1. naming convention vs surviving frames
    print("[stage 1] naming convention vs surviving sheets")
    folder_res = {}
    for folder in sorted({f for f, _, _ in frames}):
        ok, bad, unr, res = check_convention(root, folder)
        print(f"   {folder}: {ok} match, {bad} MISMATCH, {unr} unreadable, pixel size {res}")
        if bad:
            sys.exit("[error] the sheet-code convention does not match some surviving sheets; stopping.")
        if ok == 0:
            sys.exit(f"[error] no readable reference sheet in {folder}; cannot confirm convention.")
        folder_res[folder] = res

    # 2. tiles
    tiles = scan_tiles(tile_dir)
    print(f"[stage 2] {len(tiles)} catalogue tiles read from {tile_dir}")
    if not tiles:
        sys.exit("[error] no tiles found. Copy the downloaded tiles there first.")
    for t in tiles:
        if "3021" not in t["crs"]:
            print(f"   [warn] {t['id']} is {t['crs']}, expected EPSG:3021 (RT90 2.5 gon V)")

    os.makedirs(stage_dir, exist_ok=True)
    rows = []
    for folder, code, year in frames:
        bbox = frame_bbox(code)
        stem = f"{code}_{year}"
        tfw = read_tfw(os.path.join(root, folder, stem + ".tfw"))
        res = tfw[0] if tfw else folder_res[folder]
        if tfw and (abs(tfw[1] - bbox[0]) > 1 or abs(tfw[2] - bbox[3]) > 1):
            sys.exit(f"[error] {stem}.tfw origin ({tfw[1]}, {tfw[2]}) disagrees with the sheet code {bbox}.")
        cand = pick_tiles(tiles, bbox, year)
        print(f"\n== {stem}  bbox {bbox}  pixel {res} m  ({int(SHEET / res)} px)")
        if not cand:
            print("   no source tiles cover this frame -- download them first")
            continue
        print("   candidate tiles (own year first): " + ", ".join(f"{t['id']}" for t in cand))

        if a.plan:
            coarse = res * (20 if res < 1 else 10)
            out, contrib, _ = warp_frame(bbox, coarse, cand, a.threads)
            tot = out.size
            by_year = {}
            for t in cand:
                by_year[t["year"]] = by_year.get(t["year"], 0) + contrib.get(t["id"], 0)
            print("   estimated coverage {:.1f}%  ({})".format(
                100 * (out > 0).mean(),
                ", ".join(f"{y}: {100 * n / tot:.1f}%" for y, n in sorted(by_year.items()) if n)))
            continue

        if a.build:
            t0 = time.time()
            out, contrib, tr = warp_frame(bbox, res, cand, a.threads)
            path = os.path.join(stage_dir, folder.split("/")[-1], stem + ".tif")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_tif(path, out, tr)
            w, h, nod, mean = verify(path)
            tot = out.size
            by_year = {}
            for t in cand:
                by_year[t["year"]] = by_year.get(t["year"], 0) + contrib.get(t["id"], 0)
            ys = ", ".join(f"{y}: {100 * n / tot:.1f}%" for y, n in sorted(by_year.items()) if n)
            print(f"   built {w}x{h}, nodata {100 * nod:.2f}%, mean {mean:.1f}, sources by year [{ys}] "
                  f"({time.time() - t0:.0f} s)")
            rows.append(dict(frame=stem, folder=folder, nodata=f"{nod:.4f}", mean=f"{mean:.1f}",
                             sources=ys, tiles=" ".join(t["id"] for t in cand if contrib.get(t["id"]))))
            continue

        # install
        path = os.path.join(stage_dir, folder.split("/")[-1], stem + ".tif")
        if not os.path.exists(path):
            print("   not built yet -- run --build first")
            continue
        w, h, nod, mean = verify(path)
        if nod > MAX_NODATA and not a.force:
            print(f"   NOT installed: {100 * nod:.1f}% nodata (limit {100 * MAX_NODATA:.0f}%); use --force to override")
            continue
        dst_dir = os.path.join(root, folder)
        keep = os.path.join(bad_dir, folder.split("/")[-1])
        os.makedirs(keep, exist_ok=True)
        for suf in (".tif", ".tif.ovr", ".tif.aux.xml", ".ovr", ".aux.xml"):
            p = os.path.join(dst_dir, stem + suf)
            if os.path.exists(p):
                shutil.move(p, os.path.join(keep, os.path.basename(p)))
        shutil.copy2(path, os.path.join(dst_dir, stem + ".tif"))
        wf = os.path.join(dst_dir, stem + ".tfw")
        if not os.path.exists(wf):
            with open(wf, "w") as f:
                f.write(f"{res}\n0\n0\n{-res}\n{bbox[0] + res / 2}\n{bbox[3] - res / 2}\n")
        print(f"   installed {stem}.tif ({w}x{h}, nodata {100 * nod:.2f}%); old files in {keep}")

    if a.build and rows:
        rp = os.path.join(stage_dir, "report.csv")
        with open(rp, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nreport written to {rp}")
    if a.plan:
        print("\n[plan only] nothing was written.")


if __name__ == "__main__":
    main()
