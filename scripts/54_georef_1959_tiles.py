#!/usr/bin/env python3
"""54_georef_1959_tiles.py -- footprints for training tiles without georeferencing (malmo/tiles_1959).

The 9 malmo/tiles_1959 tiles carry no EPSG:3006 georeferencing, and no script in the repo (or its history)
created them. They hold the 1959 label polygons (data/processed/malmo/validation/labels_malmo_1959.shp,
~1 x 1 km in sheet 616_37_00), so their source is the validation clip mmo_clip.tif next to it (09 uses it
as the assessment extent) or the 1959 frame it was cut from. Per tile, sources tried in that order:
  1 the tile's own geotransform, if it has one without a CRS;
  2 offsets in the file name (<sheet>_<year>_clip_<a>_<b> or <stem>_<a>_<b>, pixels, row/col either way)
    from the top-left of the source;
  3 a search by normalised cross-correlation (NCC): coarse at ~2 m over the label extent + --search-m m,
    then to the pixel at full resolution, for each tile pixel size in --res.
Each position is checked by NCC between tile and source (and the tile's label raster, ../labels/<name>,
against the label polygons: IoU, reported only). ok = NCC >= --min-ncc and, for a search, the best coarse
peak beats the second (>= one tile away) by >= --min-gap. If a tile cannot be placed, the reason is given.

53 calls locate() for its ungeoreferenced training tiles. Run alone, this writes --out as a record
(file, xmin, ymin, xmax, ymax, res, ncc, ncc_2nd, label_iou, method, ok, why). Inputs are only read.

    python scripts/54_georef_1959_tiles.py
gis env (GDAL Python bindings).
"""
import argparse, csv, glob, os, re
import numpy as np
from osgeo import gdal, ogr

gdal.UseExceptions()
ogr.UseExceptions()
TILES = "data/processed/malmo/tiles_1959/images/*.tif"
CLIP = "data/processed/malmo/validation/mmo_clip.tif"
FRAMES = "data/raw/malmo/1960_OF_gray_mmo/*_1959.tif"
LABELS = "data/processed/malmo/validation/labels_malmo_1959.shp"
NAME_SHEET = re.compile(r"(\d{3}_\d{2}_\d{2})_(\d{4})_clip_(\d+)_(\d+)")
NAME_OFFS = re.compile(r"_(\d{3,5})_(\d{3,5})$")


def read_tile(path):
    d = gdal.Open(path)
    a = d.ReadAsArray().astype(np.float64)
    return (a[:3].mean(0) if a.ndim == 3 else a), d.GetGeoTransform()


def region(src, box, res):
    alg = "near" if res <= 0.5 else "average"
    a = gdal.Warp("", src, format="MEM", outputBounds=box, xRes=res, yRes=res, dstSRS="EPSG:3006",
                  resampleAlg=alg, srcNodata=0, dstNodata=0, outputType=gdal.GDT_Float32).ReadAsArray()
    return (a[:3].mean(0) if a.ndim == 3 else a).astype(np.float64)


def down(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))


def ncc_map(S, T):
    """NCC of template T at every valid offset in S (FFT); -1 where the window has > 1 % nodata (0)."""
    (H, W), (h, w) = S.shape, T.shape
    t = T - T.mean()
    corr = np.fft.irfft2(np.fft.rfft2(S) * np.conj(np.fft.rfft2(t, S.shape)), S.shape)[:H - h + 1, :W - w + 1]

    def box_sum(A):
        I = np.zeros((H + 1, W + 1))
        I[1:, 1:] = A.cumsum(0).cumsum(1)
        return I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]

    n = h * w
    s1, s2 = box_sum(S), box_sum(S * S)
    r = corr / np.maximum(np.sqrt(np.maximum(s2 - s1 * s1 / n, 0)) * np.sqrt((t * t).sum()), 1e-9)
    r[box_sum((S == 0).astype(np.float64)) > 0.01 * n] = -1
    return r


def ncc1(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))


def check(T, src, box, res):
    S = region(src, box, res)[:T.shape[0], :T.shape[1]]
    return ncc1(S, T) if S.shape == T.shape and (S == 0).mean() <= 0.01 else -1.0


def search(T, src, sbox, res):
    """-> (ncc, ncc of the second coarse peak, footprint box) for pixel size res."""
    S = region(src, sbox, res)
    if S.shape[0] < T.shape[0] or S.shape[1] < T.shape[1]:
        return -1.0, -1.0, None
    k = max(1, int(round(2.0 / res)))
    Tc = down(T, k)
    m = ncc_map(down(S, k), Tc)
    y0, x0 = np.unravel_index(np.argmax(m), m.shape)
    m2 = m.copy()
    hy, hx = Tc.shape
    m2[max(0, y0 - hy):y0 + hy, max(0, x0 - hx):x0 + hx] = -1
    second = float(m2.max())
    h, w = T.shape
    best = (-2.0, 0, 0)
    for y in range(max(0, y0 * k - 2 * k), min(S.shape[0] - h, y0 * k + 2 * k) + 1):
        for x in range(max(0, x0 * k - 2 * k), min(S.shape[1] - w, x0 * k + 2 * k) + 1):
            win = S[y:y + h, x:x + w]
            if (win == 0).mean() <= 0.01:
                best = max(best, (ncc1(win, T), y, x))
    r, y, x = best
    xmin, ymax = sbox[0] + x * res, sbox[3] - y * res
    return r, second, (xmin, ymax - h * res, xmin + w * res, ymax)


def name_boxes(f, shape, srcs):
    """Footprints implied by offsets in the file name: [(box, res, method)], both row/col orders."""
    stem, (h, w) = os.path.splitext(os.path.basename(f))[0], shape
    m = NAME_SHEET.search(stem)
    if m:
        cands, offs = [s for s in srcs if os.path.basename(s).startswith(f"{m.group(1)}_{m.group(2)}")], m.group(3, 4)
    else:
        m = NAME_OFFS.search(stem)
        if not m:
            return []
        cands, offs = srcs, m.group(1, 2)
    out = []
    for s in cands:
        gt = gdal.Open(s).GetGeoTransform()
        for r, c in ((int(offs[0]), int(offs[1])), (int(offs[1]), int(offs[0]))):
            xmin, ymax = gt[0] + c * gt[1], gt[3] + r * gt[5]
            out.append(((xmin, ymax - h * gt[1], xmin + w * gt[1], ymax), gt[1],
                        f"name offsets row {r} col {c} in {os.path.basename(s)}"))
    return out


def label_iou(tile, box, shp):
    lab = os.path.join(os.path.dirname(os.path.dirname(tile)), "labels", os.path.basename(tile))
    if not os.path.exists(lab) or not os.path.exists(shp):
        return ""
    L = gdal.Open(lab).ReadAsArray()
    L = L[0] if L.ndim == 3 else L
    L = (L == 1) if (L == 1).any() else (L > 0) & (L != 255)
    h, w = L.shape
    m = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    m.SetGeoTransform((box[0], (box[2] - box[0]) / w, 0, box[3], 0, -(box[3] - box[1]) / h))
    m.SetProjection(ogr.Open(shp).GetLayer(0).GetSpatialRef().ExportToWkt())
    gdal.Rasterize(m, shp, burnValues=[1])
    P = m.ReadAsArray().astype(bool)
    u = (L | P).sum()
    return round(float((L & P).sum() / u), 3) if u else ""


def locate(tiles, clip=CLIP, frames=FRAMES, labels=LABELS, search_m=1000.0, res=(0.5, 1.0), min_ncc=0.8,
           min_gap=0.1, log=print):
    """-> one row per tile: file, xmin, ymin, xmax, ymax, res, ncc, ncc_2nd, label_iou, method, ok, why."""
    groups = [(n, s) for n, s in (("validation clip", [clip] if os.path.exists(clip) else []),
                                  ("1959 frames", sorted(glob.glob(frames)))) if s]
    log(f"  sources: " + ("; ".join(f"{n} ({len(s)} files)" for n, s in groups) or "NONE") +
        f" [looked for {clip} and {frames}]")
    if not groups:
        return [dict(file=f, ok=0, why=f"no source imagery: neither {clip} nor {frames} exists") for f in tiles]
    if os.path.exists(labels):
        e = ogr.Open(labels).GetLayer(0).GetExtent()          # xmin, xmax, ymin, ymax
        sbox = (np.floor(e[0] - search_m), np.floor(e[2] - search_m), np.ceil(e[1] + search_m), np.ceil(e[3] + search_m))
    else:
        sbox = None
        log(f"  no label polygons at {labels}: search limited to the validation clip's extent")
    rows = []
    for f in tiles:
        T, gt = read_tile(f)
        h, w = T.shape
        tried, best = [], None                       # best: (ncc, second, box, res, method)
        for gname, srcs in groups:
            vrt = gdal.BuildVRT("", srcs, srcNodata=0, VRTNodata=0, resolution="highest")
            cands = []
            if gt != (0, 1, 0, 0, 0, 1) and 2e5 < gt[0] < 1e6 and 6e6 < gt[3] < 7.7e6:
                cands.append(((gt[0], gt[3] + gt[5] * h, gt[0] + gt[1] * w, gt[3]), gt[1], "own geotransform"))
            cands += name_boxes(f, (h, w), srcs)
            for box, r_, method in cands:
                r = check(T, vrt, box, r_)
                tried.append(f"{method}: NCC {r:.2f}")
                if best is None or r > best[0]:
                    best = (r, "", box, r_, method)
            if best and best[0] >= min_ncc:
                break
            g = sbox
            if g is None:
                d = gdal.Open(srcs[0]); x = d.GetGeoTransform()
                g = (x[0], x[3] + x[5] * d.RasterYSize, x[0] + x[1] * d.RasterXSize, x[3])
            for r_ in res:
                r, second, box = search(T, vrt, tuple(float(v) for v in g), r_)
                tried.append(f"search {gname} at {r_} m: NCC {r:.2f} (2nd {second:.2f})")
                if box and (best is None or r > best[0]):
                    best = (r, second, box, r_, f"search in {gname}")
            if best and best[0] >= min_ncc and (best[1] == "" or best[0] - best[1] >= min_gap):
                break
        r, second, box, r_, method = best if best and best[2] else (-1.0, "", None, "", "")
        ok = box is not None and r >= min_ncc and (second == "" or r - second >= min_gap)
        why = "" if ok else ("best NCC below %.2f or ambiguous peak; tried: " % min_ncc + "; ".join(tried))
        row = dict(file=f, xmin="", ymin="", xmax="", ymax="", res=r_, ncc=round(r, 4),
                   ncc_2nd="" if second == "" else round(second, 4), label_iou="", method=method, ok=int(ok), why=why)
        if box:
            row.update(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3], label_iou=label_iou(f, box, labels))
        rows.append(row)
        log(f"    {os.path.basename(f)} {w}x{h}: " +
            (f"{method}, res {r_} -> E {box[0]:.1f}-{box[2]:.1f} N {box[1]:.1f}-{box[3]:.1f}, NCC {r:.3f}, "
             f"label IoU {row['label_iou']}" if ok else f"NOT PLACED ({why})"))
    return rows


def main(a):
    if os.path.exists(a.out) and not a.overwrite:
        print(f"{a.out} exists, kept (--overwrite to redo)"); return
    tiles = sorted(glob.glob(a.tiles))
    print(f"{len(tiles)} tiles ({a.tiles})")
    if not tiles:
        return
    rows = locate(tiles, a.clip, a.frames, a.labels, a.search_m, a.res, a.min_ncc, a.min_gap)
    fields = ["file", "xmin", "ymin", "xmax", "ymax", "res", "ncc", "ncc_2nd", "label_iou", "method", "ok", "why"]
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", restval="")
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {a.out}: {sum(r['ok'] for r in rows)}/{len(rows)} ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default=TILES)
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--out", default="data/processed/malmo/tiles_1959_footprints.csv")
    ap.add_argument("--search-m", type=float, default=1000.0, help="search margin around the label extent, m")
    ap.add_argument("--res", type=float, nargs="+", default=[0.5, 1.0], help="tile pixel sizes to try, m")
    ap.add_argument("--min-ncc", type=float, default=0.8)
    ap.add_argument("--min-gap", type=float, default=0.1)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
