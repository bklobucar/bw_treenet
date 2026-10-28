#!/usr/bin/env python3
"""54_georef_1959_tiles.py -- footprints for training tiles without georeferencing (malmo/tiles_1959).

53 excludes test points near every training tile, but the 9 malmo/tiles_1959 tiles carry no EPSG:3006
georeferencing. They were cut from the 1959 Malmö imagery under the 901 label polygons
(data/processed/malmo/validation/labels_malmo_1959.shp, ~1 x 1 km in sheet 616_37_00). Each tile is located
in that imagery (--src) by normalised cross-correlation (NCC): coarse at ~2 m over the label extent +
--search-m m, then to the pixel at full resolution, for each tile pixel size in --res. A tile that has a
geotransform but no CRS is first checked at that position. As a check, the tile's label raster
(../labels/<same name>) is compared with the label polygons rasterised on the found footprint (IoU).

ok = 1 when NCC >= --min-ncc and the best coarse peak beats the second (>= one tile away) by >= --min-gap.
Writes --out (file, xmin, ymin, xmax, ymax, res, ncc, ncc_2nd, label_iou, method, ok); 53 adds the ok rows
to its historical training exclusion. Tiles, labels and imagery are only read; an existing --out is kept
unless --overwrite.

    python scripts/54_georef_1959_tiles.py
gis env (GDAL Python bindings).
"""
import argparse, csv, glob, os
import numpy as np
from osgeo import gdal, ogr

gdal.UseExceptions()
ogr.UseExceptions()


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


def search(T, src, sbox, res):
    """-> (ncc, ncc of the second coarse peak, footprint box) for pixel size res."""
    S = region(src, sbox, res)
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


def label_iou(tile, box, shp):
    lab = os.path.join(os.path.dirname(os.path.dirname(tile)), "labels", os.path.basename(tile))
    if not os.path.exists(lab):
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


def main(a):
    if os.path.exists(a.out) and not a.overwrite:
        print(f"{a.out} exists, kept (--overwrite to redo)"); return
    tiles = sorted(glob.glob(a.tiles))
    srcs = sorted(glob.glob(a.src))
    print(f"{len(tiles)} tiles ({a.tiles}); {len(srcs)} source images ({a.src})")
    if not tiles or not srcs:
        print("nothing to do"); return
    src = gdal.BuildVRT("", srcs, srcNodata=0, VRTNodata=0, resolution="highest")
    ext = ogr.Open(a.labels).GetLayer(0).GetExtent()           # xmin, xmax, ymin, ymax
    g = a.search_m
    sbox = tuple(float(v) for v in (np.floor(ext[0] - g), np.floor(ext[2] - g), np.ceil(ext[1] + g), np.ceil(ext[3] + g)))
    print(f"label extent E {ext[0]:.0f}-{ext[1]:.0f} N {ext[2]:.0f}-{ext[3]:.0f}; search box {sbox}")
    rows = []
    for f in tiles:
        T, gt = read_tile(f)
        h, w = T.shape
        res_, best = None, None
        if gt != (0, 1, 0, 0, 0, 1) and 2e5 < gt[0] < 1e6 and 6e6 < gt[3] < 7.7e6:
            box = (gt[0], gt[3] + gt[5] * h, gt[0] + gt[1] * w, gt[3])
            S = region(src, box, gt[1])[:h, :w]
            best = (ncc1(S, T), "", box, gt[1], "geotransform")
        if best is None or best[0] < a.min_ncc:
            for res in a.res:
                r, second, box = search(T, src, sbox, res)
                if best is None or r > best[0]:
                    best = (r, second, box, res, "search")
        r, second, box, res_, method = best
        ok = r >= a.min_ncc and (second == "" or r - second >= a.min_gap)
        iou = label_iou(f, box, a.labels)
        rows.append(dict(file=f, xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3], res=res_, ncc=round(r, 4),
                         ncc_2nd="" if second == "" else round(second, 4), label_iou=iou, method=method, ok=int(ok)))
        print(f"  {os.path.basename(f)} {w}x{h}: {method} res {res_} -> E {box[0]:.1f}-{box[2]:.1f} "
              f"N {box[1]:.1f}-{box[3]:.1f}; NCC {r:.3f} (2nd {second if second == '' else f'{second:.3f}'}), "
              f"label IoU {iou}; {'OK' if ok else 'NOT OK'}", flush=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {a.out}: {sum(r['ok'] for r in rows)}/{len(rows)} ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default="data/processed/malmo/tiles_1959/images/*.tif")
    ap.add_argument("--src", default="data/raw/malmo/1960_OF_gray_mmo/*_1959.tif")
    ap.add_argument("--labels", default="data/processed/malmo/validation/labels_malmo_1959.shp")
    ap.add_argument("--out", default="data/processed/malmo/tiles_1959_footprints.csv")
    ap.add_argument("--search-m", type=float, default=1000.0, help="search margin around the label extent, m")
    ap.add_argument("--res", type=float, nargs="+", default=[0.5, 1.0], help="tile pixel sizes to try, m")
    ap.add_argument("--min-ncc", type=float, default=0.8)
    ap.add_argument("--min-gap", type=float, default=0.1)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
