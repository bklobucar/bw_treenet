#!/usr/bin/env python3
"""57_label_stats.py -- geometry of the historical training labels, to check the labelling protocol rules
"fill gaps narrower than 2 m" and "minimum crown 2 m across" against the labels the model was trained on.

Rasters: every data/processed/*/tiles_19*/labels/*.tif (or --sets). Canopy = label value 1; other values (255,
2-254) are counted and reported, not counted as canopy here (Forest_dataset sets every value > 1 to 1 in
training). Pixel size from the geotransform; tiles without one (malmo/tiles_1959) are taken as --default-res
and flagged. Per tile and per set:
  * canopy patches: 8-connected canopy components (gdal.Polygonize, so areas are exact pixel areas). "interior"
    = not touching the tile edge (edge patches are cut by the tile, so their size is a lower bound). Smallest
    sizes, percentiles, and the count share below pi m2 (a 2 m disc) and below 4 m2 (2 m x 2 m).
  * holes: 4-connected background (value 0) components inside canopy, i.e. not touching the tile edge. Size
    distribution, share of holes < 4 m2 by count and by area, hole area as % of canopy area.
  * 2 m closing / opening with a disc of radius round(1 m / pixel) px (5 px across at 0.5 m, 3 px at 1 m):
    closing = canopy area that "fill gaps narrower than 2 m" would add (gaps inside canopy AND between separate
    canopy polygons), as % of canopy; opening = canopy area in parts narrower than 2 m, as % of canopy. Tiles
    are edge-padded, so the tile border neither adds nor removes canopy.
Vectors: every label shapefile/GeoPackage under data/raw/clips_for_labelling and data/processed/*/validation
(or --vectors): features, interior rings (holes kept in the polygons), overlap between features, connected parts
after a union (features that touch merge), and polygon areas. Many small disjoint features with no rings that
stay separate after the union = individual crowns; large features, rings, or features that merge = merged canopy.
Reflect-padded 512 px tiles (the *new sets) repeat their content, so their patches and holes are counted up to 4x.

Outputs in --out (results/label_stats): label_stats.md, per_set.csv, per_tile.csv, holes.csv (one row per hole),
patches_small.csv (interior patches < 10 m2), vectors.csv.
    python scripts/57_label_stats.py                 # gis env (GDAL + numpy)
"""
import argparse, csv, glob, os
import numpy as np
from osgeo import gdal, ogr

gdal.UseExceptions()
ogr.UseExceptions()
SETS = "data/processed/*/tiles_19*"
VECTORS = ["data/raw/clips_for_labelling/*.shp", "data/raw/clips_for_labelling/*.gpkg",
           "data/processed/*/validation/*.shp", "data/processed/*/validation/*.gpkg"]
DISC = np.pi            # m2, a disc 2 m across
SQ = 4.0                # m2, 2 m x 2 m
HOLE_BINS = [0, 1, 4, 10, 25, 100, np.inf]      # m2


# ── rasters ──────────────────────────────────────────────────────────────────
def read_label(path, default_res):
    d = gdal.Open(path)
    a = d.GetRasterBand(1).ReadAsArray()
    gt = d.GetGeoTransform()
    if gt == (0, 1, 0, 0, 0, 1):
        return a, default_res, False
    return a, abs(gt[1]), True


def components(mask, res, eight):
    """Connected components of mask -> [(area m2, touches the tile edge)], via gdal.Polygonize."""
    h, w = mask.shape
    m = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    m.SetGeoTransform((0, res, 0, 0, 0, -res))
    b = m.GetRasterBand(1)
    b.WriteArray(mask.astype(np.uint8))
    vds = ogr.GetDriverByName("Memory").CreateDataSource("")
    lyr = vds.CreateLayer("c", geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(b, b, lyr, 0, ["8CONNECTED=8"] if eight else [])
    out, eps = [], res / 4
    for f in lyr:
        x0, x1, y0, y1 = f.GetGeometryRef().GetEnvelope()
        edge = x0 < eps or y1 > -eps or x1 > w * res - eps or y0 < -h * res + eps
        out.append((f.GetGeometryRef().GetArea(), edge))
    return out


def disc(r):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]


def morph(mask, r, op):
    """Binary dilation (op 'or') or erosion (op 'and') with a disc of radius r px; edge-padded."""
    p = np.pad(mask, r, mode="edge")
    h, w = mask.shape
    out = np.zeros_like(mask) if op == "or" else np.ones_like(mask)
    for dy, dx in disc(r):
        s = p[r + dy:r + dy + h, r + dx:r + dx + w]
        out = (out | s) if op == "or" else (out & s)
    return out


def tile_stats(path, default_res):
    a, res, geo = read_label(path, default_res)
    can = a == 1
    px = res * res
    vals = {"v0": int((a == 0).sum()), "v1": int(can.sum()), "v255": int((a == 255).sum())}
    vals["vother"] = a.size - sum(vals.values())
    patches = components(can, res, eight=True)
    holes = [(ar, e) for ar, e in components(a == 0, res, eight=False) if not e]
    r = max(1, int(round(1.0 / res)))
    closed = morph(morph(can, r, "or"), r, "and")
    opened = morph(morph(can, r, "and"), r, "or")
    return dict(res=res, georef=geo, px=px, n=a.size, canopy_m2=can.sum() * px, vals=vals, patches=patches,
                holes=[ar for ar, _ in holes], close_add_m2=(closed & ~can).sum() * px,
                open_rm_m2=(can & ~opened).sum() * px, r_px=r)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarise(rows):
    """rows: tile_stats dicts -> summary dict (one tile or a whole set)."""
    can = sum(r["canopy_m2"] for r in rows)
    area = sum(r["n"] * r["px"] for r in rows)
    pin = np.array([p for r in rows for p, e in r["patches"] if not e])
    pall = np.array([p for r in rows for p, _ in r["patches"]])
    hol = np.array([h for r in rows for h in r["holes"]])
    v = {k: sum(r["vals"][k] for r in rows) for k in ("v0", "v1", "v255", "vother")}
    n = sum(v.values())
    s = dict(tiles=len(rows), res="/".join(sorted({f"{r['res']:g}" for r in rows})),
             no_georef=sum(not r["georef"] for r in rows), canopy_pct=100 * can / area if area else 0,
             pct_255=100 * v["v255"] / n, pct_other=100 * v["vother"] / n,
             patches=len(pall), patches_interior=len(pin),
             patch_min=float(pin.min()) if len(pin) else float("nan"), patch_p5=pct(pin, 5),
             patch_p25=pct(pin, 25), patch_median=pct(pin, 50),
             patch_lt_disc_pct=100 * (pin < DISC).mean() if len(pin) else float("nan"),
             patch_lt_4_pct=100 * (pin < SQ).mean() if len(pin) else float("nan"),
             patch_smallest=" ".join(f"{x:g}" for x in np.sort(pin)[:8]),
             holes=len(hol), hole_area_pct_of_canopy=100 * hol.sum() / can if can else float("nan"),
             hole_median=pct(hol, 50), hole_p90=pct(hol, 90), hole_max=float(hol.max()) if len(hol) else float("nan"),
             hole_lt_4_pct=100 * (hol < SQ).mean() if len(hol) else float("nan"),
             hole_lt_4_area_pct=100 * hol[hol < SQ].sum() / hol.sum() if hol.sum() else float("nan"),
             close2m_add_pct=100 * sum(r["close_add_m2"] for r in rows) / can if can else float("nan"),
             open2m_rm_pct=100 * sum(r["open_rm_m2"] for r in rows) / can if can else float("nan"))
    for lo, hi in zip(HOLE_BINS[:-1], HOLE_BINS[1:]):
        s[f"holes_{lo:g}_{hi:g}"] = int(((hol >= lo) & (hol < hi)).sum())
    return s


# ── vectors ──────────────────────────────────────────────────────────────────
def polys(g):
    n = g.GetGeometryName()
    if n == "POLYGON":
        return [g]
    if n in ("MULTIPOLYGON", "GEOMETRYCOLLECTION"):
        return [p for i in range(g.GetGeometryCount()) for p in polys(g.GetGeometryRef(i))]
    return []


def vector_stats(path):
    out = []
    ds = ogr.Open(path)
    for li in range(ds.GetLayerCount()):
        lyr = ds.GetLayer(li)
        if lyr.GetGeomType() not in (ogr.wkbPolygon, ogr.wkbMultiPolygon, ogr.wkbPolygon25D, ogr.wkbMultiPolygon25D,
                                     ogr.wkbUnknown):
            continue
        areas, rings, ring_areas, parts = [], 0, [], ogr.Geometry(ogr.wkbMultiPolygon)
        for f in lyr:
            g = f.GetGeometryRef()
            if g is None:
                continue
            ps = polys(g)
            for p in ps:
                parts.AddGeometry(p)
                for j in range(1, p.GetGeometryCount()):
                    r = ogr.Geometry(ogr.wkbPolygon)
                    r.AddGeometry(p.GetGeometryRef(j))
                    ring_areas.append(r.GetArea())
            rings += sum(p.GetGeometryCount() - 1 for p in ps)
            areas.append(g.GetArea())
        if not areas:
            continue
        a = np.array(areas)
        un = parts.UnionCascaded() if parts.GetGeometryCount() else None
        n_un = len(polys(un)) if un is not None else 0
        out.append(dict(file=path, layer=lyr.GetName(), features=len(a), area_m2=round(a.sum(), 1),
                        union_m2=round(un.GetArea(), 1) if un else 0,
                        overlap_pct=round(100 * (1 - un.GetArea() / a.sum()), 2) if un and a.sum() else 0,
                        parts_after_union=n_un, rings=rings,
                        ring_lt_4=sum(x < SQ for x in ring_areas),
                        ring_median_m2=round(pct(ring_areas, 50), 2) if ring_areas else "",
                        area_min=round(a.min(), 2), area_p10=round(pct(a, 10), 2), area_median=round(pct(a, 50), 2),
                        area_p90=round(pct(a, 90), 1), area_max=round(a.max(), 1),
                        lt_disc=int((a < DISC).sum()), lt_4=int((a < SQ).sum())))
    return out


# ── output ───────────────────────────────────────────────────────────────────
def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})


def f1(x):
    return "" if x != x else f"{x:.1f}"


def main(a):
    os.makedirs(a.out, exist_ok=True)
    sets = sorted({d for p in a.sets for d in glob.glob(p) if os.path.isdir(os.path.join(d, "labels"))})
    print(f"{len(sets)} label sets")
    per_set, per_tile, hole_rows, small_rows = [], [], [], []
    for s in sets:
        name = os.path.relpath(s, "data/processed").replace("\\", "/")
        files = sorted(glob.glob(os.path.join(s, "labels", "*.tif")))
        rows = []
        for f in files:
            t = tile_stats(f, a.default_res)
            rows.append(t)
            tn = os.path.basename(f)
            per_tile.append(dict(set=name, tile=tn, **summarise([t])))
            hole_rows += [dict(set=name, tile=tn, area_m2=round(h, 3)) for h in t["holes"]]
            small_rows += [dict(set=name, tile=tn, area_m2=round(p, 3)) for p, e in t["patches"] if not e and p < 10]
        if rows:
            per_set.append(dict(set=name, **summarise(rows)))
            r = per_set[-1]
            print(f"  {name}: {r['tiles']} tiles, res {r['res']}, canopy {r['canopy_pct']:.1f} %, "
                  f"{r['patches_interior']} interior patches (min {f1(r['patch_min'])} m2), {r['holes']} holes "
                  f"({f1(r['hole_lt_4_pct'])} % < 4 m2), 2 m closing +{f1(r['close2m_add_pct'])} %", flush=True)
    vec = []
    for p in a.vectors:
        for f in sorted(glob.glob(p)):
            try:
                vec += vector_stats(f)
            except RuntimeError as e:
                print(f"  vector {f}: unreadable ({e})")
    for v in vec:
        print(f"  {v['file']}: {v['features']} features, {v['parts_after_union']} parts after union, "
              f"{v['rings']} rings, median area {v['area_median']} m2")
    write_csv(os.path.join(a.out, "per_set.csv"), per_set)
    write_csv(os.path.join(a.out, "per_tile.csv"), per_tile)
    write_csv(os.path.join(a.out, "holes.csv"), hole_rows)
    write_csv(os.path.join(a.out, "patches_small.csv"), small_rows)
    write_csv(os.path.join(a.out, "vectors.csv"), vec)
    report(a, per_set, vec)


def report(a, per_set, vec):
    L = ["# Historical training labels: geometry (57_label_stats.py)", "",
         "Canopy = label value 1. Patches: 8-connected canopy, interior = not touching the tile edge. Holes: "
         "4-connected value-0 areas not touching the tile edge. 2 m closing/opening: disc of radius "
         "round(1 m / pixel) px. *new sets are 512 px tiles reflect-padded to 1000 px, so counts repeat.", "",
         "## Canopy patches (minimum crown 2 m across)", "",
         "| set | tiles | res m | canopy % | value 255 % | other % | patches | interior | min m2 | p5 | p25 | median "
         "| % < pi m2 | % < 4 m2 | smallest interior (m2) | 2 m opening removes % of canopy |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_set:
        L.append(f"| {r['set']} | {r['tiles']}{' (no georef)' if r['no_georef'] else ''} | {r['res']} | "
                 f"{r['canopy_pct']:.1f} | {r['pct_255']:.2f} | {r['pct_other']:.2f} | {r['patches']} | "
                 f"{r['patches_interior']} | {f1(r['patch_min'])} | {f1(r['patch_p5'])} | {f1(r['patch_p25'])} | "
                 f"{f1(r['patch_median'])} | {f1(r['patch_lt_disc_pct'])} | {f1(r['patch_lt_4_pct'])} | "
                 f"{r['patch_smallest']} | {f1(r['open2m_rm_pct'])} |")
    bins = [f"holes_{lo:g}_{hi:g}" for lo, hi in zip(HOLE_BINS[:-1], HOLE_BINS[1:])]
    L += ["", "## Holes inside canopy (fill gaps narrower than 2 m)", "",
          "| set | holes | hole area % of canopy | median m2 | p90 | max | % of holes < 4 m2 | % of hole area in holes "
          "< 4 m2 | " + " | ".join(b.replace("holes_", "").replace("_", "-") + " m2" for b in bins)
          + " | 2 m closing adds % of canopy |",
          "|---|---|---|---|---|---|---|---|" + "---|" * len(bins) + "---|"]
    for r in per_set:
        L.append(f"| {r['set']} | {r['holes']} | {f1(r['hole_area_pct_of_canopy'])} | {f1(r['hole_median'])} | "
                 f"{f1(r['hole_p90'])} | {f1(r['hole_max'])} | {f1(r['hole_lt_4_pct'])} | "
                 f"{f1(r['hole_lt_4_area_pct'])} | " + " | ".join(str(r[b]) for b in bins)
                 + f" | {f1(r['close2m_add_pct'])} |")
    L += ["", "## Label vectors", "",
          "| file | layer | features | parts after union | overlap % | rings | rings < 4 m2 | area min | p10 | median "
          "| p90 | max | < pi m2 | < 4 m2 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in vec:
        L.append(f"| {v['file']} | {v['layer']} | {v['features']} | {v['parts_after_union']} | {v['overlap_pct']} | "
                 f"{v['rings']} | {v['ring_lt_4']} | {v['area_min']} | {v['area_p10']} | {v['area_median']} | "
                 f"{v['area_p90']} | {v['area_max']} | {v['lt_disc']} | {v['lt_4']} |")
    L += ["", "Reading: if labels follow 'fill gaps narrower than 2 m', the 2 m closing adds ~0 % and holes < 4 m2 "
          "are rare; if they follow 'minimum crown 2 m across', interior patches < pi m2 are rare and the 2 m "
          "opening removes ~0 %. Crown-by-crown labels show a large closing gain (gaps between adjacent crowns)."]
    p = os.path.join(a.out, "label_stats.md")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=[SETS], help="set dirs (globs) holding labels/*.tif")
    ap.add_argument("--vectors", nargs="+", default=VECTORS, help="label vector files (globs)")
    ap.add_argument("--default-res", type=float, default=0.5, help="pixel size of tiles without a geotransform, m")
    ap.add_argument("--out", default="results/label_stats")
    main(ap.parse_args())
