#!/usr/bin/env python3
"""38_eval_1990_labels.py -- which inference resolution matches the hand-labelled 1990s canopy better:
production at native 1 m (era 1990) or the same imagery upsampled to 0.5 m (era 1990r05)?

For each city it takes the labelling clip (data/raw/clips_for_labelling/<city>_1990_clip.tif) as the
evaluation area, rasterises labels_<city>_1990.shp onto a 0.5 m grid over it, reads the production
canopy maps (<sheet>_<year>_canopy.tif, value 1 = canopy) of both runs onto the same grid (nearest),
and reports cover, IoU, precision and recall inside the clip's valid image area.

Needs only the GDAL Python bindings (gis env). Run after job res1990 has finished:
    python scripts/38_eval_1990_labels.py
Writes results/eval_1990_labels.csv.
"""
import csv, glob, os, sys, tempfile
import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
CITIES = ["malmo", "gtb", "sth"]
RUNS = [("1 m (production)", "1990"), ("0.5 m", "1990r05")]
RES = 0.5
tmpdir = tempfile.mkdtemp()


def bounds(ds):
    g = ds.GetGeoTransform()
    x0, y1 = g[0], g[3]
    return x0, y1 + g[5] * ds.RasterYSize, x0 + g[1] * ds.RasterXSize, y1


rows = []
for city in CITIES:
    clip = f"data/raw/clips_for_labelling/{city}_1990_clip.tif"
    shp = f"data/raw/clips_for_labelling/labels_{city}_1990.shp"
    if not (os.path.exists(clip) and os.path.exists(shp)):
        print(f"{city}: clip or labels missing"); continue
    cds = gdal.Open(clip)
    x0, y0, x1, y1 = bounds(cds)
    srs = osr.SpatialReference(wkt=cds.GetProjection())
    nx, ny = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))

    # labels -> clip CRS, then rasterise on the 0.5 m grid
    lds = ogr.Open(shp)
    lyr = lds.GetLayer()
    fields = [lyr.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(lyr.GetLayerDefn().GetFieldCount())]
    nfeat = lyr.GetFeatureCount()
    lab_gpkg = os.path.join(tmpdir, f"{city}.gpkg")
    gdal.VectorTranslate(lab_gpkg, shp, format="GPKG", dstSRS=srs.ExportToWkt())
    lab = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    lab.SetGeoTransform((x0, RES, 0, y1, 0, -RES)); lab.SetProjection(cds.GetProjection())
    gdal.Rasterize(lab, lab_gpkg, burnValues=[1])
    L = lab.ReadAsArray().astype(bool)

    img = gdal.Warp("", clip, format="MEM", outputBounds=(x0, y0, x1, y1), xRes=RES, yRes=RES,
                    resampleAlg="near").ReadAsArray()
    if img.ndim == 3:
        img = img[0]
    valid = img > 0
    print(f"\n{city}: clip {(x1-x0)/1000:.2f} x {(y1-y0)/1000:.2f} km, {nfeat} label polygons, fields {fields}, "
          f"labelled canopy {100*L[valid].mean():.1f}% of valid area")

    for name, era in RUNS:
        files = sorted(glob.glob(f"results/inference_v19/{city}_{era}/*_canopy.tif"))
        sel = []
        for f in files:
            fx0, fy0, fx1, fy1 = bounds(gdal.Open(f))
            if fx0 < x1 and fx1 > x0 and fy0 < y1 and fy1 > y0:
                sel.append(f)
        if not sel:
            print(f"  {name:18s} no canopy map overlaps the clip ({len(files)} maps in {city}_{era})"); continue
        P = gdal.Warp("", sel, format="MEM", outputBounds=(x0, y0, x1, y1), xRes=RES, yRes=RES,
                      resampleAlg="near", srcNodata="None", dstNodata=255, outputType=gdal.GDT_Byte).ReadAsArray()
        # production writes prob = -1 where it masks (outside 2023 tatort, water); score only where it predicted
        Q = gdal.Warp("", [f.replace("_canopy.tif", "_prob.tif") for f in sel], format="MEM",
                      outputBounds=(x0, y0, x1, y1), xRes=RES, yRes=RES, resampleAlg="near", srcNodata="None",
                      dstNodata=-9, outputType=gdal.GDT_Float32).ReadAsArray()
        ok = valid & (P != 255) & (Q >= 0)
        if not ok.any():
            print(f"  {name:18s} clip lies entirely in production's masked area -- not evaluable"); continue
        print(f"  {name:18s} scoring {100 * ok.sum() / max(1, valid.sum()):.0f}% of clip (inside tatort)")
        p, l = (P == 1) & ok, L & ok
        tp, fp, fn = int((p & l).sum()), int((p & ~l).sum()), int((~p & l).sum())
        iou = tp / max(1, tp + fp + fn); prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        cov_p, cov_l = 100 * p.sum() / max(1, ok.sum()), 100 * l.sum() / max(1, ok.sum())
        print(f"  {name:18s} maps {len(sel)}  cover {cov_p:5.1f}% (labels {cov_l:5.1f}%)  "
              f"IoU {iou:.3f}  precision {prec:.3f}  recall {rec:.3f}")
        rows.append(dict(city=city, run=name, era=era, maps=len(sel), cover_pred=round(cov_p, 2),
                         cover_label=round(cov_l, 2), iou=round(iou, 4), precision=round(prec, 4), recall=round(rec, 4)))

if rows:
    with open("results/eval_1990_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print("\nwrote results/eval_1990_labels.csv")
