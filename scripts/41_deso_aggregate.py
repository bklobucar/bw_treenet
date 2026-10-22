#!/usr/bin/env python3
"""41_deso_aggregate.py -- canopy cover per DeSO 2025 and era (v19), for the v0.1 web map.

Per city and era, on each 5 x 5 km sheet's grid at 1 m:
  * DeSO codes are rasterised (pixel centres);
  * measured = prob >= 0 (production writes prob = -1 outside the 2023 tätort and on water),
    as a fraction per 1 m cell (average warp of the 0/1 mask), restricted to the tätort mask;
  * canopy = average warp of the 0/1 canopy map (srcNodata="None": map value 0 is NOT nodata).

Per DeSO and era: measured_km2, canopy_pct (canopy / measured).
Common footprint per DeSO within each series (film: 1960 1970 1990r05; digital: 2005 2010 [+2020 if
present for that city]): 1 m cells fully measured in every era of the series. canopy_pct_common is
measured on that footprint, coverage_pct = measured_common_km2 / deso_tatort_km2.
NO change is computed, in particular never between film and digital eras (source effect, see
notes/paper2_notes_2026-10.md).

licence_status per DeSO and era, from the flight years in the file names (<sheet>_<year>_canopy.tif)
of the sheets that measured that DeSO: "public" if all years <= 2005 (Lantmäteriet historical
orthophotos, CC0), "pending" if any year >= 2006 (Ortofoto Nedladdning, CC BY 4.0, awaiting
Lantmäteriet). Mixed DeSOs are "pending" (conservative).

Only DeSOs with measured area in at least one era are written. kommunkod and kommunnamn are carried
over from the DeSO layer where those fields exist (DeSO 2025 has kommunkod only).

    python scripts/41_deso_aggregate.py --deso data/ext/DeSO_2025.gpkg
Writes results/deso/deso_canopy_v19.gpkg (one row per DeSO, columns per era) and
results/deso/deso_canopy_v19_long.csv (one row per DeSO and era). gis env (GDAL Python bindings).
"""
import argparse, csv, importlib, os, sys
import numpy as np
from osgeo import gdal, ogr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cf = importlib.import_module("39_common_footprint")   # sheet_box, warp, tatort, SHEET, INF, TAT, CITIES

gdal.UseExceptions()
ogr.UseExceptions()
SERIES = {"film": ["1960", "1970", "1990r05"], "digital": ["2005", "2010"]}
OPTIONAL = {"digital": ["2020"]}          # used only where the inference exists
PUBLIC_MAX_YEAR = 2005
RES = 1.0
OUT_DIR = "results/deso"
OUT_GPKG = os.path.join(OUT_DIR, "deso_canopy_v19.gpkg")
OUT_CSV = os.path.join(OUT_DIR, "deso_canopy_v19_long.csv")
CARRY = ("kommunkod", "kommunnamn")      # copied from the DeSO layer when present


def deso_src(deso_ds, layer):
    return deso_ds.GetLayerByName(layer) if layer else deso_ds.GetLayer(0)


def carry_fields(deso_ds, layer):
    d = deso_src(deso_ds, layer).GetLayerDefn()
    names = {d.GetFieldDefn(i).GetName() for i in range(d.GetFieldCount())}
    return [k for k in CARRY if k in names]


def index(city, era):
    """sheet -> (flight year, canopy path, prob path) from results/inference_v19/<city>_<era>/."""
    out = {}
    d = os.path.join(cf.INF, f"{city}_{era}")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith("_canopy.tif"):
            continue
        m = cf.SHEET.search(f)
        prob = os.path.join(d, f.replace("_canopy.tif", "_prob.tif"))
        if not m:
            continue
        if not os.path.exists(prob):
            print(f"  [warn] {city}_{era} {f}: no _prob.tif, sheet skipped"); continue
        out[m.group(1)] = (int(m.group(2)), os.path.join(d, f), prob)
    return out


def licence(years):
    if not years:
        return None
    return "public" if max(years) <= PUBLIC_MAX_YEAR else "pending"


def measured_frac(prob_path, box):
    """Fraction of each 1 m cell with prob >= 0 (average warp of the 0/1 mask at native resolution)."""
    d = gdal.Open(prob_path)
    p = d.GetRasterBand(1).ReadAsArray()
    m = gdal.GetDriverByName("MEM").Create("", d.RasterXSize, d.RasterYSize, 1, gdal.GDT_Byte)
    m.SetGeoTransform(d.GetGeoTransform())
    m.SetProjection(d.GetProjection())
    m.GetRasterBand(1).WriteArray((p >= 0).astype(np.uint8))
    del p
    return cf.warp(m, box, RES, "average", gdal.GDT_Float32)


def deso_layer(deso_ds, layer, field, bbox, carry):
    """DeSOs intersecting bbox -> (codes, carried attributes, geometries,
    in-memory layer with integer field rid = 1..N)."""
    src = deso_src(deso_ds, layer)
    src.SetSpatialFilterRect(*bbox)
    codes, attrs, geoms = [], [], []
    for f in src:
        codes.append(f.GetField(field)); geoms.append(f.GetGeometryRef().Clone())
        attrs.append({k: f.GetField(k) for k in carry})
    src.SetSpatialFilter(None)
    drv = ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")
    mem = drv.CreateDataSource("")
    lyr = mem.CreateLayer("deso", src.GetSpatialRef(), ogr.wkbMultiPolygon)
    lyr.CreateField(ogr.FieldDefn("rid", ogr.OFTInteger))
    for i, g in enumerate(geoms, start=1):
        f = ogr.Feature(lyr.GetLayerDefn()); f.SetField("rid", i); f.SetGeometry(g); lyr.CreateFeature(f)
    return codes, attrs, geoms, mem


def rasterise_ids(mem, box):
    nx, ny = int(round((box[2] - box[0]) / RES)), int(round((box[3] - box[1]) / RES))
    r = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Int32)
    r.SetGeoTransform((box[0], RES, 0, box[3], 0, -RES))
    r.SetProjection(mem.GetLayer(0).GetSpatialRef().ExportToWkt())
    gdal.Rasterize(r, mem, attribute="rid", layers=["deso"])
    return r.ReadAsArray()


def tatort_km2(geom, tat_lyr):
    tat_lyr.SetSpatialFilter(geom)
    a = 0.0
    for t in tat_lyr:
        tg = t.GetGeometryRef()
        try:
            a += geom.Intersection(tg).GetArea()
        except RuntimeError:
            a += geom.Buffer(0).Intersection(tg.Buffer(0)).GetArea()
    tat_lyr.SetSpatialFilter(None)
    return a / 1e6


def city_rows(city, a, deso_ds, tat_ds, carry):
    series = {}
    idx = {}
    for s, eras in SERIES.items():
        for e in eras:
            idx[e] = index(city, e)
            if not idx[e]:
                sys.exit(f"[error] {city}: no inference for required era {e} in {cf.INF}/{city}_{e}/")
        opt = [e for e in OPTIONAL.get(s, []) if index(city, e)]
        for e in opt:
            idx[e] = index(city, e)
        series[s] = eras + opt
    eras = [e for s in series for e in series[s]]
    series_of = {e: s for s in series for e in series[s]}
    sheets = sorted(set().union(*[set(idx[e]) for e in eras]))
    boxes = [cf.sheet_box(sh) for sh in sheets]
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    codes, attrs, geoms, mem = deso_layer(deso_ds, a.layer, a.field, bbox, carry)
    n = len(codes) + 1
    print(f"\n{city}: eras {eras}, {len(sheets)} sheets, {len(codes)} DeSOs touch the sheet extent")

    meas = {e: np.zeros(n) for e in eras}
    can = {e: np.zeros(n) for e in eras}
    years = {e: [set() for _ in range(n)] for e in eras}
    common = {s: np.zeros(n) for s in series}
    can_common = {e: np.zeros(n) for e in eras}

    for k, sh in enumerate(sheets, 1):
        box = cf.sheet_box(sh)
        ids = rasterise_ids(mem, box).ravel()
        if not ids.any():
            continue
        tat = cf.tatort(box, RES, tat_ds).ravel()
        m, c = {}, {}
        for e in eras:
            if sh not in idx[e]:
                continue
            yr, can_path, prob_path = idx[e][sh]
            m[e] = measured_frac(prob_path, box).ravel() * tat
            c[e] = cf.warp(can_path, box, RES, "average", gdal.GDT_Float32).ravel() * tat
            meas[e] += np.bincount(ids, weights=m[e], minlength=n)
            can[e] += np.bincount(ids, weights=c[e], minlength=n)
            for i in np.unique(ids[m[e] > 0]):
                years[e][i].add(yr)
        for s, se in series.items():
            if not all(e in m for e in se):
                continue          # a series era is missing on this sheet: no common footprint here
            full = np.logical_and.reduce([m[e] >= 0.999 for e in se])
            common[s] += np.bincount(ids[full], minlength=n)
            for e in se:
                can_common[e] += np.bincount(ids[full], weights=c[e][full], minlength=n)
        print(f"  [{k}/{len(sheets)}] {sh}: eras {sorted(m)}", flush=True)

    tat_lyr = tat_ds.GetLayer(0)
    rows, wide = [], []
    for i in range(1, n):
        if not any(meas[e][i] > 0 for e in eras):
            continue
        code = codes[i - 1]
        dk = geoms[i - 1].GetArea() / 1e6
        tk = tatort_km2(geoms[i - 1], tat_lyr)
        w = dict(desokod=code, **attrs[i - 1], city=city, deso_type=code[4], deso_km2=round(dk, 4),
                 deso_tatort_km2=round(tk, 4), geom=geoms[i - 1])
        for s in series:
            ck = common[s][i] * RES ** 2 / 1e6
            w[f"measured_common_km2_{s}"] = round(ck, 4)
            w[f"coverage_pct_{s}"] = round(100 * ck / tk, 2) if tk > 0 else None
        for e in eras:
            s = series_of[e]
            mk = meas[e][i] * RES ** 2 / 1e6
            r = dict(desokod=code, **attrs[i - 1], city=city, deso_type=code[4], era=e, series=s,
                     flight_years=";".join(str(y) for y in sorted(years[e][i])) or None,
                     licence_status=licence(years[e][i]),
                     measured_km2=round(mk, 4),
                     canopy_pct=round(100 * can[e][i] / meas[e][i], 3) if meas[e][i] > 0 else None,
                     measured_common_km2=w[f"measured_common_km2_{s}"],
                     canopy_pct_common=round(100 * can_common[e][i] / common[s][i], 3) if common[s][i] > 0 else None,
                     coverage_pct=w[f"coverage_pct_{s}"],
                     deso_km2=w["deso_km2"], deso_tatort_km2=w["deso_tatort_km2"])
            rows.append(r)
            for key in ("canopy_pct", "canopy_pct_common", "measured_km2", "flight_years", "licence_status"):
                w[f"{key}_{e}"] = r[key]
        wide.append(w)
    print(f"{city}: {len(wide)} DeSOs with measured area")
    return rows, wide


def write_gpkg(wide, eras, srs, carry):
    ds = gdal.GetDriverByName("GPKG").Create(OUT_GPKG, 0, 0, 0, gdal.GDT_Unknown)
    lyr = ds.CreateLayer("deso_canopy_v19", srs, ogr.wkbMultiPolygon)
    real = ogr.OFTReal
    fields = [("desokod", ogr.OFTString)] + [(k, ogr.OFTString) for k in carry] + [
              ("city", ogr.OFTString), ("deso_type", ogr.OFTString),
              ("deso_km2", real), ("deso_tatort_km2", real)]
    for s in SERIES:
        fields += [(f"measured_common_km2_{s}", real), (f"coverage_pct_{s}", real)]
    for e in eras:
        fields += [(f"canopy_pct_{e}", real), (f"canopy_pct_common_{e}", real), (f"measured_km2_{e}", real),
                   (f"flight_years_{e}", ogr.OFTString), (f"licence_status_{e}", ogr.OFTString)]
    for name, t in fields:
        lyr.CreateField(ogr.FieldDefn(name, t))
    for w in wide:
        f = ogr.Feature(lyr.GetLayerDefn())
        for name, _ in fields:
            if w.get(name) is not None:
                f.SetField(name, w[name])
        f.SetGeometry(ogr.ForceToMultiPolygon(w["geom"]))
        lyr.CreateFeature(f)
    ds = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deso", required=True, help="DeSO 2025 GeoPackage")
    ap.add_argument("--layer", default="DeSO_2025")
    ap.add_argument("--field", default="desokod")
    ap.add_argument("--cities", nargs="+", default=cf.CITIES)
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs in results/deso/")
    a = ap.parse_args()

    if not a.overwrite and any(os.path.exists(p) for p in (OUT_GPKG, OUT_CSV)):
        print(f"[skip] {OUT_GPKG} or {OUT_CSV} exists; nothing done (pass --overwrite to rebuild)")
        sys.exit(0)
    deso_ds = gdal.OpenEx(a.deso, gdal.OF_VECTOR)
    tat_ds = gdal.OpenEx(cf.TAT, gdal.OF_VECTOR)
    carry = carry_fields(deso_ds, a.layer)
    print(f"carrying DeSO fields: {carry or 'none'}")

    rows, wide = [], []
    for city in a.cities:
        r, w = city_rows(city, a, deso_ds, tat_ds, carry)
        rows += r; wide += w
    if not rows:
        sys.exit("[error] no DeSO with measured area")

    eras = [e for s in SERIES for e in SERIES[s] + OPTIONAL.get(s, []) if any(r["era"] == e for r in rows)]
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (OUT_GPKG, OUT_CSV):
        if os.path.exists(p):
            os.remove(p)
    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
    write_gpkg(wide, eras, deso_src(deso_ds, a.layer).GetSpatialRef(), carry)
    print(f"\nwrote {OUT_GPKG} ({len(wide)} DeSOs) and {OUT_CSV} ({len(rows)} rows)")
    for city in a.cities:
        for e in eras:
            sel = [r for r in rows if r["city"] == city and r["era"] == e and r["measured_km2"] > 0]
            if sel:
                mk = sum(r["measured_km2"] for r in sel)
                print(f"  {city:6s} {e:8s} {len(sel):4d} DeSOs, measured {mk:7.1f} km2, "
                      f"licence {sorted({r['licence_status'] for r in sel})}")
