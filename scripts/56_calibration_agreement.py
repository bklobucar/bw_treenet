#!/usr/bin/env python3
"""56_calibration_agreement.py -- agreement between two annotators on the calibration set (00_calibration).

Runs locally (GDAL/OGR + numpy; no cluster). --a and --b are two copies of 00_calibration, one per annotator.

Windows (--windows, K01-K04): canopy and ignore from each <cal>/<K>/labels.gpkg are rasterised at 0.5 m on the
window extent (the extent layer, not the 25 m image margin; pixel centres inside a polygon). Pixels in either
annotator's ignore are excluded. Per window: IoU of the two canopy masks on the remaining pixels, canopy % per
annotator (of the remaining pixels), % excluded by ignore. Targets: IoU >= 0.80 for K03 and K04 (digital),
>= 0.70 for K01 and K02 (film). disagreement_<K>.png: the window image (A's image.tif, 2-98 % stretch) with
A-only canopy red, B-only blue, both green, excluded (ignore) grey.

Points (K05_points/points.gpkg, layer points): joined on pid. Agreement on label 1/0 and Cohen's kappa over the
points both annotators labelled 1 or 0; label 9 (cannot tell) and empty labels are counted per annotator and
excluded. Target agreement >= 90 %. Every point whose labels differ is listed (also 9 or empty against a label).
With --key (anna_labelling_key/calibration.csv: pid -> city, era, stratum) agreement is broken down by era and
stratum.

Outputs in --out: agreement_report.md, disagreement_K0x.png, disagreements_K05.csv.
    python scripts/56_calibration_agreement.py --a anna/00_calibration --b blaz/00_calibration \\
        --name-a Anna --name-b Blaz --out results/calibration_agreement [--key anna_labelling_key/calibration.csv]
"""
import argparse, csv, os
import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()
RES = 0.5
WINDOWS = ["K01", "K02", "K03", "K04"]
TARGET_IOU = {"K01": 0.70, "K02": 0.70, "K03": 0.80, "K04": 0.80}     # film K01/K02, digital K03/K04
TARGET_AGREE = 90.0
POINTS = "K05_points"
SRS = osr.SpatialReference()
SRS.ImportFromEPSG(3006)
# overlay colours (RGB) and opacity
COL = {"a": (230, 30, 30), "b": (30, 90, 240), "both": (20, 200, 60), "ignore": (150, 150, 150)}
ALPHA = 0.5


# ── windows ──────────────────────────────────────────────────────────────────
def extent(gpkg):
    ds = ogr.Open(gpkg)
    x0, x1, y0, y1 = ds.GetLayerByName("extent").GetExtent()
    return x0, y0, x1, y1


def burn(gpkg, layer, box):
    nx, ny = int(round((box[2] - box[0]) / RES)), int(round((box[3] - box[1]) / RES))
    m = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    m.SetGeoTransform((box[0], RES, 0, box[3], 0, -RES))
    m.SetProjection(SRS.ExportToWkt())
    src = gdal.OpenEx(gpkg, gdal.OF_VECTOR)
    if src.GetLayerByName(layer) is not None:
        gdal.Rasterize(m, src, layers=[layer], burnValues=[1])
    return m.ReadAsArray().astype(bool)


def image(path, box):
    """Window image at 0.5 m on box -> uint8 array stretched 2-98 %, or a grey field if missing."""
    nx, ny = int(round((box[2] - box[0]) / RES)), int(round((box[3] - box[1]) / RES))
    if not os.path.exists(path):
        return np.full((ny, nx), 200, np.uint8)
    a = gdal.Warp("", path, format="MEM", outputBounds=box, xRes=RES, yRes=RES, resampleAlg="near").ReadAsArray()
    a = a.astype(np.float32).mean(0) if a.ndim == 3 else a.astype(np.float32)
    v = a[a > 0]
    lo, hi = np.percentile(v, (2, 98)) if v.size else (0.0, 255.0)
    return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def write_png(path, gray, a, b, excl):
    rgb = np.repeat(gray[None].astype(np.float32), 3, 0)
    for mask, key in ((excl, "ignore"), (a & b & ~excl, "both"), (a & ~b & ~excl, "a"), (b & ~a & ~excl, "b")):
        for i in range(3):
            rgb[i][mask] = (1 - ALPHA) * rgb[i][mask] + ALPHA * COL[key][i]
    m = gdal.GetDriverByName("MEM").Create("", gray.shape[1], gray.shape[0], 3, gdal.GDT_Byte)
    for i in range(3):
        m.GetRasterBand(i + 1).WriteArray(rgb[i].round().astype(np.uint8))
    gdal.GetDriverByName("PNG").CreateCopy(path, m)
    aux = path + ".aux.xml"
    if os.path.exists(aux):
        os.remove(aux)


def window(k, a, out):
    ga, gb = (os.path.join(c, k, "labels.gpkg") for c in (a.a, a.b))
    missing = [p for p in (ga, gb) if not os.path.exists(p)]
    if missing:
        return dict(id=k, note="missing " + ", ".join(missing))
    box = extent(ga)
    note = "" if np.allclose(box, extent(gb), atol=0.01) else f"extents differ, {a.name_a}'s used"
    ca, cb = burn(ga, "canopy", box), burn(gb, "canopy", box)
    ia, ib = burn(ga, "ignore", box), burn(gb, "ignore", box)
    excl = ia | ib
    ok = ~excl
    inter, union = (ca & cb & ok).sum(), ((ca | cb) & ok).sum()
    iou = inter / union if union else float("nan")
    t = TARGET_IOU.get(k)
    write_png(os.path.join(out, f"disagreement_{k}.png"), image(os.path.join(a.a, k, "image.tif"), box), ca, cb, excl)
    n = ok.sum()
    return dict(id=k, iou=iou, target=t, result=("n/a" if t is None or iou != iou else "PASS" if iou >= t else "FAIL"),
                canopy_a=100 * (ca & ok).sum() / n if n else float("nan"),
                canopy_b=100 * (cb & ok).sum() / n if n else float("nan"),
                excl=100 * excl.mean(), ign_a=100 * ia.mean(), ign_b=100 * ib.mean(),
                a_only=100 * (ca & ~cb & ok).sum() / n if n else float("nan"),
                b_only=100 * (cb & ~ca & ok).sum() / n if n else float("nan"), note=note)


# ── points ───────────────────────────────────────────────────────────────────
def read_points(cal):
    p = os.path.join(cal, POINTS, "points.gpkg")
    if not os.path.exists(p):
        return None
    ds = ogr.Open(p)                                    # kept open while the layer is read
    lyr = ds.GetLayerByName("points")
    names = [lyr.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(lyr.GetLayerDefn().GetFieldCount())]
    out = {}
    for f in lyr:
        g = f.GetGeometryRef()
        rec = {k: (f.GetField(k) if k in names and f.IsFieldSetAndNotNull(k) else None)
               for k in ("label", "conf", "issue", "note")}
        rec["xy"] = (g.GetX(), g.GetY()) if g else None
        out[int(f.GetField("pid"))] = rec
    return out


def lab(v):
    """label value -> '1', '0', '9', '' (empty) or the raw value as text."""
    if v is None or str(v).strip() == "":
        return ""
    try:
        return str(int(float(v)))
    except ValueError:
        return str(v).strip()


def kappa(pairs):
    """pairs: [(a, b)] with a, b in {'0','1'} -> (n, agreement %, kappa)"""
    n = len(pairs)
    if not n:
        return 0, float("nan"), float("nan")
    po = sum(x == y for x, y in pairs) / n
    pa1, pb1 = sum(x == "1" for x, _ in pairs) / n, sum(y == "1" for _, y in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return n, 100 * po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def read_key(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["pid"]): r for r in csv.DictReader(fh)}


def f(x, d=1):
    return "n/a" if x is None or x != x else f"{round(x, d) + 0.0:.{d}f}"      # + 0.0: no "-0.000"


def main(a):
    os.makedirs(a.out, exist_ok=True)
    A, B = a.name_a, a.name_b
    L = [f"# Calibration agreement: {A} vs {B}", "", f"- {A}: `{a.a}`", f"- {B}: `{a.b}`", ""]

    # windows
    rows = [window(k, a, a.out) for k in a.windows]
    L += ["## Windows (canopy IoU)", "",
          f"Rasterised at {RES} m on the window extent (not the margin). Pixels in either annotator's ignore are "
          "excluded; canopy % and the only-% are of the remaining pixels. PNG: "
          f"{A}-only red, {B}-only blue, both green, ignore grey.", "",
          f"| window | IoU | target | result | canopy % {A} | canopy % {B} | {A}-only % | {B}-only % | "
          f"excluded % (ignore {A} / {B}) | note |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "iou" not in r:
            L.append(f"| {r['id']} | | | MISSING | | | | | | {r['note']} |")
            print(f"{r['id']}: {r['note']}")
            continue
        L.append(f"| {r['id']} | {f(r['iou'], 3)} | >= {f(r['target'], 2)} | **{r['result']}** | {f(r['canopy_a'])} | "
                 f"{f(r['canopy_b'])} | {f(r['a_only'])} | {f(r['b_only'])} | {f(r['excl'])} ({f(r['ign_a'])} / "
                 f"{f(r['ign_b'])}) | {r['note']} |")
        print(f"{r['id']}: IoU {f(r['iou'], 3)} (target {f(r['target'], 2)}) {r['result']}; canopy {A} "
              f"{f(r['canopy_a'])} % {B} {f(r['canopy_b'])} %; excluded {f(r['excl'])} %")

    # points
    pa, pb = read_points(a.a), read_points(a.b)
    L += ["", f"## {POINTS} (point labels)", ""]
    if pa is None or pb is None:
        L.append(f"{POINTS}/points.gpkg missing for " + " and ".join(n for n, p in ((A, pa), (B, pb)) if p is None))
    else:
        pids = sorted(set(pa) | set(pb))
        only = [p for p in pids if p not in pa or p not in pb]
        moved = [p for p in pids if p in pa and p in pb and pa[p]["xy"] and pb[p]["xy"]
                 and np.hypot(pa[p]["xy"][0] - pb[p]["xy"][0], pa[p]["xy"][1] - pb[p]["xy"][1]) > 0.01]
        both = [p for p in pids if p in pa and p in pb]
        la = {p: lab(pa[p]["label"]) for p in both}
        lb = {p: lab(pb[p]["label"]) for p in both}
        valid = [p for p in both if la[p] in ("0", "1") and lb[p] in ("0", "1")]
        n, agree, kap = kappa([(la[p], lb[p]) for p in valid])
        cnt = lambda d, v: sum(x == v for x in d.values())
        res = "n/a" if not n else "PASS" if agree >= TARGET_AGREE else "FAIL"
        L += [f"{len(both)} points in both copies" + (f"; pid only in one copy: {only}" if only else "")
              + (f"; pid with different positions: {moved}" if moved else "") + ".", "",
              f"| | {A} | {B} |", "|---|---|---|",
              f"| label 9 (cannot tell) | {cnt(la, '9')} | {cnt(lb, '9')} |",
              f"| empty | {cnt(la, '')} | {cnt(lb, '')} |",
              f"| other values | {sum(v not in ('0', '1', '9', '') for v in la.values())} | "
              f"{sum(v not in ('0', '1', '9', '') for v in lb.values())} |", "",
              f"**Agreement on 1/0: {f(agree)} % over {n} points (target >= {TARGET_AGREE:g} %): {res}.** "
              f"Cohen's kappa {f(kap, 3)}. Points with 9, empty or another value in either copy are excluded.", "",
              f"Confusion ({A} rows, {B} columns; all points in both copies):", "",
              f"| {A} \\ {B} | 1 | 0 | 9 | empty |", "|---|---|---|---|---|"]
        cats = ["1", "0", "9", ""]
        for x in cats:
            L.append(f"| {x or 'empty'} | " + " | ".join(str(sum(la[p] == x and lb[p] == y for p in both)) for y in cats)
                     + " |")
        print(f"{POINTS}: agreement {f(agree)} % over {n} points, kappa {f(kap, 3)} -> {res}; 9s {A} {cnt(la, '9')} "
              f"{B} {cnt(lb, '9')}")

        key = read_key(a.key) if a.key else None
        if key is not None:
            missing = [p for p in both if p not in key]
            L += ["", f"By era and stratum (key `{a.key}`" + (f"; pid not in the key: {missing}" if missing else "")
                  + "):", ""]
            for col in ("era", "stratum"):
                L += [f"| {col} | points 1/0 | agreement % | kappa | 9 {A} | 9 {B} |", "|---|---|---|---|---|---|"]
                for v in sorted({key[p][col] for p in both if p in key}):
                    ps = [p for p in both if p in key and key[p][col] == v]
                    nv, ag, kv = kappa([(la[p], lb[p]) for p in ps if p in valid])
                    L.append(f"| {v} | {nv} | {f(ag)} | {f(kv, 3)} | {sum(la[p] == '9' for p in ps)} | "
                             f"{sum(lb[p] == '9' for p in ps)} |")
                L.append("")

        dis = [p for p in both if la[p] != lb[p]]
        cols = ["pid"] + (["city", "era", "stratum"] if key else []) + \
               [f"{k}_{s}" for s in ("a", "b") for k in ("label", "conf", "issue", "note")]
        with open(os.path.join(a.out, f"disagreements_{POINTS.split('_')[0]}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(cols)
            for p in dis:
                w.writerow([p] + ([key.get(p, {}).get(c, "") for c in ("city", "era", "stratum")] if key else [])
                           + [lab(pa[p]["label"]), pa[p]["conf"] or "", pa[p]["issue"] or "", pa[p]["note"] or "",
                              lab(pb[p]["label"]), pb[p]["conf"] or "", pb[p]["issue"] or "", pb[p]["note"] or ""])
        L += ["", f"Disagreeing points ({len(dis)}; labels differ, including 9 or empty "
              f"against a label; details in disagreements_K05.csv): " + (", ".join(map(str, dis)) or "none")]

    p = os.path.join(a.out, "agreement_report.md")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="annotator A's 00_calibration folder")
    ap.add_argument("--b", required=True, help="annotator B's 00_calibration folder")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--key", help="anna_labelling_key/calibration.csv (era/stratum breakdown of K05)")
    ap.add_argument("--windows", nargs="+", default=WINDOWS)
    ap.add_argument("--out", default="results/calibration_agreement")
    main(ap.parse_args())
