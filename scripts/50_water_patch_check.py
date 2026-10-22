#!/usr/bin/env python3
"""50_water_patch_check.py -- do failed inference patches contain large water areas?

Hypothesis: water is masked in the output (prob = -1) but not in the input, so dark water enters the
per-patch GroupNorm/LayerNorm statistics and a patch with much water fails on its land.

1. For every production patch (PATCH 1000, OVERLAP 200, stride 800, last patch snapped) of the --cities x
   --eras prob maps: the water fraction of the FULL 1000 px patch (19's output water mask: hydrography
   polygons, all_touched) and the mean prob of its core (the 600 px inside the taper, prob >= 0 only),
   against the mean core prob of its 4 neighbours. A core FAILS if it is >= 50% valid, its neighbours
   average >= 0.2 (so there is canopy around) and its own mean is < 0.25 x the neighbours'.
   Failure rate (fails / cores with canopy around) by water band 0, 0-5, 5-15, 15-30, >30 %.
2. The 46 drop windows (results/digital_flight_drop_windows.csv): water fraction of the era-b production
   patch whose core holds the window centre; drop rate (change < -20 pp) by the same bands.
The hydrography files do not cover every sheet (e.g. Stockholm's polygons stop at N 6575.4 km), and where
they are missing 19 masked no water either. Patches outside the extent of the city's water polygons are
flagged (in_hydro = False) and left out of the tables, since their 0 % water is unknown.
Cores go to results/water_patch_check.csv (not replaced without --overwrite).

    python scripts/50_water_patch_check.py
bwtreenet env (rasterio + geopandas).
"""
import argparse, csv, glob, importlib, os, sys
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m49 = importlib.import_module("49_norm_fix_test")     # water_mask (19's hydrography mask)
m47, p19 = m49.m47, m49.p19                            # core_of; positions, SOURCES, OUT_ROOT

P, O = 1000, 200
S = P - O
MIN_VALID, NEIGH_MIN, RATIO = 0.5, 0.2, 0.25
BANDS = [("0", 0, 0), ("0-5", 0, 5), ("5-15", 5, 15), ("15-30", 15, 30), (">30", 30, 101)]
DROPS = "results/digital_flight_drop_windows.csv"
OUT = "results/water_patch_check.csv"


def band(w):
    if w == 0:
        return "0"
    for name, lo, hi in BANDS[1:]:
        if lo < w <= hi:
            return name
    return ">30"


def frame_patches(city, era, img_path):
    """-> (records, transform, y positions, x positions) for one frame, or None without a prob map."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    prob_p = os.path.join(p19.OUT_ROOT, f"{city}_{era}", f"{stem}_prob.tif")
    if not os.path.exists(prob_p):
        return None
    with rasterio.open(prob_p) as s:
        prob, tr, crs = s.read(1), s.transform, s.crs
    water = m49.water_mask(city, prob.shape, tr, crs)
    if water is None:
        water = np.zeros(prob.shape, bool)
    hb = m49._HYDRO[city].total_bounds if city in m49._HYDRO else None
    H, W = prob.shape
    py, px = p19.positions(H, P, S), p19.positions(W, P, S)
    rec = {}
    for y in py:
        for x in px:
            core = prob[y + O:y + P - O, x + O:x + P - O]
            ok = core >= 0
            x0, x1 = tr.c + x * tr.a, tr.c + (x + P) * tr.a
            y1, y0 = tr.f - y * abs(tr.e), tr.f - (y + P) * abs(tr.e)
            inside = hb is not None and hb[0] <= x0 and x1 <= hb[2] and hb[1] <= y0 and y1 <= hb[3]
            rec[(y, x)] = dict(city=city, era=era, frame=stem, y=y, x=x,
                               centre_e=round(tr.c + (x + P / 2) * tr.a), centre_n=round(tr.f - (y + P / 2) * abs(tr.e)),
                               in_hydro=bool(inside),
                               water_pct=round(100 * float(water[y:y + P, x:x + P].mean()), 2),
                               valid=round(float(ok.mean()), 3),
                               core_mean=float(core[ok].mean()) if ok.any() else float("nan"))
    yi, xi = {y: n for n, y in enumerate(py)}, {x: n for n, x in enumerate(px)}
    for (y, x), r in rec.items():
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j, i = yi[y] + dy, xi[x] + dx
            if 0 <= j < len(py) and 0 <= i < len(px):
                q = rec[(py[j], px[i])]
                if q["valid"] >= MIN_VALID:
                    nb.append(q["core_mean"])
        r["neigh_mean"] = float(np.mean(nb)) if nb else float("nan")
        r["eligible"] = r["valid"] >= MIN_VALID and r["neigh_mean"] >= NEIGH_MIN
        r["fail"] = bool(r["eligible"] and r["core_mean"] < RATIO * r["neigh_mean"])
    return list(rec.values()), tr, py, px


def band_table(recs, title):
    out = [r for r in recs if not r["in_hydro"]]
    recs = [r for r in recs if r["in_hydro"]]
    el = [r for r in recs if r["eligible"]]
    print(f"  {title}: {len(recs)} patches inside the hydrography extent ({len(out)} outside, left out), "
          f"{len(el)} with canopy around (neighbours >= {NEIGH_MIN}), {sum(r['fail'] for r in el)} fail"
          + (f"; outside the extent {sum(r['fail'] for r in out)} of {sum(r['eligible'] for r in out)} fail" if out else ""))
    print(f"    {'water %':>8s} {'patches':>8s} {'eligible':>9s} {'fail':>6s} {'rate':>7s} {'median core/neigh':>18s}")
    for name, _, _ in BANDS:
        b = [r for r in recs if band(r["water_pct"]) == name]
        e = [r for r in b if r["eligible"]]
        nf = sum(r["fail"] for r in e)
        rat = np.median([r["core_mean"] / r["neigh_mean"] for r in e]) if e else float("nan")
        print(f"    {name:>8s} {len(b):8d} {len(e):9d} {nf:6d} {100 * nf / max(len(e), 1):6.1f}% {rat:18.2f}")


def main(a):
    if os.path.exists(OUT) and not a.overwrite:
        sys.exit(f"{OUT} exists; pass --overwrite to replace it")
    allrec, cache = [], {}
    print("== 1. production patches: failure rate by water fraction of the full 1000 px patch")
    for city in a.cities:
        for era in a.eras:
            if (city, era) not in p19.SOURCES:
                continue
            rel, _ = p19.SOURCES[(city, era)]
            recs = []
            for f in sorted(glob.glob(os.path.join(p19.HERE, rel, "*.tif"))):
                fp = frame_patches(city, era, f)
                if fp is None:
                    continue
                recs += fp[0]
                sheet = "_".join(os.path.basename(f).split("_")[:3])
                cache[(city, era, sheet)] = ({(r["y"], r["x"]): r for r in fp[0]}, fp[1], fp[2], fp[3])
            if recs:
                band_table(recs, f"{city} {era}")
                allrec += recs
            else:
                print(f"  {city} {era}: no prob maps")
    for city in a.cities:
        cr = [r for r in allrec if r["city"] == city]
        if cr:
            band_table(cr, f"{city}, {' + '.join(a.eras)} pooled")

    print(f"\n== 2. drop windows ({DROPS}): drop rate (< -{a.drop:.0f} pp) by water fraction of the {a.era_b} patch")
    if not os.path.exists(DROPS):
        print(f"  {DROPS} not found; run 46 first")
    else:
        wins = list(csv.DictReader(open(DROPS)))
        for city in sorted({w["city"] for w in wins}):
            rows = []
            for w in (w for w in wins if w["city"] == city):
                key = (city, a.era_b, w["sheet"])
                if key not in cache:
                    rel, _ = p19.SOURCES.get((city, a.era_b), (None, None))
                    f = sorted(glob.glob(os.path.join(p19.HERE, rel, f"{w['sheet']}_*.tif"))) if rel else []
                    fp = frame_patches(city, a.era_b, f[0]) if f else None
                    cache[key] = ({(r["y"], r["x"]): r for r in fp[0]}, fp[1], fp[2], fp[3]) if fp else None
                if cache[key] is None:
                    continue
                recs, tr, py, px = cache[key]
                col = (float(w["centre_e"]) - tr.c) / tr.a
                row = (tr.f - float(w["centre_n"])) / abs(tr.e)
                r = recs[(m47.core_of(py, row), m47.core_of(px, col))]
                rows.append((r["water_pct"], float(w["change_pp"]) < -a.drop, r["in_hydro"]))
            if not rows:
                continue
            out = [q for q in rows if not q[2]]
            rows = [(wp, d) for wp, d, ins in rows if ins]
            nd = sum(d for _, d in rows)
            print(f"  {city}: {len(rows)} windows inside the hydrography extent, {nd} drops"
                  + (f" ({len(out)} windows outside, {sum(q[1] for q in out)} drops, left out)" if out else ""))
            print(f"    {'water %':>8s} {'windows':>8s} {'drops':>6s} {'rate':>7s}")
            for name, _, _ in BANDS:
                b = [d for wp, d in rows if band(wp) == name]
                print(f"    {name:>8s} {len(b):8d} {sum(b):6d} {100 * sum(b) / max(len(b), 1):6.1f}%")

    if allrec:
        with open(OUT, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(allrec[0]), lineterminator="\n"); wr.writeheader(); wr.writerows(allrec)
        print(f"\nwrote {OUT} ({len(allrec)} patches)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="+", default=["gtb", "sth"])
    ap.add_argument("--eras", nargs="+", default=["2005", "2010"])
    ap.add_argument("--era-b", default="2010", help="era of the 46 drop windows' second flight")
    ap.add_argument("--drop", type=float, default=20.0)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
