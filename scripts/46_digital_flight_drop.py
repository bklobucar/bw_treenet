#!/usr/bin/env python3
"""46_digital_flight_drop.py -- where does canopy drop between the two digital flights (~2005 -> ~2010)?

Per city (default gtb, sth, malmo), on the sheets that have both eras, at the maps' native 0.5 m:
valid = inside the 2023 tätort AND prob >= 0 in both eras. Cut into 250 m windows (kept if at least
--min-valid of the window is valid) and reports:
  * citywide cover of both flights on the common valid area;
  * the distribution of the per-window change (pp): mean, percentiles, binned counts;
  * the windows dropping more than --drop pp: number, window area and valid area;
  * the --top windows with the largest drop: sheet, centre E/N, cover in both flights, change.
Also prints the flight year(s) per era (from the file names) and the flight date: from the image's
TIFF metadata if present, else from the STAC catalogue (32's results/leafon_scan/flights.csv), labelled.
All windows go to results/digital_flight_drop_windows.csv (not replaced without --overwrite).

    python scripts/46_digital_flight_drop.py                       # gtb sth malmo, 2005 vs 2010
gis env (GDAL Python bindings).
"""
import argparse, csv, importlib, os, sys
import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m43 = importlib.import_module("43_export_web_layers")   # STAC (city, year) -> collection; via 41: index(); 39 helpers
m41, cf = m43.m41, m43.cf

gdal.UseExceptions()
RES = 0.5
FLIGHTS = "results/leafon_scan/flights.csv"
OUT_CSV = "results/digital_flight_drop_windows.csv"
EDGES = [-100, -40, -20, -10, -5, 5, 10, 20, 40, 100]


def flight_dates(city, era, idx):
    """Lines describing the flight year(s) and date(s) of one city-era."""
    years = sorted({v[0] for v in idx.values()})
    out = [f"{era}: flight year(s) {', '.join(map(str, years))} ({len(idx)} sheets)"]
    img = next((p for p in (os.path.join(cf.src_dir(city, era), f) for f in sorted(os.listdir(cf.src_dir(city, era))))
                if p.endswith(".tif")), None) if os.path.isdir(cf.src_dir(city, era)) else None
    found = {}
    if img:
        d = gdal.Open(img)
        for dom in [""] + [m for m in (d.GetMetadataDomainList() or []) if m]:
            for k_, v in (d.GetMetadata(dom) or {}).items():
                if any(s in k_.upper() for s in ("DATE", "TIME", "ACQ")):
                    found[f"{dom or 'default'}:{k_}"] = v
    out.append("   image metadata: " + (", ".join(f"{k_}={v}" for k_, v in found.items()) if found
                                        else f"no date ({os.path.basename(img) if img else 'no image'})"))
    if os.path.exists(FLIGHTS):
        rows = list(csv.DictReader(open(FLIGHTS)))
        for y in years:
            coll = m43.STAC.get((city, y))
            hit = [r for r in rows if r["city"] == city and r["collection"] == coll] if coll else []
            if hit:
                r = hit[0]
                out.append(f"   catalogue ({FLIGHTS}): {coll} {r['first_date']}..{r['last_date']}, leaf {r['leaf']}")
            else:
                out.append(f"   catalogue: no entry for {city} {y}" + (f" ({coll})" if coll else ""))
    else:
        out.append(f"   catalogue: {FLIGHTS} not found")
    return out


def city_windows(city, ea, eb, a, tat_ds):
    ia, ib = m41.index(city, ea), m41.index(city, eb)
    sheets = sorted(set(ia) & set(ib))
    print(f"\n== {city}: {ea} vs {eb}, {len(sheets)} sheets with both eras "
          f"({len(ia)} and {len(ib)} sheets in each)")
    for ln in flight_dates(city, ea, ia) + flight_dates(city, eb, ib):
        print("  " + ln)
    if not sheets:
        return []
    k = int(round(a.win / RES))
    tot = dict(valid=0, a=0, b=0)
    wins = []
    for n, sh in enumerate(sheets, 1):
        box = cf.sheet_box(sh)
        valid = cf.tatort(box, RES, tat_ds)
        valid &= cf.warp(ia[sh][2], box, RES, "near", gdal.GDT_Float32) >= 0
        valid &= cf.warp(ib[sh][2], box, RES, "near", gdal.GDT_Float32) >= 0
        ca = (cf.warp(ia[sh][1], box, RES, "near") > 0) & valid
        cb = (cf.warp(ib[sh][1], box, RES, "near") > 0) & valid
        tot["valid"] += int(valid.sum()); tot["a"] += int(ca.sum()); tot["b"] += int(cb.sum())
        ny, nx = (valid.shape[0] // k) * k, (valid.shape[1] // k) * k

        def blk(x):
            return x[:ny, :nx].reshape(ny // k, k, nx // k, k).sum(axis=(1, 3)).astype(np.float64)

        nv, na, nb = blk(valid), blk(ca), blk(cb)
        for i in range(ny // k):
            for j in range(nx // k):
                if nv[i, j] >= a.min_valid * k * k:
                    e = box[0] + (j + 0.5) * a.win
                    nn = box[3] - (i + 0.5) * a.win
                    wins.append(dict(city=city, sheet=sh, centre_e=round(e), centre_n=round(nn),
                                     valid_pct=round(100 * nv[i, j] / (k * k), 1),
                                     year_a=ia[sh][0], year_b=ib[sh][0],
                                     cover_a=round(100 * na[i, j] / nv[i, j], 2),
                                     cover_b=round(100 * nb[i, j] / nv[i, j], 2),
                                     change_pp=round(100 * (nb[i, j] - na[i, j]) / nv[i, j], 2),
                                     valid_km2=nv[i, j] * RES ** 2 / 1e6))
        print(f"    [{n}/{len(sheets)}] {sh}", flush=True)

    V = tot["valid"]
    if not V:
        print("  no common valid area"); return []
    print(f"  citywide on the common area ({V * RES ** 2 / 1e6:.1f} km2): {ea} {100 * tot['a'] / V:.2f} %  "
          f"{eb} {100 * tot['b'] / V:.2f} %  change {100 * (tot['b'] - tot['a']) / V:+.2f} pp")
    if not wins:
        print("  no window passes --min-valid"); return []
    ch = np.array([w["change_pp"] for w in wins])
    print(f"  per-window change, {len(wins)} windows of {a.win:.0f} m (>= {100 * a.min_valid:.0f}% valid): "
          f"mean {ch.mean():+.2f}  median {np.median(ch):+.2f} pp")
    print("    percentiles " + "  ".join(f"p{q} {v:+.1f}" for q, v in zip((1, 5, 10, 25, 75, 90, 95, 99),
                                                                       np.percentile(ch, (1, 5, 10, 25, 75, 90, 95, 99)))))
    h = np.histogram(ch, bins=EDGES)[0]
    for lo, hi, c in zip(EDGES[:-1], EDGES[1:], h):
        rng = f"< {hi:+d}" if lo == EDGES[0] else (f">= {lo:+d}" if hi == EDGES[-1] else f"{lo:+d} .. {hi:+d}")
        print(f"    {rng:>12s} pp  {c:5d}  {100 * c / len(wins):5.1f}%  {'#' * int(round(50 * c / len(wins)))}")
    drop = [w for w in wins if w["change_pp"] < -a.drop]
    print(f"  windows dropping more than {a.drop:.0f} pp: {len(drop)} ({100 * len(drop) / len(wins):.1f}% of windows), "
          f"window area {len(drop) * (a.win / 1000) ** 2:.2f} km2, valid area {sum(w['valid_km2'] for w in drop):.2f} km2")
    top = sorted(wins, key=lambda w: w["change_pp"])[:a.top]
    print(f"  top {len(top)} drops:")
    print(f"    {'sheet':10s} {'centre E':>9s} {'centre N':>9s} {'valid':>6s} {ea + ' %':>9s} {eb + ' %':>9s} {'change':>8s}")
    for w in top:
        print(f"    {w['sheet']:10s} {w['centre_e']:9d} {w['centre_n']:9d} {w['valid_pct']:5.0f}% "
              f"{w['cover_a']:8.1f}  {w['cover_b']:8.1f}  {w['change_pp']:+7.1f}")
    return wins


def main(a):
    if os.path.exists(OUT_CSV) and not a.overwrite:
        sys.exit(f"{OUT_CSV} exists; pass --overwrite to replace it")
    tat_ds = gdal.OpenEx(cf.TAT, gdal.OF_VECTOR)
    rows = []
    for city in a.cities:
        rows += city_windows(city, a.era_a, a.era_b, a, tat_ds)
    if rows:
        for r in rows:
            r["valid_km2"] = round(r["valid_km2"], 5)
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {OUT_CSV} ({len(rows)} windows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="+", default=["gtb", "sth", "malmo"])
    ap.add_argument("--era-a", default="2005")
    ap.add_argument("--era-b", default="2010")
    ap.add_argument("--win", type=float, default=250.0, help="window size in m")
    ap.add_argument("--min-valid", type=float, default=0.5, help="minimum valid share of a window")
    ap.add_argument("--drop", type=float, default=20.0, help="report windows dropping more than this (pp)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
