#!/usr/bin/env python3
"""51_remask_water.py -- apply the hydro_v2 water mask to the existing v19 maps, without re-running inference.

For every results/inference_v19/<city>_<era>/ (all eras unless --eras is given), per frame: rasterise 19's
HYDRO water (now data/hydro_v2, Fastighetskartan) on the frame grid exactly as 19 does (all_touched), and
where prob >= 0 and water: prob = -1 and canopy = 0, production's convention for masked pixels.
Re-masked maps go to results/inference_v19_hydro2/<city>_<era>/ with the same names and profiles; the
originals are not touched. Existing outputs are kept (a rerun resumes) unless --overwrite.

Per city and era it reports: valid area before and after, the valid area removed (km2), canopy on the
removed water (km2), cover % before, after and the change (pp). Also "old-water only": pixels production
masked inside the tätort on valid imagery that are not water in hydro_v2 -- these stay masked, since 19
overwrote their probability; unmasking them needs a new inference run.
Summary: results/remask_hydro2_summary.csv (rewritten each run).

    python scripts/51_remask_water.py --workers 8
bwtreenet env (rasterio + geopandas).
"""
import argparse, csv, glob, importlib, os, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
p19 = importlib.import_module("19_production_inference")    # HYDRO, TATORT, SOURCES, OUT_ROOT, rgb_to_gray

OUT_ROOT = os.path.join(p19.HERE, "results", "inference_v19_hydro2")
SUMMARY = os.path.join(p19.HERE, "results", "remask_hydro2_summary.csv")
G = {}                                    # vector layers, loaded before the workers fork


def burn(gdf, shape, transform):
    w, s, e, n = array_bounds(shape[0], shape[1], transform)
    sub = gdf.cx[w:e, s:n]
    shp = [(g, 1) for g in sub.geometry if g is not None and not g.is_empty]
    if not shp:
        return np.zeros(shape, bool)
    return rasterize(shp, out_shape=shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True) > 0


def frame(task):
    city, era, prob_p, overwrite = task
    stem = os.path.basename(prob_p)[:-len("_prob.tif")]
    can_p = os.path.join(os.path.dirname(prob_p), f"{stem}_canopy.tif")
    with rasterio.open(prob_p) as s:
        prob, pprof = s.read(1), s.profile
    with rasterio.open(can_p) as s:
        can, cprof = s.read(1), s.profile
    water = burn(G["water"][city], prob.shape, pprof["transform"])
    valid = prob >= 0
    removed = valid & water
    canb = (can > 0) & valid
    r = dict(city=city, era=era, frame=stem, px_km2=abs(pprof["transform"].a * pprof["transform"].e) / 1e6,
             valid_before=int(valid.sum()), valid_after=int((valid & ~water).sum()), removed=int(removed.sum()),
             canopy_before=int(canb.sum()), canopy_after=int((canb & ~water).sum()),
             canopy_removed=int((canb & water).sum()), old_only=-1)

    # old-water-only pixels: masked by production inside the tätort on valid imagery, not water now
    src = p19.SOURCES.get((city, era))
    img_p = os.path.join(p19.HERE, src[0], f"{stem}.tif") if src else None
    if img_p and os.path.exists(img_p) and G["tat"] is not None:
        with rasterio.open(img_p) as s:
            img = p19.rgb_to_gray(s.read([1, 2, 3])) if src[1] else s.read(1)
            nodv = 0 if s.nodata is None else s.nodata
        if img.shape == prob.shape:
            urban = burn(G["tat"], prob.shape, pprof["transform"])
            r["old_only"] = int(((prob < 0) & (img != nodv) & urban & ~water).sum())

    out_dir = os.path.join(OUT_ROOT, f"{city}_{era}")
    outs = (os.path.join(out_dir, f"{stem}_prob.tif"), os.path.join(out_dir, f"{stem}_canopy.tif"))
    if overwrite or not all(os.path.exists(o) for o in outs):
        os.makedirs(out_dir, exist_ok=True)
        prob = prob.copy(); prob[removed] = -1.0
        can = can.copy(); can[removed] = 0
        for path, arr, prof in ((outs[0], prob, pprof), (outs[1], can, cprof)):
            tmp = path + ".part"
            with rasterio.open(tmp, "w", **prof) as d:
                d.write(arr, 1)
            os.replace(tmp, path)
        r["written"] = True
    else:
        r["written"] = False
    return r


def main(a):
    import geopandas as gpd
    dirs = sorted(d for d in glob.glob(os.path.join(p19.OUT_ROOT, "*_*")) if os.path.isdir(d))
    tasks = []
    for d in dirs:
        city, era = os.path.basename(d).split("_", 1)
        if city not in a.cities or (a.eras and era not in a.eras):
            continue
        tasks += [(city, era, p, a.overwrite) for p in sorted(glob.glob(os.path.join(d, "*_prob.tif")))]
    if not tasks:
        sys.exit("[error] no prob maps found")
    G["water"] = {}
    for c in a.cities:
        print(f"[water] {c}: {p19.HYDRO[c]}")
        G["water"][c] = gpd.read_file(p19.HYDRO[c]).to_crs(3006)
    G["tat"] = gpd.read_file(p19.TATORT, layer=p19.TATORT_LAYER).to_crs(3006) if os.path.exists(p19.TATORT) else None
    print(f"[info] {len(tasks)} frames in {len({(t[0], t[1]) for t in tasks})} city-eras -> {OUT_ROOT}")

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for n, r in enumerate(ex.map(frame, tasks, chunksize=1), 1):
            rows.append(r)
            if n % 25 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] {r['city']} {r['era']} {r['frame']}", flush=True)

    print(f"\n{'city':6s} {'era':8s} {'frames':>6s} {'valid km2':>10s} {'removed km2':>12s} {'canopy rm km2':>14s} "
          f"{'cover before':>13s} {'after':>7s} {'change pp':>10s} {'old-water only km2':>19s}")
    summ = []
    for key in sorted({(r["city"], r["era"]) for r in rows}):
        rr = [r for r in rows if (r["city"], r["era"]) == key]
        k2 = lambda f: sum(r[f] * r["px_km2"] for r in rr)
        vb, va = k2("valid_before"), k2("valid_after")
        cb, ca = k2("canopy_before"), k2("canopy_after")
        old = sum(r["old_only"] * r["px_km2"] for r in rr if r["old_only"] >= 0)
        s = dict(city=key[0], era=key[1], frames=len(rr), written=sum(r["written"] for r in rr),
                 valid_before_km2=round(vb, 3), valid_after_km2=round(va, 3), removed_km2=round(vb - va, 3),
                 canopy_removed_km2=round(k2("canopy_removed"), 3),
                 cover_before_pct=round(100 * cb / vb, 3) if vb else None,
                 cover_after_pct=round(100 * ca / va, 3) if va else None,
                 old_water_only_km2=round(old, 3) if any(r["old_only"] >= 0 for r in rr) else None)
        s["cover_change_pp"] = round(s["cover_after_pct"] - s["cover_before_pct"], 3) if vb and va else None
        summ.append(s)
        f = lambda v, spec: "-".rjust(len(format(0.0, spec))) if v is None else format(v, spec)
        print(f"{key[0]:6s} {key[1]:8s} {len(rr):6d} {vb:10.2f} {vb - va:12.2f} {s['canopy_removed_km2']:14.3f} "
              f"{f(s['cover_before_pct'], '12.2f')}% {f(s['cover_after_pct'], '6.2f')}% {f(s['cover_change_pp'], '+10.2f')} "
              f"{f(s['old_water_only_km2'], '19.2f')}")
    with open(SUMMARY, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]), lineterminator="\n"); w.writeheader(); w.writerows(summ)
    print(f"\nwrote {SUMMARY}; maps in {OUT_ROOT} ({sum(r['written'] for r in rows)} frames written this run)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="+", default=["malmo", "gtb", "sth"])
    ap.add_argument("--eras", nargs="*", default=[], help="default: every era directory present")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="rewrite re-masked maps that already exist")
    main(ap.parse_args())
