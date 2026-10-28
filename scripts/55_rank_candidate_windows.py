#!/usr/bin/env python3
"""55_rank_candidate_windows.py -- rank 250 m candidate windows for the empty rows of label_windows.csv.

Windows on 46's grid (--win m squares from each sheet's corner, so centres at sheet origin + (i + 0.5) * win),
from the v19 maps re-masked with hydro_v2 (--inf, 51) and NMD2023 basskikt v2.1 (--nmd). Per window:
  measured area = inside the 2023 tätort and not hydro_v2 water (all_touched, as 19 and 53); a window
                  needs >= --min-measured of its area measured. All shares below are of the measured area.
  valid <era>   = share of the measured area with prob >= 0 in that era (imagery present).
  canopy %      = canopy pixels / valid pixels of the era.
  NMD shares    = forest (111-117, 121-127), open vegetated (4211-4233), built-up (51-53), other (the rest,
                  118/128 temporarily non-forest included); mixed = 53's 'mixed' stratum (51-53 or
                  4211-4233 with 20-60 % forest within 50 m).
Categories (each needs the ranked era valid >= --min-valid, 0.95):
  A  Malmo closed stands on film (C11-C12): forest >= 70 %; rank by 2010 canopy % - 1990r05 canopy %, both
     on the area valid in both eras, descending.                                     -> A.csv
  B  shrub over-calling, digital eras (S01-S08, K04): all --cities, 2005 and 2010 separately; open vegetated
     >= 40 %; rank by the share of the open vegetated pixels called canopy, desc.    -> B_2005.csv, B_2010.csv
  C  Malmo mixed urban on film (K01): era 1990r05; rank by the mixed share, desc.   -> C.csv
Exclusion as 53 (its code): windows within 500 m of a historical training tile, a clips_for_labelling clip
or a label_windows.csv row with coordinates, or within the per-city modern buffer of a modern training tile
(--modern-buffer-city, 53's default gtb=-1,malmo=-1,sth=100: Stockholm + 100 m only) are dropped. Then at most
one window per 1 km x 1 km cell (the best ranked), top --top (15) per list. Per list a PNG strip: rows = windows
(rank, sheet, centre, ranking values); columns = the era's image with predicted canopy outlined in red | the same
image | NMD classes (forest, open vegetated, built-up, other). Outputs in results/window_candidates/ (not replaced
without --overwrite).

First, for every label_windows.csv row with coordinates: the centre inside its sheet, the flight year(s) of the
row's era on that sheet (as 53 finds the imagery: <sheet>_<flight or *>.tif in data/raw/<city>/<era>_OF_gray_*)
and the share of the window covered by the era's mosaic. --check-only stops after that.

    python scripts/55_rank_candidate_windows.py
    python scripts/55_rank_candidate_windows.py --check-only
gis env (GDAL Python bindings).
"""
import argparse, ast, csv, glob, importlib, os, sys
import numpy as np
from osgeo import gdal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
m53 = importlib.import_module("53_build_label_queue")   # exclusion, NMD codes/strata, windows CSV, imagery
m41 = importlib.import_module("41_deso_aggregate")      # index(): sheet -> (flight year, canopy, prob)
cf = m53.cf

gdal.UseExceptions()
RES = m53.RES
OUT = "results/window_candidates"
ERAS = {"malmo": ["1990r05", "2005", "2010"], "gtb": ["2005", "2010"], "sth": ["2005", "2010"]}
NMD_CLASSES = [("forest", m53.FOREST, (34, 110, 34)), ("open", m53.OPEN_VEG, (200, 225, 110)),   # key, codes, colour
               ("built", m53.ARTIFICIAL, (140, 140, 140))]
LEGEND = {"forest": "forest", "open": "open vegetated", "built": "built-up"}
OTHER_RGB = (245, 245, 245)


def from_40(names):
    """FONT, text(), outline() ... from 40 without running it (40 is a script, not importable)."""
    p = os.path.join(HERE, "40_film_digital_windows.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names
                                                                       for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in names)]
    ns = {"np": np}
    exec(compile(ast.Module(keep, type_ignores=[]), p, "exec"), ns)
    return ns


P40 = from_40({"FONT", "SCALE", "LABEL_H", "text", "outline"})


# ── label_windows.csv check ──────────────────────────────────────────────────
def check_csv(path):
    located, _, _ = m53.read_windows(path)
    print(f"== {path}: {len(located)} rows with coordinates")
    vrts = {}
    print(f"  {'id':4s} {'city':5s} {'era':8s} {'sheet':10s} {'E':>7s} {'N':>8s}  centre   flight(s) on sheet    "
          f"CSV flight  mosaic cover")
    for r in located:
        e, n, city, era, sh = float(r["E"]), float(r["N"]), r["city"], r["era"], r["sheet"]
        inside = "-"
        if sh:
            sb = cf.sheet_box(sh)
            inside = "in" if sb[0] <= e < sb[2] and sb[1] <= n < sb[3] else "OUTSIDE"
        flights, cover = [], "-"
        if city and era:
            files = sorted(glob.glob(os.path.join(cf.src_dir(city, era), f"{sh or '*'}_*.tif")))
            flights = sorted({cf.SHEET.search(os.path.basename(f)).group(2) for f in files
                              if cf.SHEET.search(os.path.basename(f))})
            if (city, era) not in vrts:
                vrts[city, era] = m53.era_vrt(city, era)
            vrt = vrts[city, era]
            if vrt is not None:
                img = gdal.Warp("", vrt, format="MEM", outputBounds=r["box"], xRes=RES, yRes=RES,
                                resampleAlg="near", srcNodata=0, dstNodata=0).ReadAsArray()
                img = img if img.ndim == 2 else img.max(0)
                cover = f"{100 * (img > 0).mean():.1f}%"
        fl = ",".join(flights) or "NONE"
        warn = []
        if len(flights) > 1:
            warn.append("MORE THAN ONE FLIGHT")
        if r["flight"] and r["flight"] not in flights:
            warn.append("CSV FLIGHT NOT FOUND")
        if cover not in ("-", "100.0%"):
            warn.append("MOSAIC DOES NOT COVER THE WINDOW")
        if inside == "OUTSIDE":
            warn.append("CENTRE OUTSIDE SHEET")
        print(f"  {r['id']:4s} {city:5s} {era:8s} {sh:10s} {e:7.0f} {n:8.0f}  {inside:7s}  {fl:20s}  "
              f"{r['flight'] or '(empty)':10s}  {cover:>7s}  {'  '.join(warn)}")


# ── window statistics ────────────────────────────────────────────────────────
def city_windows(city, a, tat_ds):
    """-> [window dict] for every window of the city with >= --min-measured measured area."""
    eras = [e for e in ERAS[city] if e in a.eras_needed[city]]
    idx = {e: m41.index(city, e) for e in eras}
    for e in eras:
        multi = [s for s in idx[e] if len(glob.glob(os.path.join(cf.INF, f"{city}_{e}", f"{s}_*_canopy.tif"))) > 1]
        print(f"  {city} {e}: {len(idx[e])} sheets in {cf.INF}/{city}_{e}, flight year(s) "
              f"{','.join(map(str, sorted({v[0] for v in idx[e].values()}))) or '-'}"
              + (f"; {len(multi)} sheets with more than one flight (last used): {multi}" if multi else ""))
    sheets = sorted(set().union(*[set(idx[e]) for e in eras]))
    water_ds = gdal.OpenEx(m53.HYDRO[city], gdal.OF_VECTOR)
    names = [tat_ds.GetLayer(i).GetName() for i in range(tat_ds.GetLayerCount())]
    tat_layers = [m53.TAT_LAYER] if m53.TAT_LAYER in names else None
    f10, nw = int(round(m53.NMD_RES / RES)), int(round(a.win / m53.NMD_RES))   # 0.5 m px per NMD cell; cells per window
    kk = int(round(a.win / RES)) ** 2
    mixed_codes, mixed_rule, _ = m53.STRATA["mixed"]
    out = []
    for n, sh in enumerate(sheets, 1):
        box = cf.sheet_box(sh)
        meas = m53.burn(tat_ds, box, RES, layers=tat_layers)
        if not meas.any():
            continue
        meas &= ~m53.burn(water_ds, box, RES, all_touched=True)
        n10 = meas.shape[0] // f10
        cells = lambda x: x.reshape(n10, f10, n10, f10).sum((1, 3), dtype=np.int64)     # 0.5 m -> 10 m counts
        nwin = n10 // nw
        wins = lambda x: x.reshape(nwin, nw, nwin, nw).sum((1, 3))                      # 10 m -> window sums
        Mc = cells(meas)
        nmd, share = m53.nmd_sheet(a.nmd, box)
        assert nmd.shape == Mc.shape, (nmd.shape, Mc.shape)
        wm = wins(Mc).astype(float)
        cls = {name: np.isin(nmd, codes) for name, codes, _ in NMD_CLASSES}
        ov = cls["open"]
        st = {name: wins(Mc * m) / np.maximum(wm, 1) for name, m in cls.items()}
        st["mixed"] = wins(Mc * (np.isin(nmd, mixed_codes) & mixed_rule(share))) / np.maximum(wm, 1)
        V, C, Vov, Cov, keep = {}, {}, {}, {}, {}
        for e in eras:
            if sh not in idx[e]:
                continue
            _, cpath, ppath = idx[e][sh]
            v = (cf.warp(ppath, box, RES, "near", gdal.GDT_Float32) >= 0) & meas
            c = (cf.warp(cpath, box, RES, "near") > 0) & v
            vc, cc = cells(v), cells(c)
            V[e], C[e], Vov[e], Cov[e] = wins(vc), wins(cc), wins(vc * ov), wins(cc * ov)
            if e in ("1990r05", "2010") and "A" in a.cats and city == "malmo":
                keep[e] = (v, c)
        A = None
        if len(keep) == 2:                       # A: both covers on the area valid in both eras
            both = keep["1990r05"][0] & keep["2010"][0]
            A = wins(cells(both)), wins(cells(keep["1990r05"][1] & both)), wins(cells(keep["2010"][1] & both))
        del keep
        for i in range(nwin):
            for j in range(nwin):
                if wm[i, j] < a.min_measured * kk:
                    continue
                w = dict(city=city, sheet=sh, centre_e=round(box[0] + (j + 0.5) * a.win),
                         centre_n=round(box[3] - (i + 0.5) * a.win), measured_pct=100 * wm[i, j] / kk,
                         **{f"nmd_{k}_pct": 100 * s[i, j] for k, s in st.items()})
                w["nmd_other_pct"] = 100 - sum(w[f"nmd_{k}_pct"] for k, _, _ in NMD_CLASSES)
                for e in V:
                    w[f"valid_{e}_pct"] = 100 * V[e][i, j] / wm[i, j]
                    w[f"canopy_{e}_pct"] = 100 * C[e][i, j] / V[e][i, j] if V[e][i, j] else np.nan
                    w[f"openveg_canopy_{e}_pct"] = 100 * Cov[e][i, j] / Vov[e][i, j] if Vov[e][i, j] else np.nan
                    w[f"flight_{e}"] = idx[e][sh][0]
                if A is not None and A[0][i, j]:
                    w["both_valid_pct"] = 100 * A[0][i, j] / wm[i, j]
                    w["canopy_1990r05_both_pct"] = 100 * A[1][i, j] / A[0][i, j]
                    w["canopy_2010_both_pct"] = 100 * A[2][i, j] / A[0][i, j]
                    w["change_pp"] = w["canopy_2010_both_pct"] - w["canopy_1990r05_both_pct"]
                out.append(w)
        print(f"    [{n}/{len(sheets)}] {sh}", flush=True)
    return out


# ── categories ───────────────────────────────────────────────────────────────
def categories(a):
    mv = 100 * a.min_valid
    cats = {
        "A": dict(era="1990r05", cities=["malmo"], key="change_pp",
                  test=lambda w: "change_pp" in w and w["nmd_forest_pct"] >= 70 and w.get("valid_1990r05_pct", 0) >= mv,
                  cols=["flight_1990r05", "flight_2010", "change_pp", "canopy_1990r05_both_pct", "canopy_2010_both_pct",
                        "both_valid_pct", "valid_1990r05_pct", "valid_2010_pct"],
                  label=lambda w: f"2010-1990R05 {w['change_pp']:+.0f} PP: {w['canopy_1990r05_both_pct']:.0f}% TO "
                                  f"{w['canopy_2010_both_pct']:.0f}%  FOREST {w['nmd_forest_pct']:.0f}%"),
        "C": dict(era="1990r05", cities=["malmo"], key="nmd_mixed_pct",
                  test=lambda w: w.get("valid_1990r05_pct", 0) >= mv,
                  cols=["flight_1990r05", "nmd_mixed_pct", "canopy_1990r05_pct", "valid_1990r05_pct"],
                  label=lambda w: f"MIXED {w['nmd_mixed_pct']:.0f}%  1990R05 CANOPY {w['canopy_1990r05_pct']:.0f}%"),
    }
    for e in ("2005", "2010"):
        cats[f"B_{e}"] = dict(era=e, cities=a.cities, key=f"openveg_canopy_{e}_pct",
                              test=lambda w, e=e: w["nmd_open_pct"] >= 40 and w.get(f"valid_{e}_pct", 0) >= mv
                              and not np.isnan(w.get(f"openveg_canopy_{e}_pct", np.nan)),
                              cols=[f"flight_{e}", f"openveg_canopy_{e}_pct", f"canopy_{e}_pct", f"valid_{e}_pct"],
                              label=lambda w, e=e: f"OPEN VEG CALLED CANOPY {w[f'openveg_canopy_{e}_pct']:.0f}%  "
                                                   f"OPEN VEG {w['nmd_open_pct']:.0f}%  CANOPY {w[f'canopy_{e}_pct']:.0f}%")
    return {k: v for k, v in cats.items() if k.split("_")[0] in a.cats}


def exclusion(a):
    """city -> N x 4 boxes; a window is dropped if it overlaps one (53's rules and code)."""
    print("== exclusion (as 53)")
    hist, clips, modern, _ = m53.train_boxes()
    located, _, _ = m53.read_windows(a.csv)
    wins = [r["box"] for r in located]
    grow = lambda bs, d: np.array([(b[0] - d, b[1] - d, b[2] + d, b[3] + d) for b in bs]).reshape(-1, 4)
    base = grow(hist + clips + wins, m53.HIST_BUFFER)
    print(f"  historical tiles {len(hist)}, clips {len(clips)}, CSV windows {len(wins)}: + {m53.HIST_BUFFER:g} m")
    out = {}
    for c in a.cities:
        d = a.modern_buffer_city.get(c, -1.0)
        print(f"  {c}: modern training tiles ({len(modern)}) " + (f"+ {d:g} m" if d >= 0 else "not excluded"))
        out[c] = np.vstack([base] + ([grow(modern, d)] if d >= 0 else []))
    return out


def pick(ws, cat, excl, a):
    """Ranked, excluded, one per 1 km cell, top --top."""
    cand = sorted((w for w in ws if w["city"] in cat["cities"] and cat["test"](w)), key=lambda w: -w[cat["key"]])
    kept, cells, n_ex, n_cell = [], set(), 0, 0
    h = a.win / 2
    for w in cand:
        b = excl[w["city"]]
        x0, y0, x1, y1 = w["centre_e"] - h, w["centre_n"] - h, w["centre_e"] + h, w["centre_n"] + h
        if len(b) and ((b[:, 0] < x1) & (b[:, 2] > x0) & (b[:, 1] < y1) & (b[:, 3] > y0)).any():
            n_ex += 1; continue
        cell = (w["city"], w["centre_e"] // 1000, w["centre_n"] // 1000)
        if cell in cells:
            n_cell += 1; continue
        cells.add(cell)
        kept.append(w)
        if len(kept) == a.top:
            break
    return kept, len(cand), n_ex, n_cell


# ── outputs ──────────────────────────────────────────────────────────────────
def write_csv(path, kept, cat):
    cols = (["rank", "city", "sheet", "centre_e", "centre_n", "era"] + cat["cols"] +
            [c for c in ["measured_pct", "nmd_forest_pct", "nmd_open_pct", "nmd_built_pct", "nmd_other_pct",
                         "nmd_mixed_pct"] if c not in cat["cols"]])
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(cols)
        for r, w in enumerate(kept, 1):
            w = dict(w, rank=r, era=cat["era"])
            wr.writerow([f"{w[c]:.1f}" if isinstance(w.get(c), float) else w.get(c, "") for c in cols])


def strip(path, kept, cat, a, vrts):
    k, gap = int(round(a.win / RES)), 6
    LH = P40["LABEL_H"]
    heads = [f"{cat['era']} image + canopy red", f"{cat['era']} image", "NMD2023"]
    H = 2 * LH + len(kept) * (LH + k + gap)
    W = 3 * k + 2 * gap
    rgb = np.full((3, H, W), 255, np.uint8)
    for c, h in enumerate(heads):
        P40["text"](rgb, h, 4, c * (k + gap) + 2)
    x = 2                                                       # legend line
    for name, rgb_c in [(LEGEND[n], c) for n, _, c in NMD_CLASSES] + [("other", OTHER_RGB)]:
        rgb[:, LH + 4:LH + 18, x:x + 14] = np.array(rgb_c, np.uint8).reshape(3, 1, 1)
        rgb[:, LH + 4:LH + 18, x] = rgb[:, LH + 4:LH + 18, x + 13] = 0
        P40["text"](rgb, name, LH + 4, x + 20)
        x += 20 + 12 * (len(name) + 2)
    for r, w in enumerate(kept):
        e, n, h = w["centre_e"], w["centre_n"], a.win / 2
        wb = (e - h, n - h, e + h, n + h)
        y = 2 * LH + r * (LH + k + gap)
        P40["text"](rgb, f"{r + 1}. {w['city']} {w['sheet']} E{e} N{n}  {cat['label'](w)}", y + 4, 2)
        y += LH
        tile = np.full((3, k, k), 60, np.uint8)
        vrt = vrts.get((w["city"], cat["era"]))
        if vrt is not None:
            im = gdal.Warp("", vrt, format="MEM", outputBounds=wb, xRes=RES, yRes=RES, resampleAlg="bilinear",
                           srcNodata=0, dstNodata=0, outputType=gdal.GDT_Float32).ReadAsArray()
            im = (im if im.ndim == 2 else im[:3].mean(0))[:k, :k]
            v = im[im > 0]
            lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 255)
            tile[:] = np.clip((im - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        plain = tile.copy()
        cp = m41.index(w["city"], cat["era"]).get(w["sheet"])
        if cp:
            ol = P40["outline"]((cf.warp(cp[1], wb, RES, "near") > 0)[:k, :k])
            tile[0][ol], tile[1][ol], tile[2][ol] = 255, 40, 40
        nm = cf.warp(a.nmd, wb, RES, "near", gdal.GDT_UInt16)[:k, :k]
        cls = np.empty((3, k, k), np.uint8)
        cls[:] = np.array(OTHER_RGB, np.uint8).reshape(3, 1, 1)
        for _, codes, rgb_c in NMD_CLASSES:
            m = np.isin(nm, codes)
            for b in range(3):
                cls[b][m] = rgb_c[b]
        for c, t in enumerate((tile, plain, cls)):
            rgb[:, y:y + k, c * (k + gap):c * (k + gap) + k] = t
    mem = gdal.GetDriverByName("MEM").Create("", W, H, 3, gdal.GDT_Byte)
    for b in range(3):
        mem.GetRasterBand(b + 1).WriteArray(rgb[b])
    gdal.GetDriverByName("PNG").CreateCopy(path, mem)


def main(a):
    cf.INF = a.inf
    check_csv(a.csv)
    if a.check_only:
        return
    if os.path.isdir(OUT) and os.listdir(OUT) and not a.overwrite:
        sys.exit(f"{OUT}/ is not empty; pass --overwrite to replace its files")
    for p in (a.nmd, cf.TAT) + tuple(m53.HYDRO[c] for c in a.cities):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
    cats = categories(a)
    a.eras_needed = {c: {cat["era"] for cat in cats.values() if c in cat["cities"]} | ({"2010"} if "A" in cats and c == "malmo" else set())
                     for c in a.cities}
    excl = exclusion(a)
    tat_ds = gdal.OpenEx(cf.TAT, gdal.OF_VECTOR)
    print(f"== windows ({a.win:g} m, >= {100 * a.min_measured:.0f}% measured), maps from {cf.INF}/")
    ws = []
    for city in a.cities:
        if a.eras_needed[city]:
            ws += city_windows(city, a, tat_ds)
    os.makedirs(OUT, exist_ok=True)
    vrts = {}
    for name, cat in cats.items():
        kept, n_c, n_ex, n_cell = pick(ws, cat, excl, a)
        print(f"\n== {name}: {n_c} windows pass the filters, {n_ex} dropped by the exclusion, {n_cell} by the 1 km "
              f"rule; {len(kept)} kept (ranked by {cat['key']})")
        for r, w in enumerate(kept, 1):
            print(f"  {r:2d}. {w['city']:5s} {w['sheet']} E{w['centre_e']} N{w['centre_n']}  {cat['label'](w)}"
                  f"  flight {w.get('flight_' + cat['era'], '-')}")
        write_csv(os.path.join(OUT, f"{name}.csv"), kept, cat)
        for c in {w["city"] for w in kept}:
            if (c, cat["era"]) not in vrts:
                vrts[c, cat["era"]] = m53.era_vrt(c, cat["era"])
        strip(os.path.join(OUT, f"{name}.png"), kept, cat, a, vrts)
        print(f"  wrote {OUT}/{name}.csv and {name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="label_windows.csv")
    ap.add_argument("--inf", default="results/inference_v19_hydro2", help="inference root (51's hydro_v2 maps)")
    ap.add_argument("--nmd", default=m53.NMD)
    ap.add_argument("--cities", nargs="+", default=list(cf.CITIES), choices=cf.CITIES)
    ap.add_argument("--cats", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--win", type=float, default=250.0, help="window size in m (46's grid)")
    ap.add_argument("--min-measured", type=float, default=0.75, help="minimum tätort, non-water share of a window")
    ap.add_argument("--min-valid", type=float, default=0.95, help="minimum valid share of the ranked era")
    ap.add_argument("--modern-buffer-city", type=m53.city_buffers, default=dict(m53.MODERN_CITY_DEFAULT),
                    metavar="CITY=M,...", help="as 53 (default gtb=-1,malmo=-1,sth=100; -1 = not excluded)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--check-only", action="store_true", help="only the label_windows.csv check")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
