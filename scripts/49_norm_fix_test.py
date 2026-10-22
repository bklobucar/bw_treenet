#!/usr/bin/env python3
"""49_norm_fix_test.py -- can bigger patches or sheet-level normalisation remove the per-patch norm failure?

47 showed that per-patch GroupNorm/LayerNorm statistics make v19 call a uniform closed stand non-canopy
(gtb 639_31_55: right patch 0.042 -> 0.795 mean prob with the left patch's norms). Three variants:
  a  production: PATCH 1000, OVERLAP 200 (19_production_inference.py)
  b  large patches: PATCH 2000, OVERLAP 400 (same taper blending; v19's SharpConnect LayerNorm affine maps
     are spatially constant, so 47's NormSwap broadcasts them exactly to the larger patch)
  c  fixed norms: PATCH 1000, OVERLAP 200, every GroupNorm/LayerNorm using the statistics pooled over all
     patches of the sheet (two passes; pass 1 is variant a). Pooled mean = mean of patch means; pooled
     var = mean of patch vars + variance of patch means (= the statistic over the whole sheet).
Per test sheet and variant: cover % (prob >= 0.6), mean prob (both inside production's valid area,
prob >= 0), patch cores of the production grid with mean prob < 0.1 over above-median image texture,
and a seam index: mean |prob(x + 25 m) - prob(x - 25 m)| across the centre lines of the variant's own
patch overlaps vs across its patch-core centre lines (ratio ~1: no seams).
Labelled sets, scored at 0.6 inside production's valid area:
  * 1990s label clips (gtb, malmo; data/raw/clips_for_labelling) on their 1990r05 frames -- these clips
    were in training (see notes), so this checks consistency, not independent accuracy;
  * v19's validation set malmo/tiles_1970 (held out of training; 1000 px tiles);
  * a sample of modern tiles per city (in training). For tiles, b reflect-pads to 2000 px as 19 would, and
    c pools the statistics over all tiles of the set.
PNGs: results/norm_fix_test/639_31_55_<era>.png around the 639_31_55 window (top row image | a | b | c
probability, viridis 0-1, grey = prob < 0; bottom row the binary masks). Table: results/norm_fix_test/summary.csv.

    python scripts/49_norm_fix_test.py
bwtreenet env (torch + rasterio + geopandas), one GPU. Existing outputs are not replaced without --overwrite.
"""
import argparse, csv, glob, importlib, os, sys, time
import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m47 = importlib.import_module("47_seam_check")          # NormSwap, run_patch, read_image, viridis, grey8
p19 = m47.p19

OUT_DIR = "results/norm_fix_test"
THR = p19.THRESHOLD
VARIANTS = {"a": (1000, 200, False), "b": (2000, 400, False), "c": (1000, 200, True)}
SHEETS = ["gtb:2010:639_31_55", "gtb:2005:639_31_55",
          "sth:2010:657_67_00", "sth:2010:658_67_00", "sth:2005:657_67_00", "sth:2005:658_67_00",
          "malmo:2010:616_37_00", "gtb:pan2000:640_30_05"]
WINDOW = ("gtb", "639_31_55", 317875.0, 6395875.0)    # PNG centre
D = 50                                                  # px either side of a line for the seam index (25 m)
LOW, MIN_VALID = 0.1, 0.8


class Pool:
    """Accumulates NormSwap.last over patches -> pooled (mean, var) per layer."""

    def __init__(self):
        self.n, self.s = 0, {}

    def add(self, last):
        self.n += 1
        for k, (m, v) in last.items():
            a = self.s.setdefault(k, [m * 0, m * 0, v * 0])
            a[0] += m; a[1] += m * m; a[2] += v

    def stats(self):
        out = {}
        for k, (sm, sm2, sv) in self.s.items():
            mean = sm / self.n
            out[k] = (mean, (sv / self.n + sm2 / self.n - mean * mean).clamp_min(0))
        return out


def infer(ctx, image, patch, overlap, frozen=None, pool=None):
    """19's infer_frame with a chosen patch size; frozen = fixed norm stats; pool collects per-patch stats."""
    model, dev, torch, sw = ctx
    H, W = image.shape
    pin = image
    ph, pw = max(0, patch - H), max(0, patch - W)
    if ph or pw:
        pin = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
    pH, pW = pin.shape
    win, stride = p19.taper(patch, overlap), patch - overlap
    acc = np.zeros((pH, pW), np.float32)
    wsum = np.zeros((pH, pW), np.float32)
    sw.mode, sw.frozen = ("apply", frozen) if frozen is not None else ("off", {})
    for y in p19.positions(pH, patch, stride):
        for x in p19.positions(pW, patch, stride):
            pr = m47.run_patch(model, torch, dev, pin[y:y + patch, x:x + patch])
            if pool is not None:
                pool.add(sw.last)
            acc[y:y + patch, x:x + patch] += pr * win
            wsum[y:y + patch, x:x + patch] += win
    sw.mode = "off"
    return (acc / np.maximum(wsum, 1e-8))[:H, :W]


def run_variants(ctx, image, variants):
    """-> ({variant: prob}, {variant: seconds}, {variant: peak GPU GB})."""
    torch = ctx[2]
    probs, secs, mem = {}, {}, {}
    pooled = None
    for v in sorted(variants, key=lambda v: v != "a"):        # a first: its pass is c's pass 1
        patch, overlap, fixed = VARIANTS[v]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if fixed:
            if pooled is None:
                pool = Pool(); infer(ctx, image, patch, overlap, pool=pool); pooled = pool.stats()
            probs[v] = infer(ctx, image, patch, overlap, frozen=pooled)
        elif v == "a" and "c" in variants:
            pool = Pool(); probs[v] = infer(ctx, image, patch, overlap, pool=pool); pooled = pool.stats()
        else:
            probs[v] = infer(ctx, image, patch, overlap)
        secs[v], mem[v] = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9
    return probs, secs, mem


def find_frame(city, era, sheet):
    if (city, era) not in p19.SOURCES:
        return None
    rel, is_rgb = p19.SOURCES[(city, era)]
    img = sorted(glob.glob(os.path.join(p19.HERE, rel, f"{sheet}_*.tif")))
    if not img:
        return None
    stem = os.path.splitext(os.path.basename(img[0]))[0]
    return img[0], os.path.join(p19.OUT_ROOT, f"{city}_{era}", f"{stem}_prob.tif"), is_rgb


def texture_cores(image, valid):
    """Production-grid patch cores with >= MIN_VALID valid -> [(slices, above-median texture)]."""
    gx = np.zeros_like(image); gx[:, 1:] = np.abs(np.diff(image, axis=1))
    gy = np.zeros_like(image); gy[1:, :] = np.abs(np.diff(image, axis=0))
    g = gx + gy
    H, W = image.shape
    cores = []
    for y in p19.positions(H, 1000, 800):
        for x in p19.positions(W, 1000, 800):
            cs = (slice(y + 200, y + 800), slice(x + 200, x + 800))
            ok = valid[cs] & (image[cs] > 0)
            if ok.mean() >= MIN_VALID:
                cores.append((cs, ok, float(g[cs][ok].mean())))
    med = np.median([c[2] for c in cores]) if cores else 0
    return [(cs, ok, t > med) for cs, ok, t in cores]


def line_diffs(prob, valid, lines, axis):
    out = []
    n = prob.shape[axis]
    for c in lines:
        c = int(round(c))
        if c - D < 0 or c + D >= n:
            continue
        if axis == 1:
            a, b, ok = prob[:, c - D], prob[:, c + D], valid[:, c - D] & valid[:, c + D]
        else:
            a, b, ok = prob[c - D, :], prob[c + D, :], valid[c - D, :] & valid[c + D, :]
        if ok.any():
            out.append(float(np.abs(a[ok] - b[ok]).mean()))
    return out


def seam_index(prob, valid, patch, overlap):
    seams, cores = [], []
    for axis in (0, 1):
        pos = p19.positions(prob.shape[axis], patch, patch - overlap)
        seams += line_diffs(prob, valid, [(q + p + patch) / 2 for p, q in zip(pos, pos[1:]) if q < p + patch], axis)
        cores += line_diffs(prob, valid, [p + patch / 2 for p in pos], axis)
    s, c = (np.mean(seams) if seams else np.nan), (np.mean(cores) if cores else np.nan)
    return s, c, s / c if c else np.nan


def score(pred, lab, ok):
    p, l = pred[ok], lab[ok]
    return np.array([(p & l).sum(), (p & ~l).sum(), (~p & l).sum(), ok.sum(), l.sum(), p.sum()], np.int64)


def fmt_score(c):
    tp, fp, fn, n, nl, npred = c
    return dict(label_cover=100 * nl / max(n, 1), pred_cover=100 * npred / max(n, 1),
                iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))


def save_png(path, image, valid, probs, variants):
    k, gap = image.shape[0], 6
    cols = 1 + len(variants)
    rgb = np.full((3, 2 * k + gap, cols * k + (cols - 1) * gap), 255, np.uint8)
    rgb[:, :k, :k] = m47.grey8(image)
    for i, v in enumerate(variants, 1):
        x = i * (k + gap)
        pv = m47.viridis(probs[v]); pv[:, ~valid] = 128
        rgb[:, :k, x:x + k] = pv
        mk = np.where(probs[v] >= THR, 255, 0).astype(np.uint8); mk[~valid] = 128
        rgb[:, k + gap:, x:x + k] = mk
    with rasterio.open(path, "w", driver="PNG", width=rgb.shape[2], height=rgb.shape[1], count=3, dtype="uint8") as d:
        d.write(rgb)


def sheets_part(a, ctx, rows):
    print(f"\n== test sheets; variants {a.variants}  (a = production 1000/200, b = 2000/400, c = fixed sheet norms)")
    print(f"   {'sheet':28s} {'v':2s} {'cover %':>8s} {'mean p':>7s} {'low cores':>10s} {'seam':>7s} {'within':>7s} "
          f"{'index':>6s} {'sec':>6s} {'GPU GB':>7s}")
    for spec in a.sheets:
        city, era, sheet = spec.split(":")
        fr = find_frame(city, era, sheet)
        if not fr or not os.path.exists(fr[1]):
            print(f"   {spec}: frame or production prob map missing, skipped"); continue
        img_p, prob_p, is_rgb = fr
        image = m47.read_image(img_p, is_rgb)
        with rasterio.open(prob_p) as s:
            valid, tr = s.read(1) >= 0, s.transform
        probs, secs, mem = run_variants(ctx, image, a.variants)
        cores = texture_cores(image, valid)
        name = f"{city} {era} {os.path.basename(img_p)[:-4]}"
        for v in a.variants:
            p = probs[v]
            pv = p[valid]
            low = sum(1 for cs, ok, tex in cores if tex and p[cs][ok].mean() < LOW)
            s, c, idx = seam_index(p, valid, *VARIANTS[v][:2])
            print(f"   {name:28s} {v:2s} {100 * (pv >= THR).mean():8.2f} {pv.mean():7.3f} {low:5d}/{len(cores):<4d} "
                  f"{s:7.4f} {c:7.4f} {idx:6.2f} {secs[v]:6.0f} {mem[v]:7.1f}")
            rows.append(dict(kind="sheet", name=name, variant=v, cover_pct=round(100 * (pv >= THR).mean(), 3),
                             mean_prob=round(float(pv.mean()), 4), low_cores=low, cores=len(cores),
                             seam=round(s, 5), within=round(c, 5), seam_index=round(idx, 3),
                             seconds=round(secs[v]), gpu_gb=round(mem[v], 1)))
        if (city, sheet) == WINDOW[:2]:
            out = os.path.join(OUT_DIR, f"{sheet}_{era}.png")
            half = int(round(a.png_size / 2 / tr.a))
            cx, cy = int((WINDOW[2] - tr.c) / tr.a), int((tr.f - WINDOW[3]) / abs(tr.e))
            sl = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
            if os.path.exists(out) and not a.overwrite:
                print(f"   {out} exists, not replaced")
            else:
                save_png(out, image[sl], valid[sl], {v: probs[v][sl] for v in a.variants}, a.variants)
                print(f"   wrote {out}  (top: image | " + " | ".join(a.variants) + " probability; bottom: masks)")


def clips_part(a, ctx, rows):
    import geopandas as gpd
    from rasterio.features import rasterize
    print("\n== 1990s label clips (in training: consistency check, not independent accuracy)")
    for city in ("gtb", "malmo"):
        clip = f"data/raw/clips_for_labelling/{city}_1990_clip.tif"
        shp = f"data/raw/clips_for_labelling/labels_{city}_1990.shp"
        if not (os.path.exists(clip) and os.path.exists(shp)):
            print(f"   {city}: clip or labels missing"); continue
        with rasterio.open(clip) as s:
            cb = s.bounds
        rel, is_rgb = p19.SOURCES[(city, "1990r05")]
        labels = gpd.read_file(shp)
        conf = {v: np.zeros(6, np.int64) for v in a.variants}
        for f in sorted(glob.glob(os.path.join(p19.HERE, rel, "*.tif"))):
            with rasterio.open(f) as s:
                fb, tr, crs = s.bounds, s.transform, s.crs
            ib = (max(cb.left, fb.left), max(cb.bottom, fb.bottom), min(cb.right, fb.right), min(cb.top, fb.top))
            if ib[0] >= ib[2] or ib[1] >= ib[3]:
                continue
            stem = os.path.splitext(os.path.basename(f))[0]
            prob_p = os.path.join(p19.OUT_ROOT, f"{city}_1990r05", f"{stem}_prob.tif")
            if not os.path.exists(prob_p):
                print(f"   {city}: no 1990r05 prob map for {stem}"); continue
            image = m47.read_image(f, is_rgb)
            with rasterio.open(prob_p) as s:
                valid = s.read(1) >= 0
            probs, _, _ = run_variants(ctx, image, a.variants)
            w = from_bounds(*ib, transform=tr).round_offsets().round_lengths()
            sl = (slice(w.row_off, w.row_off + w.height), slice(w.col_off, w.col_off + w.width))
            wtr = rasterio.windows.transform(w, tr)
            lab_g = labels.to_crs(crs) if labels.crs and labels.crs != crs else labels
            lab = rasterize([(g, 1) for g in lab_g.geometry if g is not None and not g.is_empty],
                            out_shape=(w.height, w.width), transform=wtr, fill=0, dtype="uint8") > 0
            ok = valid[sl] & (image[sl] > 0)
            for v in a.variants:
                conf[v] += score(probs[v][sl] >= THR, lab, ok)
            print(f"   {city}: {stem} overlaps the clip ({ok.sum() * tr.a ** 2 / 1e6:.3f} km2 scored)")
        for v in a.variants:
            if conf[v][3]:
                sc = fmt_score(conf[v]); print_score(f"{city} 1990 clip", v, sc, rows)


def print_score(name, v, sc, rows):
    print(f"   {name:46s} {v:2s} labels {sc['label_cover']:5.1f}%  predicted {sc['pred_cover']:5.1f}%  "
          f"IoU {sc['iou']:.3f}  precision {sc['precision']:.3f}  recall {sc['recall']:.3f}")
    rows.append(dict(kind="labels", name=name, variant=v, **{k: round(float(x), 4) for k, x in sc.items()}))


def tiles_part(a, ctx, rows):
    sets = [("malmo/tiles_1970", "v19 validation (held out)", 0)] + \
           [(f"{c}/tiles", "modern sample (in training)", a.tiles_n) for c in ("malmo", "gtb", "sth")]
    print("\n== labelled tiles (b: reflect-padded to 2000 px as 19 would; c: norms pooled over the set)")
    for rel, what, n in sets:
        d = os.path.join(p19.HERE, "data/processed", rel)
        names = sorted(os.listdir(os.path.join(d, "labels"))) if os.path.isdir(os.path.join(d, "labels")) else []
        names = [x for x in names if x.lower().endswith(".tif")]
        if n and len(names) > n:
            names = names[::len(names) // n][:n]
        if not names:
            print(f"   {rel}: no tiles"); continue
        imgs, labs = [], []
        for x in names:
            with rasterio.open(os.path.join(d, "images", x)) as s:
                imgs.append(s.read(1).astype(np.float32))
            with rasterio.open(os.path.join(d, "labels", x)) as s:
                labs.append(s.read(1) > 0)
        conf = {v: np.zeros(6, np.int64) for v in a.variants}
        pooled, pool = None, Pool()
        preds = {v: [] for v in a.variants}
        for im in imgs:                                     # a (and c's pass 1)
            if "a" in a.variants or "c" in a.variants:
                pa = infer(ctx, im, 1000, 200, pool=pool)
                if "a" in a.variants:
                    preds["a"].append(pa)
            if "b" in a.variants:
                preds["b"].append(infer(ctx, im, 2000, 400))
        if "c" in a.variants:
            pooled = pool.stats()
            preds["c"] = [infer(ctx, im, 1000, 200, frozen=pooled) for im in imgs]
        for v in a.variants:
            for p, l in zip(preds[v], labs):
                conf[v] += score(p >= THR, l, np.ones_like(l))
            print_score(f"{rel} ({len(names)}, {what})", v, fmt_score(conf[v]), rows)


def main(a):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "summary.csv")
    if os.path.exists(out) and not a.overwrite:
        sys.exit(f"{out} exists; pass --overwrite to replace it")
    model, dev, torch = p19.load_model()
    torch.backends.cudnn.allow_tf32 = False          # see 47: exact norm replay needs full fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    sw = m47.NormSwap(model, torch)
    if dev.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    ctx = (model, dev, torch, sw)
    rows = []
    sheets_part(a, ctx, rows)
    if not a.skip_labels:
        clips_part(a, ctx, rows)
        tiles_part(a, ctx, rows)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["a", "b", "c"], choices=list(VARIANTS))
    ap.add_argument("--sheets", nargs="+", default=SHEETS, help="city:era:sheet")
    ap.add_argument("--tiles-n", type=int, default=20, help="modern tiles sampled per city")
    ap.add_argument("--png-size", type=float, default=500.0, help="PNG extent around the 639_31_55 window (m)")
    ap.add_argument("--skip-labels", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
