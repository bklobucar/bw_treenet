#!/usr/bin/env python3
"""47_seam_check.py -- a prob ramp to ~0 over uniform canopy: inference-tile seam or real?

Production (19_production_inference.py) feeds raw 0-255 grey values to the model; it does no input
normalisation itself. Inside the model (BWTreeNet/GuiTest/model/BWTreeNet.py) the input goes x/255 ->
frozen Luminance Enhancer -> *255 -> 255 - x, and then every block normalises with GroupNorm(8, C) and
SharpConnect with LayerNorm([C, H, W]) -- statistics taken over the whole 1000 px patch. Patches are
1000 px with an 800 px stride, blended with a sin^2 taper over 200 px (100 m) at each patch edge.

Part 1, one window (--sheet, --e, --n; default Gothenburg 639_31_55, 2012 flight, era 2010):
  * locates the prob ramp next to the window (steepest 50 m drop of the column-mean prob) and reports its
    easting against the frame (sheet) edges and the inference patch grid / taper overlap zones;
  * picks the patch on each side of the ramp and prints its model-input stats (raw grey values, the
    Luminance-Enhancer output the network sees) and every GroupNorm/LayerNorm layer's mean/std;
  * reruns inference on the right patch alone, (a) as in production and (b) with every normalisation
    layer using the LEFT patch's statistics; prints mean prob and writes a PNG
    (rows: left patch, right patch; columns: image | production prob | prob with left stats, viridis 0-1).
Part 2 (--scan): for the --scan-cities x --scan-eras prob maps, every production patch core (the 600 px
  inside the taper) with mean prob < --zero-thr over textured imagery (mean |gradient| above the city-era
  median), ranked by the mean prob of its 4 neighbours; written to results/seam_check/seam_scan_tiles.csv.

    python scripts/47_seam_check.py --scan
bwtreenet env (torch + rasterio). Existing outputs are not replaced without --overwrite.
"""
import argparse, csv, glob, importlib, os, sys
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
p19 = importlib.import_module("19_production_inference")   # load_model, positions, taper, PATCH, OVERLAP, SOURCES

PATCH, OVERLAP = p19.PATCH, p19.OVERLAP
STRIDE = PATCH - OVERLAP
OUT_DIR = "results/seam_check"
VIRIDIS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float32)


def viridis(p):
    x = np.clip(np.nan_to_num(p), 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(x.astype(int), len(VIRIDIS) - 2)
    f = (x - i)[..., None]
    return np.moveaxis((VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f).astype(np.uint8), -1, 0)


def grey8(img):
    v = img[img > 0]
    lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 1)
    g = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.stack([g] * 3)


def frame_paths(city, era, sheet):
    rel, is_rgb = p19.SOURCES[(city, era)]
    img = sorted(glob.glob(os.path.join(p19.HERE, rel, f"{sheet}_*.tif")))
    if not img:
        sys.exit(f"[error] no {city} {era} frame for {sheet} in {rel}")
    stem = os.path.splitext(os.path.basename(img[0]))[0]
    prob = os.path.join(p19.OUT_ROOT, f"{city}_{era}", f"{stem}_prob.tif")
    return img[0], prob, is_rgb


def read_image(path, is_rgb, win=None):
    with rasterio.open(path) as s:
        if is_rgb:
            return p19.rgb_to_gray(s.read([1, 2, 3], window=win)).astype(np.float32)
        return s.read(1, window=win).astype(np.float32)


# ---- normalisation hooks ------------------------------------------------------------------------
class NormSwap:
    """Re-implements every GroupNorm/LayerNorm forward so its statistics can be recorded and replayed.
    mode "off": normal behaviour (stats still recorded in .last); "apply": use .frozen stats."""

    def __init__(self, model, torch):
        self.torch, self.mode, self.last, self.frozen = torch, "off", {}, {}
        nn = torch.nn
        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.GroupNorm, nn.LayerNorm))]
        for name, m in self.layers:
            m.forward = self._make(name, m)

    def _make(self, name, m):
        torch = self.torch
        is_gn = isinstance(m, torch.nn.GroupNorm)

        def fwd(x):
            if is_gn:
                N, C = x.shape[:2]
                xr = x.reshape(N, m.num_groups, -1)
                mean, var = xr.mean(-1, keepdim=True), xr.var(-1, unbiased=False, keepdim=True)
            else:
                dims = tuple(range(x.dim() - len(m.normalized_shape), x.dim()))
                xr = x
                mean, var = x.mean(dims, keepdim=True), x.var(dims, unbiased=False, keepdim=True)
            self.last[name] = (mean.detach(), var.detach())
            if self.mode == "apply":
                mean, var = self.frozen[name]
            y = (xr - mean) / torch.sqrt(var + m.eps)
            if is_gn:
                y = y.reshape_as(x)
                if m.affine:
                    shp = (1, C) + (1,) * (x.dim() - 2)
                    y = y * m.weight.view(shp) + m.bias.view(shp)
            elif m.elementwise_affine:
                w, b = m.weight, m.bias
                if w.shape != x.shape[-w.dim():]:
                    # other patch size than the 1000 px the LayerNorm was built for; v19's LayerNorm
                    # affine maps are spatially constant, so the per-channel mean is exact
                    w, b = w.mean(dim=(-2, -1), keepdim=True), b.mean(dim=(-2, -1), keepdim=True)
                y = y * w + b
            return y
        return fwd


def run_patch(model, torch, dev, patch):
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(patch)).float()[None, None].to(dev)
        return p19.to_prob(model(t)[0].cpu().numpy())[1]


def le_view(model, torch, dev, patch):
    """What the network body sees: 255 - LE(x/255)*255."""
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(patch)).float()[None, None].to(dev)
        e, _ = model.le(t / 255.0)
        return (255 - e * 255).cpu().numpy()[0, 0]


def stats(a):
    return (f"mean {a.mean():7.2f} std {a.std():6.2f} p2 {np.percentile(a, 2):6.1f} "
            f"p98 {np.percentile(a, 98):6.1f} zero {100 * (a == 0).mean():4.1f}%")


def core_of(pos, x):
    """Patch origin whose core (inside the taper) contains x."""
    for p in pos:
        if p + OVERLAP <= x < p + PATCH - OVERLAP:
            return p
    return min(pos, key=lambda p: abs(p + PATCH / 2 - x))


# ---- part 1 ------------------------------------------------------------------------------------
def window_check(a):
    img_p, prob_p, is_rgb = frame_paths(a.city, a.era, a.sheet)
    with rasterio.open(prob_p) as s:
        tr, W, H = s.transform, s.width, s.height
        prob = s.read(1)
    res = tr.a
    left, top = tr.c, tr.f
    cx, cy = int((a.e - left) / res), int((top - a.n) / res)
    print(f"== {a.city} {a.era} {os.path.basename(img_p)}: frame E{left:.0f}-{left + W * res:.0f} "
          f"N{top - H * res:.0f}-{top:.0f}, {res} m; window centre px col {cx} row {cy}")
    half_w = int(round(a.win / 2 / res))
    r0, r1 = max(cy - half_w, 0), min(cy + half_w, H)
    x0, x1 = max(cx - 3 * half_w, 0), min(cx + 3 * half_w, W)
    band = prob[r0:r1, x0:x1].astype(np.float64)
    band[band < 0] = np.nan
    prof = np.nanmean(band, axis=0)
    prof = np.convolve(np.nan_to_num(prof), np.ones(9) / 9, mode="same")
    d = int(round(50 / res))
    drops = prof[:-d] - prof[d:]
    i = int(np.argmax(np.abs(drops)))
    mid = i + d // 2
    lvl_l = np.median(prof[max(mid - 6 * d // 2, 0):max(mid - 3 * d // 2, 1)])
    lvl_r = np.median(prof[min(mid + 3 * d // 2, len(prof) - 1):min(mid + 6 * d // 2, len(prof))])
    hi_, lo_ = max(lvl_l, lvl_r), min(lvl_l, lvl_r)
    span = [j for j in range(len(prof)) if lo_ + 0.1 * (hi_ - lo_) <= prof[j] <= lo_ + 0.9 * (hi_ - lo_)
            and abs(j - mid) < 4 * d]
    rs, re_ = (min(span), max(span)) if span else (mid, mid)
    E = lambda col: left + (x0 + col + 0.5) * res
    print(f"   ramp: prob {lvl_l:.2f} (left) -> {lvl_r:.2f} (right); 10-90% from E{E(rs):.1f} to E{E(re_):.1f} "
          f"({(re_ - rs) * res:.0f} m), steepest 50 m drop centred E{E(mid):.1f}")
    print(f"   frame edges: {E(rs) - left:.0f} m from the left edge, {left + W * res - E(re_):.0f} m from the right edge")
    pos_x = p19.positions(W, PATCH, STRIDE)
    zones = [(q, p + PATCH) for p, q in zip(pos_x, pos_x[1:]) if q < p + PATCH]
    print("   inference patches (x origin px) near the ramp, and taper overlap zones:")
    for p in pos_x:
        if abs(left + (p + PATCH / 2) * res - E(mid)) < 1000:
            print(f"     patch x {p:5d}-{p + PATCH:5d}  E{left + p * res:.0f}-{left + (p + PATCH) * res:.0f}"
                  f"  core E{left + (p + OVERLAP) * res:.0f}-{left + (p + PATCH - OVERLAP) * res:.0f}")
    for zs, ze in zones:
        es, ee = left + zs * res, left + ze * res
        if abs((es + ee) / 2 - E(mid)) < 1000:
            inside = es - 5 <= E(rs) and E(re_) <= ee + 5
            print(f"     overlap E{es:.0f}-{ee:.0f}" + ("   <- the ramp lies in this blend zone" if inside else ""))

    # patches either side of the ramp, on the patch row whose core holds the window centre
    pos_y = p19.positions(H, PATCH, STRIDE)
    py = core_of(pos_y, cy)
    pl = core_of(pos_x, x0 + rs - int(round(25 / res)))
    pr = core_of(pos_x, x0 + re_ + int(round(25 / res)))
    print(f"   left patch x {pl}, right patch x {pr}, row y {py}" + ("  (same patch: not a patch seam)" if pl == pr else ""))
    img = read_image(img_p, is_rgb)
    left_patch, right_patch = img[py:py + PATCH, pl:pl + PATCH], img[py:py + PATCH, pr:pr + PATCH]

    model, dev, torch = p19.load_model()
    # full fp32: with TF32 the hooked norms differ from the fused kernels by up to ~1e-3 in prob; without, ~1e-5
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sw = NormSwap(model, torch)
    print(f"   model input (raw grey)    left : {stats(left_patch)}")
    print(f"                             right: {stats(right_patch)}")
    print(f"   after LE (255 - LE*255)   left : {stats(le_view(model, torch, dev, left_patch))}")
    print(f"                             right: {stats(le_view(model, torch, dev, right_patch))}")
    sw.mode = "off"
    p_left = run_patch(model, torch, dev, left_patch)
    left_stats = dict(sw.last)
    p_right = run_patch(model, torch, dev, right_patch)
    right_stats = dict(sw.last)
    sw.frozen, sw.mode = left_stats, "apply"
    p_right_l = run_patch(model, torch, dev, right_patch)
    sw.mode = "off"

    used = [name for name, _ in sw.layers if name in left_stats and name in right_stats]
    print(f"   normalisation layers ({len(used)} used in forward, of {len(sw.layers)} defined): "
          f"mean / std of each layer's input, averaged over groups")
    print(f"     {'layer':34s} {'left mean':>10s} {'left std':>9s} {'right mean':>11s} {'right std':>10s} {'std R/L':>8s}")
    for name in used:
        (ml, vl), (mr, vr) = left_stats[name], right_stats[name]
        sl, sr = vl.sqrt().mean().item(), vr.sqrt().mean().item()
        print(f"     {name:34s} {ml.mean().item():10.4f} {sl:9.4f} {mr.mean().item():11.4f} {sr:10.4f} {sr / max(sl, 1e-12):8.3f}")

    # right of the ramp, inside the right patch's core (no blending there, so production = this patch alone)
    cols = slice(max(x0 + re_ + int(round(5 / res)) - pr, OVERLAP), PATCH - OVERLAP)
    rows = slice(max(r0 - py, OVERLAP), min(r1 - py, PATCH - OVERLAP))
    with rasterio.open(prob_p) as s:
        prod = s.read(1, window=Window(pr, py, PATCH, PATCH))[rows, cols]
    pr_r, pl_r = p_right[rows, cols], p_right_l[rows, cols]
    print(f"   right of the ramp, right patch core ({pr_r.size} px):")
    print(f"     production map (blended)            mean prob {prod[prod >= 0].mean():.3f}")
    print(f"     right patch rerun, production norms  mean prob {pr_r.mean():.3f}, >= {p19.THRESHOLD}: {100 * (pr_r >= p19.THRESHOLD).mean():.1f}%"
          f"   (should match the map above; if not, the map is not reproducible from this model and image)")
    print(f"     right patch rerun, LEFT patch norms  mean prob {pl_r.mean():.3f}, >= {p19.THRESHOLD}: {100 * (pl_r >= p19.THRESHOLD).mean():.1f}%")
    print(f"   whole patches: right production {p_right.mean():.3f}, right with left norms {p_right_l.mean():.3f}, "
          f"left production {p_left.mean():.3f}")

    out = os.path.join(OUT_DIR, f"{a.sheet}_E{a.e:.0f}_N{a.n:.0f}_{a.era}.png")
    if os.path.exists(out) and not a.overwrite:
        print(f"   {out} exists, not replaced (--overwrite)"); return
    k, gap = PATCH // 2, 6
    sub = lambda x: x[::2, ::2]                    # half size: 1000 px patch -> 500 px panel
    rgb = np.full((3, 2 * k + gap, 3 * k + 2 * gap), 255, np.uint8)
    tiles = [[grey8(sub(left_patch)), viridis(sub(p_left)), None],
             [grey8(sub(right_patch)), viridis(sub(p_right)), viridis(sub(p_right_l))]]
    for r, row in enumerate(tiles):
        for c, t in enumerate(row):
            if t is not None:
                rgb[:, r * (k + gap):r * (k + gap) + k, c * (k + gap):c * (k + gap) + k] = t
    with rasterio.open(out, "w", driver="PNG", width=rgb.shape[2], height=rgb.shape[1], count=3, dtype="uint8") as d:
        d.write(rgb)
    print(f"   wrote {out}  (rows: left patch, right patch; columns: image | production prob | prob with left norms)")


# ---- part 2 ------------------------------------------------------------------------------------
def scan(a):
    out = os.path.join(OUT_DIR, "seam_scan_tiles.csv")
    if os.path.exists(out) and not a.overwrite:
        print(f"\n{out} exists, not replaced (--overwrite)"); return
    rows = []
    for city in a.scan_cities:
        for era in a.scan_eras:
            if (city, era) not in p19.SOURCES:
                continue
            rel, is_rgb = p19.SOURCES[(city, era)]
            frames = sorted(glob.glob(os.path.join(p19.HERE, rel, "*.tif")))
            recs = []
            for f in frames:
                stem = os.path.splitext(os.path.basename(f))[0]
                pp = os.path.join(p19.OUT_ROOT, f"{city}_{era}", f"{stem}_prob.tif")
                if not os.path.exists(pp):
                    continue
                with rasterio.open(pp) as s:
                    prob, tr = s.read(1), s.transform
                img = read_image(f, is_rgb)
                H, W = prob.shape
                gx = np.zeros_like(img); gx[:, 1:] = np.abs(np.diff(img, axis=1))
                gy = np.zeros_like(img); gy[1:, :] = np.abs(np.diff(img, axis=0))
                grad = gx + gy
                py, px = p19.positions(H, PATCH, STRIDE), p19.positions(W, PATCH, STRIDE)
                grid = {}
                for y in py:
                    for x in px:
                        cs = (slice(y + OVERLAP, y + PATCH - OVERLAP), slice(x + OVERLAP, x + PATCH - OVERLAP))
                        pv, iv = prob[cs], img[cs]
                        ok = (pv >= 0) & (iv > 0)
                        if ok.mean() < 0.8:
                            continue
                        grid[(y, x)] = dict(city=city, era=era, frame=stem, y=y, x=x,
                                            centre_e=round(tr.c + (x + PATCH / 2) * tr.a),
                                            centre_n=round(tr.f - (y + PATCH / 2) * abs(tr.e)),
                                            mean_prob=float(pv[ok].mean()), img_mean=float(iv[ok].mean()),
                                            img_std=float(iv[ok].std()), texture=float(grad[cs][ok].mean()))
                yi, xi = {y: n for n, y in enumerate(py)}, {x: n for n, x in enumerate(px)}
                for (y, x), r in grid.items():
                    nb = [grid.get((py[yi[y] + dy], px[xi[x] + dx]))
                          for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                          if 0 <= yi[y] + dy < len(py) and 0 <= xi[x] + dx < len(px)]
                    nb = [n["mean_prob"] for n in nb if n]
                    r["neigh_mean_prob"] = float(np.mean(nb)) if nb else float("nan")
                    recs.append(r)
            if not recs:
                print(f"\n{city} {era}: no prob maps"); continue
            tex_med = np.median([r["texture"] for r in recs])
            hits = [r for r in recs if r["mean_prob"] < a.zero_thr and r["texture"] >= tex_med]
            hits.sort(key=lambda r: -(np.nan_to_num(r["neigh_mean_prob"]) - r["mean_prob"]))
            print(f"\n{city} {era}: {len(recs)} patch cores, texture median {tex_med:.1f}; "
                  f"{len(hits)} with mean prob < {a.zero_thr} over above-median texture")
            print(f"   {'frame':22s} {'centre E':>9s} {'centre N':>9s} {'prob':>6s} {'neigh':>6s} {'img mean':>8s} {'img std':>8s} {'texture':>8s}")
            for r in hits[:a.scan_top]:
                print(f"   {r['frame']:22s} {r['centre_e']:9d} {r['centre_n']:9d} {r['mean_prob']:6.3f} "
                      f"{r['neigh_mean_prob']:6.3f} {r['img_mean']:8.1f} {r['img_std']:8.1f} {r['texture']:8.1f}")
            for r in hits:
                r["texture_median"] = round(tex_med, 2)
            rows += hits
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} flagged patch cores)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="gtb")
    ap.add_argument("--era", default="2010")
    ap.add_argument("--sheet", default="639_31_55")
    ap.add_argument("--e", type=float, default=317875)
    ap.add_argument("--n", type=float, default=6395875)
    ap.add_argument("--win", type=float, default=250.0, help="window size in m")
    ap.add_argument("--no-window", action="store_true", help="skip part 1")
    ap.add_argument("--scan", action="store_true", help="run part 2")
    ap.add_argument("--scan-cities", nargs="+", default=["gtb", "sth"])
    ap.add_argument("--scan-eras", nargs="+", default=["2005", "2010"])
    ap.add_argument("--zero-thr", type=float, default=0.02)
    ap.add_argument("--scan-top", type=int, default=25)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if not a.no_window:
        window_check(a)
    if a.scan:
        scan(a)
