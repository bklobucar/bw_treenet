"""
23_model_comparison_figure.py -- qualitative comparison of BWTreeNet versions.

Rows
  * held-out Malmo 1970 benchmark tiles (same set as 10_score_models.py), shown
    as error maps against the reference labels: TP / FP / FN, with per-tile IoU
  * windows from other city-eras (no reference): prediction overlay only.
    NOTE: these may overlap training areas -- illustrative, not evaluative.

Columns
  input | reference | one per model

Inference reuses 19_production_inference.py directly (sliding window, taper
blending, softmax detection, tatort + hydrography masking), so every panel
shows exactly what the production pipeline would produce for that model.

Every panel covers 500 m on the ground: 1000 px at 0.5 m, or the central
500 px of a 1 m (1990s) window.

Run from anywhere, on a GPU node with the bwtreenet env active:
    python scripts/23_model_comparison_figure.py
Options:
    --models v12,v18b,v18a,v19     column order; add v20 etc. if wanted
    --threshold 0.6                same threshold for every model
    --n-bench 3                    number of benchmark rows
    --extra gtb_1960,sth_1960,gtb_1970,sth_1990,malmo_1990
    --seed 7                       change to get different extra windows
"""
import os, sys, io, csv, glob, argparse, contextlib, importlib
import numpy as np
import rasterio
import geopandas as gpd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)   # production is run from the repo root; LE path may be relative
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, os.path.join(HERE, "BWTreeNet", "GuiTest"))
prod = importlib.import_module("19_production_inference")
from model.BWTreeNet import BWTreeNet

# ── configuration ───────────────────────────────────────────────────────────
MODEL_PATHS = {v: os.path.join(HERE, "models", f"bwtreenet_{v}_best.pt")
               for v in ["v12", "v18a", "v18b", "v18c", "v18d", "v19", "v20"]}
DROP_LE = {"v18b", "v18c"}          # trained with the LE bypassed (from slurm cfg)
HEADERS = {
    "v12":  "v12\nbaseline",
    "v18a": "v18a\nwith LE",
    "v18b": "v18b\nLE bypassed",
    "v18c": "v18c\nno LE + degrade",
    "v18d": "v18d\nLE + degrade",
    "v19":  "v19\n+ era labels",
    "v20":  "v20\n+ 30 more labels",
}
BENCH_IMG = os.path.join(HERE, "data/processed/malmo/tiles_1970/images")
BENCH_LBL = os.path.join(HERE, "data/processed/malmo/tiles_1970/labels")
BENCH_TAT = os.path.join(HERE, "data/processed/malmo/tiles_1970/tatortmask")
CITY = {"malmo": "Malmö", "gtb": "Gothenburg", "sth": "Stockholm"}
CTX = 1400          # context window read for extra rows (blending like production)
DISPLAY_M = 500     # ground extent of every panel

GREEN = (0 / 255, 158 / 255, 115 / 255)   # canopy / true positive   (Okabe-Ito)
VERM = (213 / 255, 94 / 255, 0 / 255)     # false positive
BLUE = (0 / 255, 114 / 255, 178 / 255)    # false negative
# ────────────────────────────────────────────────────────────────────────────


class LEPassthrough(torch.nn.Module):
    """Same bypass as 07_train_v18.py --drop-le: must return TWO values."""
    def forward(self, x):
        return x, None


def load_model(tag, dev):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        m = BWTreeNet(n_class=prod.N_CLASSES).to(dev)
    log = buf.getvalue()
    sys.stdout.write(log)
    if "Luminance Enhancer loaded" not in log:
        raise RuntimeError(
            "BWTreeNet did not report 'Luminance Enhancer loaded and frozen'. The LE "
            "path failed silently -- results would be degraded. Stopping.")
    sd = torch.load(MODEL_PATHS[tag], map_location=dev, weights_only=False)
    for key in ("model_state_dict", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
    sd = {k: v for k, v in sd.items()
          if "SharpFeature.weight" not in k and "SharpOri.weight" not in k}
    res = m.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys
               if "SharpFeature" not in k and "SharpOri" not in k and not k.startswith("le.")]
    if missing or res.unexpected_keys:
        print(f"[warn] {tag}: {len(missing)} missing / "
              f"{len(res.unexpected_keys)} unexpected keys")
    if len(missing) > 10:
        raise RuntimeError(f"{tag}: checkpoint does not match the architecture "
                           f"({len(missing)} missing keys) -- strict=False would hide this")
    if tag in DROP_LE:
        m.le = LEPassthrough().to(dev)
        print(f"[info] {tag}: Luminance Enhancer bypassed, as trained")
    m.eval()
    print(f"[info] {tag} loaded")
    return m


# ── row selection ───────────────────────────────────────────────────────────
def find_pair(folder, name):
    p = os.path.join(folder, name)
    if os.path.exists(p):
        return p
    c = sorted(glob.glob(os.path.join(folder, os.path.splitext(name)[0] + "*")))
    return c[0] if c else None


def gt_from_label(lbl):
    vals = set(np.unique(lbl).tolist())
    if 1 in vals:                       # 0/1 labels, 255 = ignore if present
        return lbl == 1, lbl == 255
    return lbl > 0, np.zeros(lbl.shape, bool)


def bench_rows(n):
    cands, shown = [], False
    for ip in sorted(glob.glob(os.path.join(BENCH_IMG, "*.tif"))):
        name = os.path.basename(ip)
        lp, tp = find_pair(BENCH_LBL, name), find_pair(BENCH_TAT, name)
        if lp is None:
            continue
        with rasterio.open(ip) as s:
            img, nodv, tr = s.read(1), (0 if s.nodata is None else s.nodata), s.transform
            res = abs(s.transform.a)
        with rasterio.open(lp) as s:
            lbl = s.read(1)
        if not shown:
            print(f"[info] benchmark label values: {sorted(np.unique(lbl).tolist())}")
            shown = True
        if img.shape != (prod.PATCH, prod.PATCH):
            continue
        valid = img != nodv
        if tp:
            with rasterio.open(tp) as s:
                valid &= s.read(1) > 0
        gt, ign = gt_from_label(lbl)
        valid &= ~ign
        if valid.mean() < 0.6:
            continue
        cx, cy = tr * (img.shape[1] / 2, img.shape[0] / 2)
        cands.append(dict(frac=float(gt[valid].mean()), source=name, img=img,
                          valid_d=valid, gt_d=gt, res=res, x=cx, y=cy))
    if not cands:
        print("[warn] no usable benchmark tiles found")
        return []
    cands.sort(key=lambda d: d["frac"])
    idx = sorted({int(round(q * (len(cands) - 1))) for q in np.linspace(0.2, 0.85, n)})
    rows = []
    for k, i in enumerate(idx, 1):
        r = cands[i]
        r.update(title=f"Malmö 1970\nbenchmark {k}", c0=0, disp=prod.PATCH, img_d=r["img"])
        rows.append(r)
        print(f"  bench {k}: {r['source']}  reference canopy {100 * r['frac']:.1f}%")
    return rows


def extra_row(city, era, rng, tat):
    rel, is_rgb = prod.SOURCES[(city, era)]
    frames = [f for f in sorted(glob.glob(os.path.join(HERE, rel, "*.tif")))
              if "_clip" not in os.path.basename(f)]
    rng.shuffle(frames)
    hp = prod.HYDRO.get(city, "")
    hyd = gpd.read_file(hp) if os.path.exists(hp) else None
    for fp in frames[:6]:
        try:
            with rasterio.open(fp) as src:
                band = (prod.rgb_to_gray(src.read([1, 2, 3])) if is_rgb
                        else src.read(1))
                nodv = 0 if src.nodata is None else src.nodata
                valid_full = (band != nodv) & prod.build_mask(src, tat, hyd)
                tr, res = src.transform, abs(src.transform.a)
        except Exception as e:
            print(f"  {city}_{era}: skip {os.path.basename(fp)} ({type(e).__name__})")
            continue
        H, W = band.shape
        if H < CTX or W < CTX:
            continue
        disp = int(round(DISPLAY_M / res))
        c0 = (CTX - disp) // 2
        passing = []
        for _ in range(400):
            y0, x0 = int(rng.integers(0, H - CTX)), int(rng.integers(0, W - CTX))
            v = valid_full[y0 + c0:y0 + c0 + disp, x0 + c0:x0 + c0 + disp]
            if v.mean() >= 0.95:
                d = band[y0 + c0:y0 + c0 + disp, x0 + c0:x0 + c0 + disp]
                passing.append((float(d.std()), y0, x0))
            if len(passing) >= 25:
                break
        if not passing:
            continue
        _, y0, x0 = max(passing)       # most textured of the valid candidates
        img = np.asarray(band[y0:y0 + CTX, x0:x0 + CTX])
        valid = valid_full[y0:y0 + CTX, x0:x0 + CTX]
        stem = os.path.splitext(os.path.basename(fp))[0]
        year = stem.split("_")[-1] if stem.split("_")[-1].isdigit() else era
        cx, cy = tr * (x0 + CTX / 2, y0 + CTX / 2)
        title = f"{CITY[city]} {year}" + (f"\n({res:g} m imagery)" if res > 0.75 else "")
        print(f"  {city}_{era}: {stem} centre E{cx:.0f} N{cy:.0f}")
        return dict(title=title, source=os.path.basename(fp), img=img, res=res,
                    x=cx, y=cy, c0=c0, disp=disp,
                    img_d=img[c0:c0 + disp, c0:c0 + disp],
                    valid_d=valid[c0:c0 + disp, c0:c0 + disp], gt_d=None)
    print(f"  {city}_{era}: no valid window found")
    return None


# ── drawing ─────────────────────────────────────────────────────────────────
def stretch(img, valid):
    v = img[valid] if valid.any() else img.ravel()
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def show(ax, base, layers, valid, alpha=0.65):
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros(base.shape + (4,), np.float32)
    for m, col in layers:
        rgba[m, :3] = col
        rgba[m, 3] = alpha
    rgba[~valid] = (1, 1, 1, 0.6)
    ax.imshow(rgba)


def scalebar(ax, npx, res, length_m=100):
    L = length_m / res
    x0, y0 = npx * 0.06, npx * 0.90
    ax.add_patch(Rectangle((x0, y0), L, npx * 0.02, facecolor="white",
                           edgecolor="black", lw=0.4))
    t = ax.text(x0 + L / 2, y0 - npx * 0.015, f"{length_m} m", ha="center",
                va="bottom", fontsize=5, color="white")
    t.set_path_effects([pe.withStroke(linewidth=1, foreground="black")])


def draw(rows, models, args):
    ncol, nrow = 2 + len(models), len(rows)
    cw = 1.3
    fig, axes = plt.subplots(nrow, ncol, figsize=(cw * ncol + 0.3, cw * nrow + 0.6),
                             squeeze=False)
    heads = ["Input", "Reference"] + [HEADERS.get(m, m) for m in models]
    stats = []
    for i, r in enumerate(rows):
        img, valid, gt = r["img_d"], r["valid_d"], r["gt_d"]
        base = stretch(img, valid)
        rec = dict(row=r["title"].replace("\n", " "), source=r["source"],
                   centre_E=round(r["x"]), centre_N=round(r["y"]), res_m=r["res"])
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(heads[j], fontsize=7)
            if j == 0:
                ax.imshow(base, cmap="gray", vmin=0, vmax=1)
                ax.set_ylabel(r["title"], fontsize=7)
                scalebar(ax, img.shape[0], r["res"])
            elif j == 1:
                if gt is None:
                    ax.imshow(base, cmap="gray", vmin=0, vmax=1, alpha=0.3)
                    ax.text(0.5, 0.5, "no reference", transform=ax.transAxes,
                            ha="center", va="center", fontsize=6, color="0.3")
                else:
                    show(ax, base, [(gt & valid, GREEN)], valid)
            else:
                tag = models[j - 2]
                pred = r["pred"][tag] & valid
                if gt is None:
                    show(ax, base, [(pred, GREEN)], valid)
                else:
                    g = gt & valid
                    tp, fp, fn = pred & g, pred & ~g, ~pred & g
                    show(ax, base, [(tp, GREEN), (fp, VERM), (fn, BLUE)], valid)
                    iou = tp.sum() / max(tp.sum() + fp.sum() + fn.sum(), 1)
                    ax.text(0.04, 0.04, f"IoU {iou:.2f}", transform=ax.transAxes,
                            fontsize=5.5, color="white", ha="left", va="bottom",
                            bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
                    rec[f"IoU_{tag}"] = round(float(iou), 3)
        stats.append(rec)

    handles = [Patch(facecolor=GREEN, label="Canopy / true positive"),
               Patch(facecolor=VERM, label="False positive"),
               Patch(facecolor=BLUE, label="False negative"),
               Patch(facecolor="white", edgecolor="0.5", lw=0.5,
                     label="Outside study area / water")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               frameon=False, handlelength=1.2)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.1, right=0.995,
                        top=1 - 0.45 / (cw * nrow + 0.6), bottom=0.35 / (cw * nrow + 0.6))

    base_name = os.path.join(args.out, "model_comparison")
    fig.savefig(base_name + ".png", dpi=300)
    fig.savefig(base_name + ".pdf")
    fields = sorted({k for s in stats for k in s},
                    key=lambda k: (not k in ("row", "source", "centre_E", "centre_N", "res_m"), k))
    with open(base_name + "_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stats)
    print(f"[done] {base_name}.png / .pdf / _rows.csv")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v12,v18b,v18a,v19")
    ap.add_argument("--threshold", type=float, default=prod.THRESHOLD)
    ap.add_argument("--n-bench", type=int, default=3)
    ap.add_argument("--extra", default="gtb_1960,sth_1960,gtb_1970,sth_1990,malmo_1990")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "figures"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_PATHS or not os.path.exists(MODEL_PATHS[m]):
            sys.exit(f"unknown or missing model: {m}")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] models={models} threshold={args.threshold} device={dev}")

    print("== selecting rows")
    rows = bench_rows(args.n_bench)
    tat = (gpd.read_file(prod.TATORT, layer=prod.TATORT_LAYER)
           if os.path.exists(prod.TATORT) else None)
    for ce in [e.strip() for e in args.extra.split(",") if e.strip()]:
        city, era = ce.split("_")
        r = extra_row(city, era, rng, tat)
        if r:
            rows.append(r)
    if not rows:
        sys.exit("no rows selected")

    print("== inference")
    for tag in models:
        model = load_model(tag, dev)
        for r in rows:
            prob = prod.infer_frame(model, dev, torch, r["img"].astype(np.float32))
            c0, d = r["c0"], r["disp"]
            r.setdefault("pred", {})[tag] = prob[c0:c0 + d, c0:c0 + d] >= args.threshold
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print("== drawing")
    draw(rows, models, args)


if __name__ == "__main__":
    main()
