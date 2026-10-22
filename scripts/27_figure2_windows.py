"""
27_figure2_windows.py -- Figure 2 for Paper 1: same window, different definitions of canopy.

One row per window. Columns:
  historical image | present-day orthophoto | released 1946 weights | fine-tuned ensemble |
  scratch ensemble | v19
Canopy is overlaid in green (as in Figure 1). The cover figure on each mask panel is canopy as a
percentage of the window's area inside the 2023 tatort (no hydrography mask), so the numbers are
directly comparable with Table 6 and results/era_visual_check/windows.csv.

Windows are the named windows of 26_era_visual_check.py (seeded; names are <city_era>_<index>,
exactly as in 26b_export_windows.py).

Present-day panel, in order of preference:
  1. results/era_visual_check/modern/<name>.png   (your own screenshot or export, e.g. from QGIS)
  2. a mosaic of the modern training tiles (data/processed/<city>/tiles/images) covering the
     window, if they cover at least half of it (grey, resampled to the window's pixel size)
  3. an empty panel saying none is available

Run on a GPU node:
    source ~/bw_setup.sh
    python scripts/27_figure2_windows.py gtb_1970_05 gtb_1990_01 malmo_1960_02
    python scripts/27_figure2_windows.py gtb_1970_05 gtb_1990_01 malmo_1960_02 --no-modern   # without the present-day column
    python scripts/27_figure2_windows.py --selftest     # layout check on synthetic data (no GPU, no data)
Output: results/figures/figure2_windows.png / .pdf and figure2_panels.csv
"""
import os, sys, csv, argparse, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "results", "era_visual_check")
OUT_DEFAULT = os.path.join(HERE, "results", "figures")
WIN = 1000
T_SW, T_V19, T_ENS = 0.5, 0.6, 0.5          # thresholds: released weights, v19, ensembles
FT_TAGS = ["ft46_s1", "ft46_s2"]
SC_TAGS = ["sc_s1", "sc_s2"]
CITY = {"malmo": "Malmö", "gtb": "Gothenburg", "sth": "Stockholm"}
ERA = {"1960": "1960s", "1970": "1970s", "1990": "1990s"}
COLS = ["Historical image", "Present-day\northophoto", "Released 1946\nweights",
        "Fine-tuned\nensemble", "Scratch\nensemble", "v19"]
MASK_KEYS = ["swiss", "ft", "sc", "v19"]
GREEN = (0.1, 0.85, 0.2, 0.5)


# ---------------------------------------------------------------------------------- drawing
def stretch(img):
    pos = img[img > 0]
    lo, hi = np.percentile(pos, [2, 98]) if pos.size else (0, 255)
    return np.clip((img - lo) / max(hi - lo, 1), 0, 1)


def panel(ax, img=None, mask=None, cover=None, res=None, scale=False, note=None):
    ax.axis("off")
    if img is None:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#eeeeee"))
        ax.text(0.5, 0.5, note or "not available", ha="center", va="center", fontsize=6, color="#555555")
        return
    if img.ndim == 3:                                   # RGB screenshot supplied by the user
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(stretch(img), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    n = img.shape[1]
    if mask is not None:
        rgba = np.zeros(mask.shape + (4,), np.float32)
        rgba[mask] = GREEN
        ax.imshow(rgba, interpolation="nearest")
    if cover is not None:
        ax.text(0.03, 0.03, f"{100 * cover:.1f}%", transform=ax.transAxes, fontsize=6, color="white",
                ha="left", va="bottom", bbox=dict(facecolor="black", alpha=0.6, pad=1.5, lw=0))
    if scale and res:
        L = 100.0 / res                                  # 100 m in pixels
        x1 = 0.95 * n; x0 = x1 - L; y = 0.93 * n
        ax.plot([x0, x1], [y, y], color="white", lw=1.6, solid_capstyle="butt",
                path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()])
        ax.text(x1, y - 0.03 * n, "100 m", color="white", fontsize=5.5, ha="right", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(img.shape[0] - 0.5, -0.5)


def draw_figure(rows, out_base, modern_col=True):
    n = len(rows)
    cols = COLS if modern_col else COLS[:1] + COLS[2:]
    first_mask = 2 if modern_col else 1
    fig, axes = plt.subplots(n, len(cols), figsize=(7.4, 1.28 * n + 0.5) if modern_col else (6.3, 1.28 * n + 0.5),
                             squeeze=False)
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=6.5, pad=3)
    for i, r in enumerate(rows):
        panel(axes[i, 0], r["img"], res=r["res"], scale=True)
        if modern_col:
            if r["modern"] is None:
                panel(axes[i, 1], None, note="no present-day\nimage")
            else:
                panel(axes[i, 1], r["modern"])
        for j, k in enumerate(MASK_KEYS):
            panel(axes[i, first_mask + j], r["img"], r["masks"][k], r["cover"][k])
        axes[i, 0].text(-0.05, 0.5, f"({chr(97 + i)}) {r['label']}", transform=axes[i, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=7)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.93 if n > 1 else 0.88, bottom=0.01,
                        wspace=0.025, hspace=0.04)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(f"{out_base}.{ext}", **kw)
    plt.close(fig)


# ---------------------------------------------------------------------------------- selftest
def selftest(out_dir):
    rng = np.random.default_rng(0)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, (lab, res) in enumerate([("Gothenburg 1970s", 0.5), ("Gothenburg 1990s", 1.0), ("Malmö 1960s", 0.5)]):
        yy, xx = np.mgrid[0:WIN, 0:WIN]
        img = (120 + 60 * np.sin(xx / 70.0 + i) * np.cos(yy / 90.0) + rng.normal(0, 12, (WIN, WIN))).clip(1, 255)
        base = np.sin(xx / 40.0) * np.cos(yy / 55.0 + i)
        masks = {k: base + 0.25 * j + rng.normal(0, 0.15, (WIN, WIN)) > 0.55 for j, k in enumerate(MASK_KEYS)}
        modern = None if i == 1 else (img[::-1] * 0.9)
        rows.append(dict(img=img, res=res, label=lab, modern=modern, masks=masks,
                         cover={k: float(m.mean()) for k, m in masks.items()}))
    draw_figure(rows, os.path.join(out_dir, "figure2_selftest"))
    print("selftest figure written to", out_dir)


# ---------------------------------------------------------------------------------- real run
_tile_index = {}


def modern_crop(city, bounds, res):
    import glob, rasterio
    from rasterio.merge import merge
    from shapely.geometry import box
    if city not in _tile_index:
        files = sorted(glob.glob(os.path.join(HERE, "data", "processed", city, "tiles", "images", "*.tif")))
        idx = []
        for f in files:
            with rasterio.open(f) as r:
                idx.append((f, box(*r.bounds)))
        _tile_index[city] = idx
        print(f"  indexed {len(idx)} modern tiles for {city}")
    want = box(*bounds)
    hits = [f for f, g in _tile_index[city] if g.intersects(want)]
    if not hits:
        return None
    srcs = [rasterio.open(f) for f in hits]
    try:
        arr, _ = merge(srcs, bounds=bounds, res=res, nodata=0)
    finally:
        for s in srcs:
            s.close()
    a = arr[0].astype(np.float32)
    return a if (a > 0).mean() >= 0.5 else None


def run(names, out_dir, t_sw, t_v19, t_ens, modern_col=True):
    import rasterio
    from rasterio.windows import Window, bounds as win_bounds
    from rasterio.features import geometry_mask

    spec = importlib.util.spec_from_file_location("c26", os.path.join(HERE, "scripts", "26_era_visual_check.py"))
    c26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c26)
    b = c26.b

    table = list(csv.DictReader(open(os.path.join(SRC, "windows.csv"))))
    counters, index = {}, {}
    for r in table:                                      # same naming as the PNGs and 26b
        i = counters.get(r["city_era"], 0); counters[r["city_era"]] = i + 1
        index[f"{r['city_era']}_{i:02d}"] = r
    missing = [w for w in names if w not in index]
    if missing:
        sys.exit(f"unknown window names: {missing}\navailable: {sorted(index)}")

    ckpt = lambda t: os.path.join(HERE, "models", f"bwtreenet_{t}_best.pt")
    absent = [t for t in FT_TAGS + SC_TAGS if not os.path.exists(ckpt(t))]
    if absent:
        sys.exit(f"missing checkpoints in models/: {absent}")

    print("loading models")
    m_sw, _ = b.build("swiss46_eq_05m")
    m_v19, _ = b.build("v19")
    load = lambda t: b.SwissRunner(b.load_swiss(ckpt(t)), "eq", None).eval()
    m_ft = [load(t) for t in FT_TAGS]
    m_sc = [load(t) for t in SC_TAGS]

    rows, csv_rows = [], []
    os.makedirs(os.path.join(SRC, "modern"), exist_ok=True)
    for name in names:
        r = index[name]
        city, era = r["city_era"].split("_")
        path = os.path.join(HERE, c26.IMG[r["city_era"]], r["frame"])
        w = Window(int(r["x"]), int(r["y"]), WIN, WIN)
        with rasterio.open(path) as src:
            img = src.read(1, window=w).astype(np.float32)
            tr, bnd, crs = src.window_transform(w), win_bounds(w, src.transform), src.crs
        res = float(r["res_m"])
        tat = c26.tatort_in(crs)
        geoms = tat.iloc[list(tat.sindex.intersection(bnd))].geometry
        inside = ~geometry_mask(geoms, out_shape=(WIN, WIN), transform=tr) if len(geoms) else np.ones((WIN, WIN), bool)

        p = {"swiss": b.predict(m_sw, img, 1),
             "v19": b.predict(m_v19, img, 1),
             "ft": np.mean([b.predict(m, img, 1) for m in m_ft], axis=0),
             "sc": np.mean([b.predict(m, img, 1) for m in m_sc], axis=0)}
        thr = {"swiss": t_sw, "v19": t_v19, "ft": t_ens, "sc": t_ens}
        masks = {k: p[k] >= thr[k] for k in MASK_KEYS}
        cover = {k: float(masks[k][inside].mean()) for k in MASK_KEYS}

        png = os.path.join(SRC, "modern", f"{name}.png")
        if not modern_col:
            modern, src_note = None, "off"
        elif os.path.exists(png):
            modern = matplotlib.image.imread(png)
            src_note = "png"
        else:
            modern = modern_crop(city, bnd, res)
            src_note = "tiles" if modern is not None else "none"
        label = f"{CITY[city]} {ERA[era]}"
        rows.append(dict(img=img, res=res, label=label, modern=modern, masks=masks, cover=cover))
        csv_rows.append(dict(window=name, frame=r["frame"], x=r["x"], y=r["y"], present_day=src_note,
                             **{f"cover_{k}": round(cover[k], 4) for k in MASK_KEYS}))
        print(f"  {name:16s} " + "  ".join(f"{k} {100 * cover[k]:5.1f}%" for k in MASK_KEYS) + f"  present-day: {src_note}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "figure2_panels.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(csv_rows[0])); wr.writeheader(); wr.writerows(csv_rows)
    draw_figure(rows, os.path.join(out_dir, "figure2_windows"), modern_col)
    print(f"\nfigure and panel table in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("windows", nargs="*", help="window names, e.g. gtb_1970_05 (see windows.csv / 26b)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--t-swiss", type=float, default=T_SW)
    ap.add_argument("--t-v19", type=float, default=T_V19)
    ap.add_argument("--t-ens", type=float, default=T_ENS, help="threshold for both ensembles")
    ap.add_argument("--no-modern", action="store_true", help="drop the present-day orthophoto column")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.out)
    elif a.windows:
        run(a.windows, a.out, a.t_swiss, a.t_v19, a.t_ens, not a.no_modern)
    else:
        ap.error("give window names or --selftest")
