"""
19_production_inference.py -- run the production model (v19) across a city-era
directory and write georeferenced canopy rasters.

Per source frame it writes, on the frame's own grid:
  <stem>_prob.tif    canopy probability 0-1 (float32, nodata -1)
  <stem>_canopy.tif  binary canopy 1 / background 0 (uint8, nodata 0 transparent)

Masking is the combination of BOTH available masks, because neither alone is
sufficient:
  * tätort (SCB 2023) defines the study area and removes open sea and rural land,
    but harbour/canal water sits INSIDE the urban boundary by construction.
  * hydrography (Lantmäteriet VATTEN polygons) removes water wherever it is,
    including the waterfront that tätort misses.
Water shown as canopy is the most conspicuous possible error on a map a
municipality will scrutinise, so both are applied.

The 2023 tätort boundary is used for EVERY era deliberately: the study measures
canopy change within today's urban extent, held constant, so change reflects
canopy rather than boundary drift. State this explicitly in any write-up.

Unreadable frames are skipped with a warning rather than aborting the run
(GTB 1960s has known LZW corruption in 2 of 17 frames).

Usage:
    python 19_production_inference.py --city malmo --era 1960
    python 19_production_inference.py --city gtb --era 1990   # RGB -> grayscale
    python 19_production_inference.py --list                  # show configured sets
"""
import os, sys, glob, argparse, time
import numpy as np
import rasterio
from rasterio.features import rasterize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── configuration ───────────────────────────────────────────────────────────
WEIGHTS   = os.path.join(HERE, "models", "bwtreenet_v19_best.pt")
TATORT    = os.path.join(HERE, "data", "Tatorter_2023.gpkg")
TATORT_LAYER = "Tatorter_2023"
OUT_ROOT  = os.path.join(HERE, "results", "inference_v19")

# hydrography per city (folders as downloaded: 1=STH, 2=Malmo, 3=GTB)
HYDRO = {
    "malmo": os.path.join(HERE, "data/hydro/2/mv_south.shp"),
    "gtb":   os.path.join(HERE, "data/hydro/3/mv_south.shp"),
    "sth":   os.path.join(HERE, "data/hydro/1/mv_south.shp"),
}

# (city, era) -> source imagery directory, and whether it needs RGB->grayscale
SOURCES = {
    ("malmo", "1960"): ("data/raw/malmo/1960_OF_gray_mmo", False),
    ("malmo", "1970"): ("data/raw/malmo/1970_OF_gray_mmo", False),
    ("malmo", "1990"): ("data/raw/malmo/1990_OF_gray_mmo", False),
    ("gtb",   "1960"): ("data/raw/gtb/1960_OF_gray_gtb",   False),
    ("gtb",   "1970"): ("data/raw/gtb/1970_OF_gray_gtb",   False),
    ("gtb",   "1990"): ("data/raw/gtb/1990_OF_rgb_gtb",    False),
    ("sth",   "1960"): ("data/raw/sth/1960_OF_gray_sth",   False),
    ("sth",   "1970"): ("data/raw/sth/1970_OF_gray_sth",   False),
    ("sth",   "1990"): ("data/raw/sth/1990_OF_gray_sth",   False),
    ("malmo", "1990r05"): ("data/raw/malmo/1990r05_OF_gray_mmo", False),
    ("gtb",   "1990r05"): ("data/raw/gtb/1990r05_OF_gray_gtb",   False),
    ("sth",   "1990r05"): ("data/raw/sth/1990r05_OF_gray_sth",   False),
    ("malmo", "1990r10"): ("data/raw/malmo/1990r10_OF_gray_mmo", False),
    ("gtb",   "1990r10"): ("data/raw/gtb/1990r10_OF_gray_gtb",   False),
    ("sth",   "1990r10"): ("data/raw/sth/1990r10_OF_gray_sth",   False),
    ("malmo", "2005"):    ("data/raw/malmo/2005_OF_gray_mmo",  False),
    ("gtb",   "2005"):    ("data/raw/gtb/2005_OF_gray_gtb",    False),
    ("sth",   "2005"):    ("data/raw/sth/2005_OF_gray_sth",    False),
    ("gtb",   "2005cmp"): ("data/raw/gtb/2005cmp_OF_gray_gtb", False),
    ("malmo", "2005deg"): ("data/raw/malmo/2005deg_OF_gray_mmo", False),
    ("gtb",   "2005deg"): ("data/raw/gtb/2005deg_OF_gray_gtb",   False),
    ("sth",   "2005deg"): ("data/raw/sth/2005deg_OF_gray_sth",   False),
    ("malmo", "2010"):    ("data/raw/malmo/2010_OF_gray_mmo",  False),
    ("gtb",   "2010"):    ("data/raw/gtb/2010_OF_gray_gtb",    False),
    ("sth",   "2010"):    ("data/raw/sth/2010_OF_gray_sth",    False),
    ("malmo", "2020"):    ("data/raw/malmo/2020_OF_gray_mmo",  False),
    ("gtb",   "2020"):    ("data/raw/gtb/2020_OF_gray_gtb",    False),
    ("sth",   "2020"):    ("data/raw/sth/2020_OF_gray_sth",    False),
}

PATCH, OVERLAP, N_CLASSES = 1000, 200, 2
THRESHOLD = 0.6      # v19 peaks here on STH and has much better precision on both
# ────────────────────────────────────────────────────────────────────────────


def rgb_to_gray(bands):
    """ITU-R BT.601 luminance. NOTE: must match how the modern training imagery
    was converted. If a different weighting was used there, change it here or the
    model sees a different input distribution than it was trained on."""
    r, g, b = bands[0].astype(np.float32), bands[1].astype(np.float32), bands[2].astype(np.float32)
    return np.floor((r + g + b) / 3.0)  # matches 07_prepare_tiles_all_cities.py (mean, uint8 truncation)


# ── model ───────────────────────────────────────────────────────────────────
def load_model():
    import torch
    sys.path.insert(0, os.path.join(HERE, "BWTreeNet", "GuiTest"))
    from model.BWTreeNet import BWTreeNet
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = BWTreeNet(n_class=N_CLASSES).to(dev)
    sd = torch.load(WEIGHTS, map_location=dev, weights_only=False)
    sd = {k: v for k, v in sd.items()
          if 'SharpFeature.weight' not in k and 'SharpOri.weight' not in k}
    m.load_state_dict(sd, strict=False)
    m.eval()
    print(f"[info] {os.path.basename(WEIGHTS)} loaded on {dev}")
    return m, dev, torch


def taper(size, t):
    t = int(max(1, min(t, size // 2)))
    r = np.sin(np.linspace(0, np.pi / 2, t, dtype=np.float32)) ** 2
    line = np.ones(size, np.float32); line[:t] = r; line[-t:] = r[::-1]
    return np.maximum(np.outer(line, line).astype(np.float32), 1e-4)


def positions(total, patch, stride):
    if total <= patch: return [0]
    pos = list(range(0, total - patch + 1, stride))
    if pos[-1] != total - patch: pos.append(total - patch)
    return pos


def to_prob(chw):
    s = chw.sum(0)
    if chw.min() >= -1e-3 and chw.max() <= 1 + 1e-3 and np.allclose(s, 1, atol=5e-2):
        return chw
    z = chw - chw.max(0, keepdims=True); e = np.exp(z)
    return e / e.sum(0, keepdims=True)


def infer_frame(model, dev, torch, image):
    H, W = image.shape
    pin = image
    ph, pw = max(0, PATCH - H), max(0, PATCH - W)
    if ph or pw:
        pin = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
    pH, pW = pin.shape
    win = taper(PATCH, OVERLAP); stride = PATCH - OVERLAP
    acc = np.zeros((N_CLASSES, pH, pW), np.float32)
    wsum = np.zeros((pH, pW), np.float32)
    with torch.no_grad():
        for y in positions(pH, PATCH, stride):
            for x in positions(pW, PATCH, stride):
                p = pin[y:y + PATCH, x:x + PATCH]
                t = torch.from_numpy(np.ascontiguousarray(p)).float()[None, None].to(dev)
                pr = to_prob(model(t)[0].cpu().numpy())
                acc[:, y:y + PATCH, x:x + PATCH] += pr * win[None]
                wsum[y:y + PATCH, x:x + PATCH] += win
    acc /= np.maximum(wsum[None], 1e-8)
    return acc[1][:H, :W]


def build_mask(src, tat_gdf, hydro_gdf):
    """valid = inside tätort AND not water."""
    shape = (src.height, src.width)
    if tat_gdf is None:
        urban = np.ones(shape, np.uint8)
    else:
        t = tat_gdf.to_crs(src.crs) if tat_gdf.crs != src.crs else tat_gdf
        b = src.bounds
        sub = t.cx[b.left:b.right, b.bottom:b.top]
        shp = [(g, 1) for g in sub.geometry if g is not None and not g.is_empty]
        urban = (rasterize(shp, out_shape=shape, transform=src.transform, fill=0,
                           dtype=np.uint8, all_touched=True)
                 if shp else np.zeros(shape, np.uint8))
    water = np.zeros(shape, np.uint8)
    if hydro_gdf is not None:
        h = hydro_gdf.to_crs(src.crs) if hydro_gdf.crs != src.crs else hydro_gdf
        b = src.bounds
        sub = h.cx[b.left:b.right, b.bottom:b.top]
        shp = [(g, 1) for g in sub.geometry if g is not None and not g.is_empty]
        if shp:
            water = rasterize(shp, out_shape=shape, transform=src.transform, fill=0,
                              dtype=np.uint8, all_touched=True)
    return (urban > 0) & (water == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["malmo", "gtb", "sth"])
    ap.add_argument("--era", choices=sorted({e for _, e in SOURCES}))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--limit", type=int, default=0, help="process only N frames (testing)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'city':>6} {'era':>5}  {'frames':>6}  source")
        for (c, e), (d, rgb) in sorted(SOURCES.items()):
            p = os.path.join(HERE, d)
            n = len([f for f in glob.glob(os.path.join(p, "*.tif"))
                     if not f.endswith(".aux.xml")]) if os.path.isdir(p) else 0
            print(f"{c:>6} {e:>5}  {n:>6}  {d}{'  [RGB]' if rgb else ''}")
        return

    if not args.city or not args.era:
        sys.exit("specify --city and --era, or --list")
    key = (args.city, args.era)
    if key not in SOURCES:
        sys.exit(f"no source configured for {key}")
    rel, is_rgb = SOURCES[key]
    src_dir = os.path.join(HERE, rel)
    if not os.path.isdir(src_dir):
        sys.exit(f"[STOP] missing {src_dir}")

    out_dir = os.path.join(OUT_ROOT, f"{args.city}_{args.era}")
    os.makedirs(out_dir, exist_ok=True)

    import geopandas as gpd
    tat = gpd.read_file(TATORT, layer=TATORT_LAYER) if os.path.exists(TATORT) else None
    if tat is None:
        print(f"[WARN] no tätort at {TATORT} -- study-area mask DISABLED")
    hydro_path = HYDRO.get(args.city, "")
    hydro = gpd.read_file(hydro_path) if os.path.exists(hydro_path) else None
    if hydro is None:
        print(f"[WARN] no hydrography at {hydro_path} -- water mask DISABLED; "
              f"harbour water may appear as canopy")

    model, dev, torch = load_model()

    frames = sorted(f for f in glob.glob(os.path.join(src_dir, "*.tif"))
                    if not f.endswith(".aux.xml"))
    if args.limit:
        frames = frames[:args.limit]
    print(f"[info] {args.city} {args.era}: {len(frames)} frames, "
          f"threshold {args.threshold}{', RGB->gray' if is_rgb else ''}\n")

    done, skipped, t0 = 0, [], time.time()
    tot_canopy, tot_valid = 0, 0

    for fp in frames:
        stem = os.path.splitext(os.path.basename(fp))[0]
        out_bin = os.path.join(out_dir, f"{stem}_canopy.tif")
        if os.path.exists(out_bin):
            print(f"  {stem:>28}  exists -> skip"); continue
        try:
            with rasterio.open(fp) as src:
                if is_rgb:
                    if src.count < 3:
                        raise ValueError(f"expected 3 bands, got {src.count}")
                    img = rgb_to_gray(src.read([1, 2, 3]))
                else:
                    img = src.read(1).astype(np.float32)
                nodv = 0 if src.nodata is None else src.nodata
                valid = (img != nodv) & build_mask(src, tat, hydro)
                transform, crs = src.transform, src.crs
                H, W = src.height, src.width
        except Exception as e:
            skipped.append((stem, type(e).__name__))
            print(f"  {stem:>28}  UNREADABLE ({type(e).__name__}) -> skipped")
            continue

        prob = infer_frame(model, dev, torch, img)
        prob[~valid] = 0.0
        binary = ((prob >= args.threshold) & valid).astype(np.uint8)

        nvalid = int(valid.sum()); ncan = int(binary.sum())
        tot_canopy += ncan; tot_valid += nvalid
        pct = 100 * ncan / max(nvalid, 1)

        base = dict(driver="GTiff", height=H, width=W, count=1,
                    crs=crs, transform=transform, compress="lzw")
        with rasterio.open(os.path.join(out_dir, f"{stem}_prob.tif"), "w",
                           dtype=rasterio.float32, nodata=-1.0, **base) as d:
            o = prob.astype(np.float32); o[~valid] = -1.0
            d.write(o, 1)
        with rasterio.open(out_bin, "w", dtype=rasterio.uint8, nodata=0, **base) as d:
            d.write(binary, 1)

        done += 1
        print(f"  {stem:>28}  canopy {pct:5.1f}% of {100*nvalid/(H*W):4.1f}% valid area")

    dt = time.time() - t0
    print(f"\n[done] {done} frames in {dt/60:.1f} min -> {out_dir}")
    if tot_valid:
        print(f"[info] {args.city} {args.era} canopy cover: "
              f"{100*tot_canopy/tot_valid:.2f}% of masked study area")
    if skipped:
        print(f"[warn] {len(skipped)} frames unreadable and excluded:")
        for s, e in skipped:
            print(f"         {s} ({e})")
        print("       These are gaps in coverage -- record them with the results.")


if __name__ == "__main__":
    main()
