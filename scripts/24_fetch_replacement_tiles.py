"""
24_fetch_replacement_tiles.py -- replace the six corrupt source frames with fresh
copies from Lantmateriet's STAC catalogue (api.lantmateriet.se/stac-bild/v1).

Why this is more than a download: the catalogue serves historical orthophotos as
COGs on Lantmateriet's own 5 km grid in RT90 (EPSG:3021), under different file
names. The pipeline's frames are SWEREF 99 TM (EPSG:3006) on a different 5 km
grid. So each missing frame is rebuilt by warping the 1-6 overlapping RT90 tiles
onto the exact extent and 0.5 m grid of the frame it replaces.

Stages (every stage is resumable, finished work is skipped):
  1. work out the SWEREF extent of each missing frame from its file name, after
     checking the naming convention against frames that survived
  2. search the catalogue (public, no login) for tiles from the right flight year
  3. download them (needs a Geotorget login: GEOTORGET_USER / GEOTORGET_PASS)
  4. warp to EPSG:3006 at 0.5 m on the exact frame extent, then fully decode the
     result to prove it is not corrupt
  5. with --install: move the corrupt originals (and their sidecar files) to
     data/raw_corrupt/ and put the new frames in their place

Run with the `gis` conda env active (needs GDAL's Python bindings, not torch):
    python scripts/24_fetch_replacement_tiles.py --plan      # stages 1-2 only
    python scripts/24_fetch_replacement_tiles.py             # stages 1-4
    python scripts/24_fetch_replacement_tiles.py --install   # stages 1-5
"""
import os, sys, glob, json, time, shutil, base64, argparse
import urllib.request, urllib.parse, urllib.error
import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAC = "https://api.lantmateriet.se/stac-bild/v1"
DL_DIR = os.path.join(HERE, "data", "raw_new", "_downloads")
OUT_DIR = os.path.join(HERE, "data", "raw_new")
BAD_DIR = os.path.join(HERE, "data", "raw_corrupt")
TILE_M = 5000
MAX_NODATA_FRAC = 0.05      # refuse to install a rebuilt frame with more nodata than this

# (city, era, folder, tile code, flight year) -- the six frames that fail a full read
MISSING = [
    ("gtb", "1960", "data/raw/gtb/1960_OF_gray_gtb", "639_32_05", 1963),
    ("gtb", "1960", "data/raw/gtb/1960_OF_gray_gtb", "639_32_50", 1960),
    ("sth", "1960", "data/raw/sth/1960_OF_gray_sth", "657_68_00", 1958),
    ("sth", "1960", "data/raw/sth/1960_OF_gray_sth", "657_68_05", 1958),
    ("sth", "1970", "data/raw/sth/1970_OF_gray_sth", "658_66_55", 1971),
    ("sth", "1970", "data/raw/sth/1970_OF_gray_sth", "658_67_00", 1971),
]


def make_srs(epsg):
    s = osr.SpatialReference()
    s.ImportFromEPSG(epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)   # always x = E/lon, y = N/lat
    return s


S3006, S4326 = make_srs(3006), make_srs(4326)


# ── stage 1: file name -> SWEREF extent ─────────────────────────────────────
def origin(code, hyp):
    """South-west corner (E, N) of a tile code like 639_32_05.
    639 / 32 = 10 km cell (N 6390 km, E 320 km); the two digits of the last group
    are the offset in km inside the cell. Which digit is N and which is E is not
    documented anywhere I could find, so both orders are tested against real frames."""
    a, b, sub = code.split("_")
    n0, e0 = int(a) * 10000, int(b) * 10000
    c, d = int(sub[0]) * 1000, int(sub[1]) * 1000
    return (e0 + (d if hyp == "A" else c), n0 + (c if hyp == "A" else d))


def detect_convention():
    skip = {f"{c}_{y}" for (_, _, _, c, y) in MISSING}
    votes = {"A": 0, "B": 0}
    checked = discriminating = 0
    misses = []
    for folder in sorted({m[2] for m in MISSING}):
        for fp in sorted(glob.glob(os.path.join(HERE, folder, "*.tif"))):
            stem = os.path.splitext(os.path.basename(fp))[0]
            parts = stem.split("_")
            if len(parts) != 4 or not parts[3].isdigit() or stem in skip:
                continue
            try:
                ds = gdal.Open(fp)
                gt = ds.GetGeoTransform()
                left, bottom = gt[0], gt[3] + gt[5] * ds.RasterYSize
                ds = None
            except Exception:
                continue
            code = "_".join(parts[:3])
            ea, na = origin(code, "A")
            eb, nb = origin(code, "B")
            ok_a = abs(left - ea) < 1 and abs(bottom - na) < 1
            ok_b = abs(left - eb) < 1 and abs(bottom - nb) < 1
            checked += 1
            discriminating += (ea, na) != (eb, nb)
            votes["A"] += ok_a
            votes["B"] += ok_b
            if not (ok_a or ok_b) and len(misses) < 5:
                misses.append((stem, left, bottom, (ea, na), (eb, nb)))
    print(f"[stage 1] naming check on {checked} surviving frames: "
          f"hypothesis A {votes['A']}, hypothesis B {votes['B']}")
    for hyp, other in (("A", "B"), ("B", "A")):
        if checked and votes[hyp] == checked and votes[other] < checked and discriminating:
            print(f"[stage 1] convention {hyp} matches every frame")
            return hyp
    print("[error] could not establish the tile-name convention. Examples of frames that fit neither:")
    for m in misses:
        print("   ", m)
    sys.exit(1)


# ── stage 2: catalogue search ───────────────────────────────────────────────
def get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/geo+json, application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def wgs_bbox(bbox, pad=0.003):
    tr = osr.CoordinateTransformation(S3006, S4326)
    x0, y0, x1, y1 = bbox
    pts = [tr.TransformPoint(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    return (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)


def search(bbox, year):
    params = {"bbox": ",".join(f"{v:.5f}" for v in wgs_bbox(bbox)),
              "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
              "limit": 50}
    url = STAC + "/search?" + urllib.parse.urlencode(params, safe=",:/")
    found = {}
    for _ in range(10):
        js = get_json(url)
        feats = js.get("features", [])
        for f in feats:
            found[f["id"]] = f
        nxt = [l["href"] for l in js.get("links", []) if l.get("rel") == "next"]
        if not feats or not nxt:
            break
        url = nxt[0]
    return list(found.values())


def footprint_3006(item):
    props, asset = item.get("properties", {}), item["assets"]["data"]
    bb = props.get("proj:bbox") or asset.get("proj:bbox")
    code = props.get("proj:code") or asset.get("proj:code")
    x0, y0, x1, y1 = bb
    n = 50
    pts = ([(x0 + (x1 - x0) * i / n, y0) for i in range(n)]
           + [(x1, y0 + (y1 - y0) * i / n) for i in range(n)]
           + [(x1 - (x1 - x0) * i / n, y1) for i in range(n)]
           + [(x0, y1 - (y1 - y0) * i / n) for i in range(n)])
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in pts:
        ring.AddPoint_2D(x, y)
    ring.CloseRings()
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    poly.AssignSpatialReference(make_srs(int(str(code).split(":")[1])))
    poly.TransformTo(S3006)
    return poly


def target_polygon(bbox):
    x0, y0, x1, y1 = bbox
    return ogr.CreateGeometryFromWkt(
        f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))")


def choose_items(tile):
    target = target_polygon(tile["bbox"])
    chosen, union = [], None
    for it in search(tile["bbox"], tile["year"]):
        p = it.get("properties", {})
        if int(p.get("flygar", 0)) != tile["year"] or p.get("spektraltyp") != "pan":
            continue
        fp = footprint_3006(it)
        if fp.Intersection(target).GetArea() > 1.0:
            chosen.append(it)
            union = fp if union is None else union.Union(fp)
    cover = union.Intersection(target).GetArea() / target.GetArea() if union else 0.0
    return chosen, cover


# ── stage 3: download ───────────────────────────────────────────────────────
def auth_header():
    user, pw = os.environ.get("GEOTORGET_USER"), os.environ.get("GEOTORGET_PASS")
    if not user or not pw:
        sys.exit("[error] set GEOTORGET_USER and GEOTORGET_PASS in this shell first.")
    tok = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": "Basic " + tok}


def download(item, hdr):
    a = item["assets"]["data"]
    size = a.get("file:size")
    dest = os.path.join(DL_DIR, item["id"] + ".tif")
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return dest
    tmp = dest + ".part"
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            req = urllib.request.Request(a["href"], headers=hdr)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                if "html" in r.headers.get("Content-Type", "").lower():
                    raise RuntimeError("received a web page, not a TIFF (login problem?)")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if size and os.path.getsize(tmp) != size:
                raise RuntimeError(f"size {os.path.getsize(tmp)} != expected {size}")
            os.replace(tmp, dest)
            print(f"   downloaded {item['id']}  {os.path.getsize(dest) / 1e6:.0f} MB "
                  f"in {time.time() - t0:.0f} s", flush=True)
            return dest
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(f"[error] HTTP {e.code} from {a['href']}\n"
                         "        The login was rejected, or this account has no permission for the\n"
                         "        product. Check the username and password, or use the browser route.")
            print(f"   attempt {attempt} failed: HTTP {e.code}", flush=True)
        except Exception as e:
            print(f"   attempt {attempt} failed: {e}", flush=True)
        time.sleep(5 * attempt)
    sys.exit(f"[error] could not download {item['id']}")


# ── stage 4: warp and verify ────────────────────────────────────────────────
def build_tile(tile, srcs, out_path):
    for s in srcs:
        d = gdal.Open(s)
        b = d.GetRasterBand(1)
        if d.RasterCount != 1 or b.DataType != gdal.GDT_Byte:
            print(f"   [warn] {os.path.basename(s)}: {d.RasterCount} band(s), "
                  f"type {gdal.GetDataTypeName(b.DataType)} -- expected 1 x Byte")
        d = None
    opts = gdal.WarpOptions(
        format="GTiff", dstSRS="EPSG:3006", outputBounds=tile["bbox"],
        xRes=0.5, yRes=0.5, resampleAlg="bilinear", srcNodata=0, dstNodata=0,
        outputType=gdal.GDT_Byte, multithread=True,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"])
    tmp = out_path + ".tmp.tif"
    ds = gdal.Warp(tmp, srcs, options=opts)
    ds = None                       # close and flush before the rename
    os.replace(tmp, out_path)


def verify(path):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray()     # decodes every block; raises if corrupt
    nz = arr[arr > 0]
    return (ds.RasterXSize, ds.RasterYSize, float((arr == 0).mean()),
            float(nz.mean()) if nz.size else 0.0)


# ── stage 5: install ────────────────────────────────────────────────────────
def install(tile, new_path):
    stem = f"{tile['code']}_{tile['year']}"
    dst_dir = os.path.join(HERE, tile["folder"])
    bad_dir = os.path.join(BAD_DIR, os.path.basename(tile["folder"]))
    os.makedirs(bad_dir, exist_ok=True)
    for f in glob.glob(os.path.join(dst_dir, stem + ".*")):     # tif plus sidecars (.ovr, .aux.xml, .tab, .tfw)
        shutil.move(f, os.path.join(bad_dir, os.path.basename(f)))
    shutil.copy2(new_path, os.path.join(dst_dir, stem + ".tif"))
    print(f"   installed {stem}.tif in {tile['folder']}  (old files moved to {bad_dir})")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="stages 1-2 only, no login needed")
    ap.add_argument("--install", action="store_true", help="also stage 5")
    ap.add_argument("--force", action="store_true", help="install even if a frame has much nodata")
    args = ap.parse_args()

    hyp = detect_convention()
    tiles = []
    for city, era, folder, code, year in MISSING:
        e, n = origin(code, hyp)
        tiles.append(dict(city=city, era=era, folder=folder, code=code, year=year,
                          bbox=(e, n, e + TILE_M, n + TILE_M)))
    print("\n[stage 1] extents in EPSG:3006 (E min, N min, E max, N max):")
    for t in tiles:
        print(f"   {t['code']}_{t['year']}  {t['bbox']}")

    print("\n[stage 2] searching the catalogue ...")
    try:
        for t in tiles:
            t["items"], cover = choose_items(t)
            ids = ", ".join(f"{i['id']} ({i['properties'].get('upplosning')} m, "
                            f"{i['collection']})" for i in t["items"]) or "NONE"
            print(f"   {t['code']}_{t['year']}: {len(t['items'])} tile(s), "
                  f"covering {100 * cover:.2f}% of the frame -> {ids}")
            if not t["items"]:
                print("      [warn] nothing found for this frame and flight year")
            elif cover < 0.999:
                print("      [warn] the catalogue does not fully cover this frame; "
                      "the gap will be nodata")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"[error] the catalogue request failed: {e}\n"
                 "        If this says the host cannot be reached, this machine has no internet "
                 "access.\n        The extents printed above are still valid; send them back and "
                 "the tiles can be fetched another way.")

    unique = {i["id"]: i for t in tiles for i in t["items"]}
    total = sum(i["assets"]["data"].get("file:size", 0) for i in unique.values())
    print(f"\n   {len(unique)} distinct files, {total / 1e9:.1f} GB to download")
    if args.plan:
        print("[plan only] stopping here.")
        return

    print("\n[stage 3] downloading ...")
    os.makedirs(DL_DIR, exist_ok=True)
    hdr = auth_header()
    local = {iid: download(it, hdr) for iid, it in unique.items()}

    print("\n[stage 4] warping to EPSG:3006 at 0.5 m and verifying ...")
    results = []
    for t in tiles:
        out = os.path.join(OUT_DIR, f"{t['city']}_{t['era']}", f"{t['code']}_{t['year']}.tif")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not t["items"]:
            print(f"   {t['code']}_{t['year']}: no source tiles found, skipped")
            continue
        if not os.path.exists(out):
            build_tile(t, [local[i["id"]] for i in t["items"]], out)
        w, h, zero, mean = verify(out)
        flag = "" if (w, h) == (10000, 10000) else "   <-- unexpected size"
        print(f"   {t['code']}_{t['year']}: {w}x{h}, {100 * zero:.2f}% nodata, "
              f"mean grey {mean:.0f}{flag}", flush=True)
        results.append((t, out, zero, (w, h) == (10000, 10000)))

    if args.install:
        print("\n[stage 5] installing ...")
        for t, out, zero, size_ok in results:
            if (zero > MAX_NODATA_FRAC or not size_ok) and not args.force:
                print(f"   {t['code']}_{t['year']}: NOT installed (nodata {100 * zero:.1f}% or "
                      f"wrong size); rerun with --force if that is expected")
                continue
            install(t, out)
        print("\nNext, in the bwtreenet env, process only the new frames (the rest are skipped):")
        print("   python scripts/19_production_inference.py --city gtb --era 1960")
        print("   python scripts/19_production_inference.py --city sth --era 1960")
        print("   python scripts/19_production_inference.py --city sth --era 1970")
        print("then rerun scripts/20_aggregate_canopy.py and 22_build_usb_package.sh.")
    else:
        print(f"\nRebuilt frames are in {OUT_DIR}. Check them, then rerun with --install.")


if __name__ == "__main__":
    main()
