#!/usr/bin/env python3
"""36_fetch_build_era.py -- download one era's Lantmateriet orthophoto tiles and turn them, sheet by sheet,
into the same 0.5 m single-band grayscale 5 x 5 km sheets the pipeline already uses. Raw tiles are deleted
after each sheet, so peak disk use is one sheet's tiles (<= ~3 GB even for 0.16 m RGBI).

Input: a list written by 32_leafon_scan.py, lines "collection tile_id  # city sheet".
Output: data/raw/<city>/<era>_OF_gray_<suffix>/<sheet>_<flight year>.tif   (Byte, LZW, tiled, nodata 0)
        results/era_build/<era>_log.csv   one row per sheet: tiles, year, nodata %, seconds, status

Grayscale = floor((R + G + B) / 3) after resampling to 0.5 m with NEAREST neighbour. That is exactly what
07_prepare_tiles_all_cities.py did for the modern training tiles (GDAL ReadAsArray with buf_xsize uses
nearest by default, then the band mean, then uint8 truncation). Pan (1-band) tiles are copied through.
RT90 tiles (2004/2005 collections) are reprojected to SWEREF 99 TM in the same gdalwarp call.

Login: your Geotorget user name and password, typed at the prompt, held in memory only, sent as HTTP
Basic auth to dl1.lantmateriet.se. Item metadata comes from the public STAC API (no login).

Needs the GDAL command-line tools (gdalwarp, gdal_translate, gdalinfo): run in the `gis` env on a GPU node.
    python scripts/36_fetch_build_era.py results/leafon_scan/final_era2005.txt --era 2005
    python scripts/36_fetch_build_era.py results/leafon_scan/final_era2005_gtb_comparison.txt --era 2005cmp
    add --dry-run to see what it would do, --only gtb to do one city, --keep-raw to keep the tiles.
Rerunning skips sheets already built, so an interrupted run just continues.
"""
import argparse, base64, csv, getpass, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request
from collections import OrderedDict, defaultdict

STAC = "https://api.lantmateriet.se/stac-bild/v1"
SUFFIX = {"malmo": "mmo", "gtb": "gtb", "sth": "sth"}


def sheet_box(code):
    nnn, ee, ab = code.split("_")
    n0 = int(nnn) * 10_000 + int(ab[0]) * 1000
    e0 = int(ee) * 10_000 + int(ab[1]) * 1000
    return e0, n0, e0 + 5000, n0 + 5000


def read_list(path):
    """-> OrderedDict[(city, sheet)] = [(collection, tile), ...]; tiles shared by sheets appear under each."""
    sheets = OrderedDict()
    for ln in open(path):
        m = re.match(r"\s*(\S+)\s+(\S+)\s+#\s*(\S+)\s+(\d{3}_\d{2}_\d{2})", ln)
        if m:
            coll, tile, city, sheet = m.groups()
            sheets.setdefault((city, sheet), []).append((coll, tile))
    return sheets


def http_json(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def item_info(coll, tile, cache_dir):
    p = os.path.join(cache_dir, f"{coll}__{tile}.json")
    if os.path.exists(p):
        return json.load(open(p))
    it = http_json(f"{STAC}/collections/{coll}/items/{tile}")
    d = it["assets"]["data"]
    info = dict(href=d["href"], size=d.get("file:size", 0), year=it["properties"].get("flygar"),
                spec=it["properties"].get("spektraltyp"), res=it["properties"].get("upplosning"),
                crs=d.get("proj:code"))
    json.dump(info, open(p, "w"))
    return info


def download(url, dst, auth, expect):
    if os.path.exists(dst) and (not expect or os.path.getsize(dst) == expect):
        return
    part = dst + ".part"
    for k in range(5):
        try:
            req = urllib.request.Request(url, headers={"Authorization": auth})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, length=8 << 20)
            if expect and os.path.getsize(part) != expect:
                raise IOError(f"size {os.path.getsize(part)} != {expect}")
            os.replace(part, dst)
            return
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(f"login refused ({e.code}) for {url} -- check user name / password / data-use permission")
            err = e
        except Exception as e:
            err = e
        time.sleep(5 * (k + 1))
    raise IOError(f"download failed after retries: {url}: {err}")


def run(cmd, dry):
    if dry:
        print("    $ " + " ".join(cmd))
        return
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip()[-400:]}")


def nodata_pct(path):
    r = subprocess.run(["gdalinfo", "-stats", "-hist", path], capture_output=True, text=True).stdout
    # histogram of a Byte band: first bucket is value 0 when nodata is not excluded; we instead count valid %
    m = re.search(r"STATISTICS_VALID_PERCENT=([\d.]+)", r)
    return round(100 - float(m.group(1)), 2) if m else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiles_file")
    ap.add_argument("--era", required=True, help="label used in the output folder name, e.g. 2005, 2010, 2020, 2005cmp")
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", choices=list(SUFFIX), help="one city only")
    ap.add_argument("--work", default=None, help="temp dir for raw tiles (default: results/era_build/tmp)")
    ap.add_argument("--resampling", default="near", help="gdalwarp -r (default near, = training tiles)")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="build only the first N sheets (a trial run)")
    ap.add_argument("--max-nodata", type=float, default=5.0, help="warn above this nodata %% (sea sheets are expected)")
    a = ap.parse_args()

    sheets = read_list(a.tiles_file)
    if a.only:
        sheets = OrderedDict((k, v) for k, v in sheets.items() if k[0] == a.only)
    if a.limit:
        sheets = OrderedDict(list(sheets.items())[:a.limit])
    if not sheets:
        sys.exit("no sheets in list")
    base = os.path.join(a.root, "results", "era_build"); os.makedirs(base, exist_ok=True)
    cache = os.path.join(base, "stac_items"); os.makedirs(cache, exist_ok=True)
    work = a.work or os.path.join(base, "tmp"); os.makedirs(work, exist_ok=True)
    logp = os.path.join(base, f"{a.era}_log.csv")

    for tool in ("gdalwarp", "gdal_translate", "gdalinfo"):
        if not a.dry_run and shutil.which(tool) is None:
            sys.exit(f"{tool} not found -- activate the gis env")

    # metadata for every tile (public), so sizes and years are known before any download
    print(f"{len(sheets)} sheets, reading tile metadata ...")
    info = {}
    for tl in sheets.values():
        for coll, tile in tl:
            if (coll, tile) not in info:
                info[(coll, tile)] = item_info(coll, tile, cache)
    total = sum(v["size"] or 0 for v in info.values()) / 1e9
    print(f"{len(info)} distinct tiles, {total:.1f} GB to download in total")

    auth = None
    if not a.dry_run:
        # GT_USER / GT_PASS from the shell (set with `read`, never written to disk) allow chained unattended runs
        user = os.environ.get("GT_USER") or input("Geotorget user name: ").strip()
        pw = os.environ.get("GT_PASS") or getpass.getpass("Geotorget password (not shown, not stored): ")
        auth = "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()
        del pw

    # tiles still needed by later sheets must not be deleted early
    remaining = defaultdict(int)
    for tl in sheets.values():
        for t in tl:
            remaining[t] += 1

    if a.dry_run:
        logp = os.devnull
    new_log = a.dry_run or not os.path.exists(logp)
    logf = open(logp, "a", newline="")
    lw = csv.writer(logf, lineterminator="\n")
    if new_log:
        lw.writerow(["city", "sheet", "year", "collections", "tiles", "res_in", "spec", "nodata_pct", "seconds", "status", "path"])

    for i, ((city, sheet), tl) in enumerate(sheets.items(), 1):
        years = sorted({info[t]["year"] for t in tl})
        year = max(years)
        outdir = os.path.join(a.root, "data", "raw", city, f"{a.era}_OF_gray_{SUFFIX[city]}")
        out = os.path.join(outdir, f"{sheet}_{year}.tif")
        print(f"[{i}/{len(sheets)}] {city} {sheet} <- {len(tl)} tile(s) {','.join(sorted({c for c, _ in tl}))} {year}")
        if os.path.exists(out):
            print("    exists, skipping")
            for t in tl:
                remaining[t] -= 1
            continue
        t0 = time.time()
        os.makedirs(outdir, exist_ok=True)
        try:
            local = []
            for coll, tile in tl:
                d = os.path.join(work, f"{tile}.tif")
                if a.dry_run:
                    print(f"    download {info[(coll, tile)]['href']}  ({(info[(coll, tile)]['size'] or 0)/1e6:.0f} MB)")
                else:
                    download(info[(coll, tile)]["href"], d, auth, info[(coll, tile)]["size"])
                local.append(d)
            x0, y0, x1, y1 = sheet_box(sheet)
            tmp = os.path.join(work, f"_{sheet}_warp.tif")
            run(["gdalwarp", "-overwrite", "-t_srs", "EPSG:3006", "-te", str(x0), str(y0), str(x1), str(y1),
                 "-tr", "0.5", "0.5", "-r", a.resampling, "-dstnodata", "0", "-co", "TILED=YES",
                 "-multi", "-wo", "NUM_THREADS=ALL_CPUS", *local, tmp], a.dry_run)
            spec = info[tl[0]]["spec"]
            if spec == "pan":
                run(["gdal_translate", "-b", "1", "-ot", "Byte", "-a_nodata", "0",
                     "-co", "COMPRESS=LZW", "-co", "TILED=YES", tmp, out], a.dry_run)
            else:
                # floor((R+G+B)/3) as uint8, exactly like the training tiles; 0 stays nodata
                from_py = [sys.executable, "-c",
                           "import sys,numpy as np\nfrom osgeo import gdal\n"
                           "src=gdal.Open(sys.argv[1]);r,g,b=[src.GetRasterBand(k).ReadAsArray().astype(np.float32) for k in (1,2,3)]\n"
                           "gray=np.floor((r+g+b)/3.0).astype(np.uint8)\n"
                           "drv=gdal.GetDriverByName('GTiff');o=drv.Create(sys.argv[2],src.RasterXSize,src.RasterYSize,1,gdal.GDT_Byte,['COMPRESS=LZW','TILED=YES'])\n"
                           "o.SetGeoTransform(src.GetGeoTransform());o.SetProjection(src.GetProjection())\n"
                           "bd=o.GetRasterBand(1);bd.WriteArray(gray);bd.SetNoDataValue(0);o.FlushCache()",
                           tmp, out]
                run(from_py, a.dry_run)
            nd = float("nan") if a.dry_run else nodata_pct(out)
            if not a.dry_run and os.path.exists(tmp):
                os.remove(tmp)
            status = "ok" if (nd != nd or nd <= a.max_nodata) else f"nodata {nd:.1f}%"
            if status != "ok":
                print(f"    WARNING {status} (fine for sea/edge sheets; check otherwise)")
        except Exception as e:
            nd, status = float("nan"), f"FAILED: {e}"
            print(f"    {status}")
            if os.path.exists(out):
                os.remove(out)
        for t in tl:
            remaining[t] -= 1
            p = os.path.join(work, f"{t[1]}.tif")
            if remaining[t] == 0 and not a.keep_raw and not a.dry_run and os.path.exists(p) and not status.startswith("FAILED"):
                os.remove(p)
        lw.writerow([city, sheet, year, ";".join(sorted({c for c, _ in tl})), len(tl), info[tl[0]]["res"], info[tl[0]]["spec"],
                     nd, round(time.time() - t0), status, out])
        logf.flush()
    logf.close()
    print("\n(dry run: nothing downloaded, written or logged)" if a.dry_run else f"\nlog: {logp}")


if __name__ == "__main__":
    main()
