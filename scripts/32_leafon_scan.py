#!/usr/bin/env python3
"""32_leafon_scan.py -- which post-1999 Lantmateriet orthophoto flights are LEAF-ON over your sheets,
and which flight to use per city for three new eras. Downloads no imagery and needs no login.

For every 5 x 5 km sheet code NNN_EE_ab found under data/raw (union over all eras, per city) it asks the
public STAC search for all orthophoto tiles from 1999 onward, measures how much of the sheet each flight
covers, reads the flight dates, and classifies each flight:

    on        every photo date between --on-from and --on-to  (default 05-25 .. 09-15)
    marginal  within --marg-from .. --marg-to but not fully "on" (default 05-10 .. 09-30)
    off       any photo date outside the marginal window (April flights, early May)
    unknown   catalogue gives only the year (most 2000-2009 collections) -> check thumbnails

Then, per city and era window, it picks one main flight (most sheets fully covered and leaf-on,
prefers ~0.5 m and RGB/pan over very fine 0.16 m) and gap-fills uncovered sheets from other flights
in the same window.

Outputs (in --out, default results/leafon_scan/):
    flights.csv           one row per city + collection: dates, leaf status, resolution, sheets covered, GB
    picks.csv             one row per city + era + sheet: chosen collection, coverage, leaf status, tiles
    tiles_<era>.txt       "collection tile_id  # sheet" lines for fetch_tiles.sh
    thumbs/               one thumbnail per unknown-date pick, for a quick visual leaf check
    cache/<sheet>.json    raw search answers (rerun with --offline to re-plan without the network)

Standard library only. Run on a compute node (it reached the catalogue before) or on HomePC:
    python scripts/32_leafon_scan.py --root .
    python3 32_leafon_scan.py --sheets-file sheets.txt        # lines ending NNN_EE_ab_YYYY.tif, city in path
"""
import argparse, csv, glob, json, math, os, re, sys, time, urllib.parse, urllib.request
from collections import defaultdict

STAC = "https://api.lantmateriet.se/stac-bild/v1/search"
SHEET_RE = re.compile(r"(\d{3})_(\d{2})_(\d{2})_\d{4}\.tif$")
CITIES = ("malmo", "gtb", "sth")

# ---------------------------------------------------------------- SWEREF 99 TM <-> WGS84 (Gauss-Krueger, GRS80)
_A, _F = 6378137.0, 1 / 298.257222101
_K0, _L0, _FE = 0.9996, math.radians(15.0), 500000.0
_E2 = _F * (2 - _F); _N = _F / (2 - _F)
_AH = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64)


def sweref_to_wgs84(E, N):
    xi = N / (_K0 * _AH); eta = (E - _FE) / (_K0 * _AH)
    d1 = _N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96 - _N**4 / 360
    d2 = _N**2 / 48 + _N**3 / 15 - 437 * _N**4 / 1440
    d3 = 17 * _N**3 / 480 - 37 * _N**4 / 840
    d4 = 4397 * _N**4 / 161280
    xp = xi - sum(d * math.sin(2 * k * xi) * math.cosh(2 * k * eta) for k, d in enumerate((d1, d2, d3, d4), 1))
    ep = eta - sum(d * math.cos(2 * k * xi) * math.sinh(2 * k * eta) for k, d in enumerate((d1, d2, d3, d4), 1))
    phis = math.asin(math.sin(xp) / math.cosh(ep))
    dl = math.atan(math.sinh(ep) / math.cos(xp))
    e2, e4, e6, e8 = _E2, _E2**2, _E2**3, _E2**4
    As = e2 + e4 + e6 + e8
    Bs = -(7 * e4 + 17 * e6 + 30 * e8) / 6
    Cs = (224 * e6 + 889 * e8) / 120
    Ds = -(4279 * e8) / 1260
    s2 = math.sin(phis)**2
    phi = phis + math.sin(phis) * math.cos(phis) * (As + Bs * s2 + Cs * s2**2 + Ds * s2**3)
    return math.degrees(_L0 + dl), math.degrees(phi)


def wgs84_to_sweref(lon, lat):
    e2 = _E2
    A = e2; B = (5 * e2**2 - e2**3) / 6; C = (104 * e2**3 - 45 * e2**4) / 120; D = (1237 * e2**4) / 1260
    phi = math.radians(lat); s2 = math.sin(phi)**2
    phis = phi - math.sin(phi) * math.cos(phi) * (A + B * s2 + C * s2**2 + D * s2**3)
    dl = math.radians(lon) - _L0
    xip = math.atan(math.tan(phis) / math.cos(dl))
    etap = math.atanh(math.cos(phis) * math.sin(dl))
    b1 = _N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16 + 41 * _N**4 / 180
    b2 = 13 * _N**2 / 48 - 3 * _N**3 / 5 + 557 * _N**4 / 1440
    b3 = 61 * _N**3 / 240 - 103 * _N**4 / 140
    b4 = 49561 * _N**4 / 161280
    x = xip + sum(b * math.sin(2 * k * xip) * math.cosh(2 * k * etap) for k, b in enumerate((b1, b2, b3, b4), 1))
    y = etap + sum(b * math.cos(2 * k * xip) * math.sinh(2 * k * etap) for k, b in enumerate((b1, b2, b3, b4), 1))
    return _FE + _K0 * _AH * y, _K0 * _AH * x


# ---------------------------------------------------------------- geometry helpers
def clip_rect(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clip of a polygon (list of (x, y)) to an axis-aligned rectangle."""
    def clip(pts, inside, inter):
        out = []
        for i, p in enumerate(pts):
            q = pts[i - 1]
            if inside(p):
                if not inside(q):
                    out.append(inter(q, p))
                out.append(p)
            elif inside(q):
                out.append(inter(q, p))
        return out

    def ix(xc):
        return lambda a, b: (xc, a[1] + (b[1] - a[1]) * (xc - a[0]) / (b[0] - a[0]))

    def iy(yc):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yc - a[1]) / (b[1] - a[1]), yc)

    pts = poly
    for inside, inter in ((lambda p: p[0] >= x0, ix(x0)), (lambda p: p[0] <= x1, ix(x1)),
                          (lambda p: p[1] >= y0, iy(y0)), (lambda p: p[1] <= y1, iy(y1))):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return pts


def area(pts):
    return abs(sum(pts[i - 1][0] * p[1] - p[0] * pts[i - 1][1] for i, p in enumerate(pts))) / 2


def sheet_box(code):
    nnn, ee, ab = code.split("_")
    n0 = int(nnn) * 10_000 + int(ab[0]) * 1000
    e0 = int(ee) * 10_000 + int(ab[1]) * 1000
    return e0, n0, e0 + 5000, n0 + 5000


def item_polygon_sweref(it):
    pa = it.get("assets", {}).get("data", {})
    if pa.get("proj:code") == "EPSG:3006" and pa.get("proj:bbox"):
        x0, y0, x1, y1 = pa["proj:bbox"]
        return [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    ring = it["geometry"]["coordinates"][0]
    pts = [wgs84_to_sweref(lon, lat) for lon, lat in ring]
    return pts[:-1] if pts[0] == pts[-1] else pts


# ---------------------------------------------------------------- catalogue
def http_json(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def search_sheet(code, cache_dir, offline, since):
    path = os.path.join(cache_dir, f"{code}.json")
    if os.path.exists(path):
        return json.load(open(path))
    if offline:
        raise SystemExit(f"--offline but no cache for {code}")
    x0, y0, x1, y1 = sheet_box(code)
    inset = 50  # metres; avoids catching neighbours that only touch the edge
    lo0, la0 = sweref_to_wgs84(x0 + inset, y0 + inset)
    lo1, la1 = sweref_to_wgs84(x1 - inset, y1 - inset)
    lo2, la2 = sweref_to_wgs84(x0 + inset, y1 - inset)
    lo3, la3 = sweref_to_wgs84(x1 - inset, y0 + inset)
    bbox = [min(lo0, lo2), min(la0, la3), max(lo1, lo3), max(la1, la2)]
    q = urllib.parse.urlencode({"bbox": ",".join(f"{v:.6f}" for v in bbox),
                                "datetime": f"{since}-01-01T00:00:00Z/..", "limit": 200})
    url, feats, pages = f"{STAC}?{q}", [], 0
    while url and pages < 50:
        page = http_json(url); pages += 1
        feats += page.get("features", [])
        url = next((l["href"] for l in page.get("links", []) if l.get("rel") == "next"), None)
    slim = []
    for it in feats:
        p, d = it["properties"], it.get("assets", {}).get("data", {})
        slim.append(dict(id=it["id"], collection=it["collection"], year=p.get("flygar"),
                         start=p.get("start_datetime") or p.get("datetime"),
                         end=p.get("end_datetime") or p.get("datetime"),
                         res=p.get("upplosning"), spec=p.get("spektraltyp"), size=d.get("file:size", 0),
                         poly=item_polygon_sweref(it),
                         thumb=it.get("assets", {}).get("thumbnail", {}).get("href")))
    with open(path, "w") as f:
        json.dump(slim, f)
    return slim


# ---------------------------------------------------------------- leaf status
def mmdd(s):
    return int(s[5:7]) * 100 + int(s[8:10])


def year_only(it):
    s, e = it["start"], it["end"]
    return s[5:10] == "01-01" and s[11:19] == "00:00:00" and (e[5:10] in ("12-31", "01-01"))


def leaf_status(it, a):
    if year_only(it):
        return "unknown"
    s, e = mmdd(it["start"]), mmdd(it["end"])
    if it["start"][:4] != it["end"][:4]:
        return "unknown"
    if a.on_from <= s and e <= a.on_to:
        return "on"
    if a.marg_from <= s and e <= a.marg_to:
        return "marginal"
    return "off"


RANK = {"on": 0, "marginal": 1, "unknown": 2, "off": 3}


def worst(statuses):
    return max(statuses, key=lambda s: RANK[s]) if statuses else "unknown"


def res_penalty(r):
    # 0.5 m is what the model runs at; 0.25 is fine; 0.16 is 9x the download; 1.0 loses detail
    return {0.5: 0, 0.25: 1, 0.16: 3, 1.0: 2}.get(r, 2)


# ---------------------------------------------------------------- main
def parse_mmdd(s):
    m, d = s.split("-")
    return int(m) * 100 + int(d)


def parse_eras(s):
    out = []
    for part in s.split(","):
        name, rng = part.split(":")
        y0, y1 = rng.split("-")
        out.append((name, int(y0), int(y1)))
    return out


def find_sheets(a):
    sheets = defaultdict(set)
    if a.sheets_file:
        lines = open(a.sheets_file).read().split()
    else:
        lines = glob.glob(os.path.join(a.root, "data", "raw", "*", "*OF_*", "*.tif"))
    for ln in lines:
        m = SHEET_RE.search(ln)
        if not m:
            continue
        city = next((c for c in CITIES if f"/{c}/" in ln.replace("\\", "/")), None)
        if city:
            sheets[city].add("_".join(m.groups()))
    return sheets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--sheets-file")
    ap.add_argument("--out", default="results/leafon_scan")
    ap.add_argument("--eras", default="2000:1999-2005,2010:2008-2013,2020:2019-2025",
                    help="name:firstyear-lastyear,... (default %(default)s)")
    ap.add_argument("--on-from", default="05-25"); ap.add_argument("--on-to", default="09-15")
    ap.add_argument("--marg-from", default="05-10"); ap.add_argument("--marg-to", default="09-30")
    ap.add_argument("--min-cover", type=float, default=99.0, help="%% of sheet that counts as covered")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-thumbs", action="store_true")
    a = ap.parse_args()
    for k in ("on_from", "on_to", "marg_from", "marg_to"):
        setattr(a, k, parse_mmdd(getattr(a, k)))
    eras = parse_eras(a.eras)
    since = min(y0 for _, y0, _ in eras)

    sheets = find_sheets(a)
    if not sheets:
        sys.exit("no sheets found: run from the bw_treenet root or give --sheets-file")
    cache = os.path.join(a.out, "cache"); os.makedirs(cache, exist_ok=True)

    # coverage[city][sheet][collection] = dict(cover%, tiles, size, statuses, dates, res, spec, thumbs)
    cov = defaultdict(lambda: defaultdict(dict))
    total = sum(len(v) for v in sheets.values()); done = 0
    for city in CITIES:
        for code in sorted(sheets.get(city, ())):
            done += 1
            print(f"\r[{done}/{total}] {city} {code}      ", end="", flush=True)
            items = search_sheet(code, cache, a.offline, since)
            x0, y0, x1, y1 = sheet_box(code)
            byc = defaultdict(list)
            for it in items:
                if it["spec"] == "cir":          # we need RGB(I) or pan, not the colour-infrared twin
                    continue
                clipped = clip_rect(it["poly"], x0, y0, x1, y1)
                if len(clipped) >= 3 and area(clipped) > 1e3:
                    byc[f'{it["collection"]}@{it["year"]}'].append((it, clipped))
            for coll, lst in byc.items():
                # union of clipped tiles ~ sum (LM tiles in one collection do not overlap)
                c = min(100.0, 100 * sum(area(p) for _, p in lst) / 25e6)
                its = [i for i, _ in lst]
                cov[city][code][coll] = dict(
                    cover=c, tiles=[i["id"] for i in its], size=sum(i["size"] or 0 for i in its),
                    status=worst([leaf_status(i, a) for i in its]),
                    dates=sorted({i["start"][:10] for i in its} | {i["end"][:10] for i in its}),
                    year=max(int(i["year"]) for i in its if i["year"]),
                    res=its[0]["res"], spec=its[0]["spec"], thumb=its[0]["thumb"])
    print()

    # ---- flights.csv + table
    flights = []
    for city in CITIES:
        agg = defaultdict(lambda: dict(full=0, part=0, size=0, st=set(), dates=set(), res=None, spec=None, year=None, seen=set()))
        tile_size = {}
        for code in sheets[city]:
            for it in json.load(open(os.path.join(cache, f"{code}.json"))):
                tile_size[it["id"]] = it["size"] or 0
        for code, cc in cov[city].items():
            for coll, d in cc.items():
                g = agg[coll]
                g["full" if d["cover"] >= a.min_cover else "part"] += 1
                g["size"] += sum(tile_size.get(t, 0) for t in d["tiles"] if t not in g["seen"]); g["seen"].update(d["tiles"])
                g["st"].add(d["status"]); g["dates"].update(d["dates"])
                g["res"], g["spec"], g["year"] = d["res"], d["spec"], d["year"]
        for coll, g in sorted(agg.items(), key=lambda kv: (kv[1]["year"], kv[0])):
            ds = sorted(g["dates"])
            flights.append(dict(city=city, collection=coll.split("@")[0], year=g["year"], res_m=g["res"], spec=g["spec"],
                                leaf=worst(list(g["st"])) if len(g["st"]) == 1 else "mixed:" + "/".join(sorted(g["st"], key=RANK.get)),
                                first_date=ds[0] if ds else "", last_date=ds[-1] if ds else "",
                                sheets_full=g["full"], sheets_partial=g["part"], sheets_total=len(sheets[city]),
                                gb=round(g["size"] / 1e9, 1)))
    with open(os.path.join(a.out, "flights.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flights[0])); w.writeheader(); w.writerows(flights)

    print("\n=== FLIGHTS PER CITY (cir twins left out) ===")
    for city in CITIES:
        print(f"\n{city}  ({len(sheets[city])} sheets)")
        print(f"  {'year':4s} {'collection':28s} {'res':>5s} {'spec':5s} {'dates':23s} {'leaf':18s} {'full':>4s} {'part':>4s} {'GB':>6s}")
        for r in flights:
            if r["city"] != city:
                continue
            dr = r["first_date"] if r["first_date"] == r["last_date"] else f"{r['first_date']}..{r['last_date'][5:]}"
            if r["leaf"] == "unknown" or r["leaf"].startswith("mixed") and "unknown" in r["leaf"]:
                dr = f"{r['year']} (year only)"
            print(f"  {r['year']:4d} {r['collection']:28s} {r['res_m']:5.2f} {r['spec']:5s} {dr:23s} {r['leaf']:18s} "
                  f"{r['sheets_full']:4d} {r['sheets_partial']:4d} {r['gb']:6.1f}")

    # ---- picks per city x era
    tile_size_all = {}
    for city in CITIES:
        for code in sheets[city]:
            for it in json.load(open(os.path.join(cache, f"{code}.json"))):
                tile_size_all[it["id"]] = it["size"] or 0
    picks, summary = [], []
    thumbs = os.path.join(a.out, "thumbs"); os.makedirs(thumbs, exist_ok=True)
    for name, y0, y1 in eras:
        tiles_lines = []
        for city in CITIES:
            cand = defaultdict(lambda: [0, 0, 0, 0, None])   # coll -> [full_on, full_marg, full_unknown, full_any, info]
            for code, cc in cov[city].items():
                for coll, d in cc.items():
                    if not (y0 <= d["year"] <= y1) or d["cover"] < a.min_cover:
                        continue
                    c = cand[coll]
                    c[0] += d["status"] == "on"; c[1] += d["status"] == "marginal"
                    c[2] += d["status"] == "unknown"; c[3] += d["status"] != "off"
                    c[4] = d
            if not cand:
                summary.append((name, city, "-", "NO FLIGHT", 0, 0, len(sheets[city]), 0.0, ""))
                continue
            main_coll = min(cand, key=lambda k: (-cand[k][0], -cand[k][1], -cand[k][3], -cand[k][2],
                                                 res_penalty(cand[k][4]["res"]), k))
            used, size, statuses, seen_t = defaultdict(int), 0, defaultdict(int), set()
            for code in sorted(sheets[city]):
                cc = cov[city].get(code, {})
                opts = [(coll, d) for coll, d in cc.items() if y0 <= d["year"] <= y1 and d["status"] != "off"]
                best = None
                if main_coll in cc and cc[main_coll]["cover"] >= a.min_cover and cc[main_coll]["status"] != "off":
                    best = (main_coll, cc[main_coll])
                elif opts:
                    best = min(opts, key=lambda o: (o[1]["cover"] < a.min_cover, RANK[o[1]["status"]],
                                                    -o[1]["cover"], res_penalty(o[1]["res"]), abs(o[1]["year"] - cand[main_coll][4]["year"])))
                if best is None:
                    picks.append(dict(era=name, city=city, sheet=code, collection="", year="", cover_pct=0,
                                      leaf="none", dates="", tiles=""))
                    statuses["none"] += 1
                    continue
                coll, d = best
                used[coll] += 1; statuses[d["status"]] += 1
                size += sum(tile_size_all.get(t, 0) for t in d["tiles"] if t not in seen_t); seen_t.update(d["tiles"])
                picks.append(dict(era=name, city=city, sheet=code, collection=coll, year=d["year"],
                                  cover_pct=round(d["cover"], 1), leaf=d["status"], dates=";".join(d["dates"]),
                                  tiles=";".join(d["tiles"])))
                for t in d["tiles"]:
                    tiles_lines.append(f"{coll.split('@')[0]} {t}  # {city} {code}")
            info = cand[main_coll][4]
            st = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items(), key=lambda kv: RANK.get(kv[0], 9)))
            fill = ", ".join(f"{k} x{v}" for k, v in used.items() if k != main_coll)
            summary.append((name, city, main_coll, st, used[main_coll], sum(used.values()), len(sheets[city]),
                            size / 1e9, fill))
            if not a.no_thumbs and (info["status"] == "unknown" or cand[main_coll][2]) and info["thumb"]:
                dst = os.path.join(thumbs, f"{name}_{city}_{main_coll}.jpg")
                if not os.path.exists(dst) and not a.offline:
                    try:
                        urllib.request.urlretrieve(info["thumb"], dst)
                    except Exception as e:
                        print(f"  thumbnail failed for {main_coll}: {e}")
        # de-duplicate tiles shared by neighbouring sheets
        seen, uniq = set(), []
        for ln in tiles_lines:
            key = ln
            if key not in seen:
                seen.add(key); uniq.append(ln)
        with open(os.path.join(a.out, f"tiles_{name}.txt"), "w") as f:
            f.write("\n".join(uniq) + "\n")

    with open(os.path.join(a.out, "picks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picks[0])); w.writeheader(); w.writerows(picks)

    print("\n=== PICKS PER ERA (main flight + gap-fill) ===")
    print(f"  {'era':5s} {'city':6s} {'main collection':28s} {'main/used/total':>16s} {'GB':>6s}  leaf status of used sheets; gap-fill")
    for name, city, coll, st, nmain, nused, ntot, gb, fill in summary:
        print(f"  {name:5s} {city:6s} {coll:28s} {nmain:5d}/{nused:3d}/{ntot:3d}    {gb:6.1f}  {st}" + (f"; fill: {fill}" if fill else ""))
    print(f"\nWrote flights.csv, picks.csv, tiles_<era>.txt and thumbs/ in {a.out}")


if __name__ == "__main__":
    main()
