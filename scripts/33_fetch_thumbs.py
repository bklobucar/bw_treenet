#!/usr/bin/env python3
"""33_fetch_thumbs.py -- download the public thumbnails (~20 KB each) of one flight per city, so leaf
status can be judged by eye for flights the catalogue dates only by year. Uses the cache written by
32_leafon_scan.py; no login needed.

    python3 scripts/33_fetch_thumbs.py malmo:orto-historiska-2000-2004:2004 gtb:orto-b-2009:2009
Output: results/leafon_scan/thumbs_check/<city>_<collection>_<year>/<tile>.jpg
"""
import csv, glob, json, os, sys, urllib.request

BASE = "results/leafon_scan"
city_of = {}
with open(os.path.join(BASE, "picks.csv")) as f:
    for r in csv.DictReader(f):
        city_of[r["sheet"]] = r["city"]

for arg in sys.argv[1:]:
    city, coll, year = arg.split(":")
    out = os.path.join(BASE, "thumbs_check", f"{city}_{coll}_{year}")
    os.makedirs(out, exist_ok=True)
    seen, n = set(), 0
    for path in sorted(glob.glob(os.path.join(BASE, "cache", "*.json"))):
        sheet = os.path.basename(path)[:-5]
        if city_of.get(sheet) != city:
            continue
        for it in json.load(open(path)):
            if it["collection"] != coll or str(it["year"]) != year or it["spec"] == "cir":
                continue
            if it["id"] in seen or not it.get("thumb"):
                continue
            seen.add(it["id"])
            dst = os.path.join(out, f"{it['id']}.jpg")
            if not os.path.exists(dst):
                try:
                    urllib.request.urlretrieve(it["thumb"], dst); n += 1
                except Exception as e:
                    print(f"  failed {it['id']}: {e}")
    print(f"{arg}: {len(seen)} tiles, {n} downloaded -> {out}")
