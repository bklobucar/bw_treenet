p39 = "scripts/39_common_footprint.py"
s = open(p39).read()
if "def own_footprints" in s:
    print("39: already patched")
else:
    FUNC = r'''def own_footprints(a, tat_ds):
    """Series eras and each partial-coverage era, on exactly the area where that era has imagery."""
    if not a.own_footprint:
        return
    print("\n== series on the footprint of each partial-coverage era")
    rows = []
    for city in CITIES:
        idx = {e: index(city, e) for e in a.series + a.own_footprint}
        if any(not idx[e] for e in a.series):
            continue
        base = set.intersection(*[set(idx[e]) for e in a.series])
        for x in a.own_footprint:
            if not idx[x]:
                continue
            sheets = sorted(base & set(idx[x]))
            if not sheets:
                continue
            area, can = 0.0, {e: 0.0 for e in a.series + [x]}
            for sh in sheets:
                box = sheet_box(sh)
                f = tatort(box, a.res, tat_ds)
                for e in a.series + [x]:
                    f &= warp(idx[e][sh][0], box, a.res, "near") > 0
                if not f.any():
                    continue
                area += f.sum()
                for e in a.series + [x]:
                    can[e] += float(warp(idx[e][sh][1], box, a.res, "average", gdal.GDT_Float32)[f].sum())
            if not area:
                continue
            km2 = area * a.res ** 2 / 1e6
            order = sorted(a.series + [x], key=lambda e: int("".join(ch for ch in e if ch.isdigit())[:4]))
            print(f"{city:6s} on {x} footprint ({len(sheets)} sheets, {km2:.1f} km2): " +
                  "  ".join(f"{e} {100 * can[e] / area:.2f}" for e in order))
            for e in order:
                rows.append(dict(city=city, footprint=x, era=e, area_km2=round(km2, 2),
                                 canopy_pct=round(100 * can[e] / area, 3)))
    if rows:
        with open("results/canopy_own_footprints.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print("wrote results/canopy_own_footprints.csv")


'''
    anchors = ["def diagnose_clips(tat_ds):",
               '    ap.add_argument("--res", type=float, default=1.0)',
               "    common_footprint(a, tat_ds)\n"]
    bad = [x for x in anchors if s.count(x) != 1]
    if bad:
        print("39: anchor not found, NOT patched:", bad)
    else:
        s = s.replace(anchors[0], FUNC + anchors[0])
        s = s.replace(anchors[1], '    ap.add_argument("--own-footprint", nargs="*", default=[])\n' + anchors[1])
        s = s.replace(anchors[2], anchors[2] + "    own_footprints(a, tat_ds)\n")
        open(p39, "w").write(s)
        print("39: patched")

p38 = "scripts/38_eval_1990_labels.py"
s = open(p38).read()
o1 = "    lyr = ogr.Open(shp).GetLayer()\n"
if o1 in s:
    s = s.replace(o1, "    lds = ogr.Open(shp)\n    lyr = lds.GetLayer()\n"); print("38: layer fix applied")
else:
    print("38: layer line not found")
o2 = "        ok = valid & (P != 255)\n"
n2 = r'''        # production writes prob = -1 where it masks (outside 2023 tatort, water); score only where it predicted
        Q = gdal.Warp("", [f.replace("_canopy.tif", "_prob.tif") for f in sel], format="MEM",
                      outputBounds=(x0, y0, x1, y1), xRes=RES, yRes=RES, resampleAlg="near", srcNodata="None",
                      dstNodata=-9, outputType=gdal.GDT_Float32).ReadAsArray()
        ok = valid & (P != 255) & (Q >= 0)
        if not ok.any():
            print(f"  {name:18s} clip lies entirely in production's masked area -- not evaluable"); continue
        print(f"  {name:18s} scoring {100 * ok.sum() / max(1, valid.sum()):.0f}% of clip (inside tatort)")
'''
if o2 in s:
    s = s.replace(o2, n2); print("38: mask fix applied")
else:
    print("38: mask line not found")
open(p38, "w").write(s)
