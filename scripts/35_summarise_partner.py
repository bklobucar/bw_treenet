#!/usr/bin/env python3
"""35_summarise_partner.py -- one table describing every canopy GeoPackage in a folder tree
(e.g. the partner's v19_UniSthm run), without loading geometries. Standard library only:
a GeoPackage is an SQLite database, so attributes and gpkg_contents are read directly.

Per file: layer, era values, polygons, tiles, tile size, canopy km2, covered km2 (tiles x tile area),
canopy % of covered area, smallest polygon (-> pixel size used), lowest max_conf (-> threshold used),
bounding box, and when it was written.

    python3 ~/bw_treenet/scripts/35_summarise_partner.py ~/bw_treenet/v19_UniSthm
Writes <folder>/partner_summary.csv as well.
"""
import csv, glob, math, os, sqlite3, sys
from collections import Counter

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/bw_treenet/v19_UniSthm")
rows = []
for path in sorted(glob.glob(os.path.join(root, "**", "*.gpkg"), recursive=True)):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    for table, x0, y0, x1, y1, changed in cur.execute(
            "SELECT table_name, min_x, min_y, max_x, max_y, last_change FROM gpkg_contents "
            "WHERE data_type='features'").fetchall():
        cols = {r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')}
        n, area, amin, cmin = cur.execute(
            f'SELECT COUNT(*), SUM(area_m2), MIN(area_m2), MIN(max_conf) FROM "{table}"').fetchone()
        tiles = Counter(r[0] for r in cur.execute(f'SELECT tile FROM "{table}"')) if "tile" in cols else Counter()
        eras = sorted({r[0] for r in cur.execute(f'SELECT DISTINCT era FROM "{table}"')}) if "era" in cols else []
        # tile codes: NNN_EE_xxyy -> 2.5 km, NNN_EE_ab -> 5 km
        sizes = Counter(len(t.split("_")[-1]) for t in tiles) if tiles else Counter()
        side_km = {4: 2.5, 2: 5.0}.get(sizes.most_common(1)[0][0], float("nan")) if sizes else float("nan")
        covered = len(tiles) * side_km**2
        canopy_km2 = (area or 0) / 1e6
        rows.append(dict(
            city=os.path.basename(os.path.dirname(path)), file=os.path.basename(path), layer=table,
            era=";".join(map(str, eras)), polygons=n, tiles=len(tiles), tile_km=side_km,
            canopy_km2=round(canopy_km2, 3), covered_km2=round(covered, 2),
            canopy_pct=round(100 * canopy_km2 / covered, 1) if covered else "",
            pixel_m=round(math.sqrt(amin), 3) if amin else "", min_max_conf=round(cmin, 3) if cmin is not None else "",
            bbox_km=f"E{x0/1e3:.1f}-{x1/1e3:.1f} N{y0/1e3:.1f}-{y1/1e3:.1f}" if x0 is not None else "",
            written=str(changed)[:10]))
    con.close()

if not rows:
    sys.exit(f"no GeoPackages under {root}")
out = os.path.join(root, "partner_summary.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

hdr = f"{'city':8s} {'file':36s} {'era':10s} {'tiles':>5s} {'tile':>4s} {'cov km2':>8s} {'canopy':>7s} {'pct':>5s} {'pix m':>5s} {'thr':>5s} {'bbox':27s} written"
print(hdr)
for r in rows:
    print(f"{r['city']:8s} {r['file'][:36]:36s} {r['era'][:10]:10s} {r['tiles']:5d} {r['tile_km']:4.1f} "
          f"{r['covered_km2']:8.2f} {r['canopy_km2']:7.2f} {str(r['canopy_pct']):>5s} {str(r['pixel_m']):>5s} "
          f"{str(r['min_max_conf']):>5s} {r['bbox_km']:27s} {r['written']}")
print(f"\nwrote {out}")
