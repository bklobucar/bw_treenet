#!/usr/bin/env python3
"""53_build_label_queue.py -- labelling package for the research assistant: windows to digitise, test points.

Windows (label_windows.csv: id, folder, city, era, flight, sheet, E, N, size_m; E/N = window centre).
Per row, in <out>/<folder>/<id>/:
  image.tif    production imagery of the era (data/raw/<city>/<era>_OF_gray_*), window + --margin m,
               0.5 m, EPSG:3006, Byte, 0 = nodata. The row's sheet/flight file is drawn on top; the era's
               other sheets fill the margin where the window is near a sheet edge.
  labels.gpkg  layers extent (the window square), canopy and ignore (empty polygons, to digitise)
  window.qgz   QGIS project, top to bottom: extent, ignore, canopy, image; styles from styles/
Rows with an empty E, N or era are skipped and listed.

Test points (<out>/01_test_points), per city: stratified random points from NMD 2018 (--nmd), STRATA x
--per-stratum, at the cell centres of a --grid m grid that are inside the 2023 tätort, not hydro_v2
water (all_touched, as 19), have imagery in every --point-eras era, and are >= --min-dist m from every
training tile / clip (TRAIN_GLOBS) and every window in the CSV with coordinates (placeholder rows
cannot be excluded, so rerun once they are filled in). Each footprint is grown by --min-dist on every
side, so corners are slightly over-excluded. The same locations go into every era. Per set (city_era):
  points.gpkg  layer points: pid (1..n, shuffled per set), label, conf, issue (empty, to fill in)
  chips/       one --chip-m image chip per point (named by pid); chips.vrt mosaics them
  points.qgz   QGIS project: points (styles/points.qml, value-map form), chips
The strata never go into the package: <key-dir>/<city>_<era>.csv maps pid -> point uid, E, N, stratum,
NMD class. Same --seed, CSV and inputs -> same points and orders.

Existing window/set folders (and their keys) are kept, so a rerun resumes; --overwrite rebuilds them,
which REPLACES any labels already digitised into them. <out>.zip is rebuilt from <out> on every run.

    python scripts/53_build_label_queue.py --nmd data/ext/NMD2018bas_ogeneraliserad_v1_1.tif
gis env (GDAL Python bindings).
"""
import argparse, csv, glob, importlib, os, shutil, sys, uuid, zipfile
import xml.etree.ElementTree as ET
import numpy as np
from osgeo import gdal, ogr, osr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cf = importlib.import_module("39_common_footprint")    # src_dir, sheet_box, warp, SHEET, TAT, CITIES

gdal.UseExceptions()
ogr.UseExceptions()
RES = 0.5
HYDRO = {c: f"data/hydro_v2/{c}_water.gpkg" for c in cf.CITIES}
TAT_LAYER = "Tatorter_2023"
POINT_ERAS = ["1960", "1970", "1990r05", "2005", "2010"]
# every tile or clip the production model was trained on (modern and historical)
TRAIN_GLOBS = ["data/processed/*/tiles*/images/*.tif", "data/raw/clips_for_labelling/*_clip.tif"]
STYLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "styles")
POINTS_DIR = "01_test_points"
# NMD 2018 basskikt class -> stratum (water, 61/62, is in no stratum)
STRATA = {
    "forest":           list(range(111, 119)) + list(range(121, 129)),
    "open_vegetated":   [2, 42],                 # open wetland, other open land with vegetation
    "arable_bare":      [3, 41],                 # arable, other open land without vegetation
    "buildings":        [51],
    "other_artificial": [52, 53],                # other artificial surfaces, roads and railways
}
SRS = osr.SpatialReference()
SRS.ImportFromEPSG(3006)
DOCTYPE = "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"


# ── imagery ──────────────────────────────────────────────────────────────────
def era_index(city, era):
    """sheet -> [image files] for one city-era."""
    out = {}
    for f in sorted(glob.glob(os.path.join(cf.src_dir(city, era), "*.tif"))):
        m = cf.SHEET.search(os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def era_vrt(city, era, top=None):
    """In-memory mosaic of an era's sheets; `top` (one of its files) is drawn last, i.e. on top."""
    files = [f for f in sorted(glob.glob(os.path.join(cf.src_dir(city, era), "*.tif"))) if f != top]
    files += [top] if top else []
    return gdal.BuildVRT("", files, srcNodata=0, VRTNodata=0, resolution="highest") if files else None


def clip(src, box, path):
    """src clipped to box at 0.5 m -> single-band Byte GeoTIFF (plain band mean for RGB, as 19). -> array"""
    m = gdal.Warp("", src, format="MEM", outputBounds=box, xRes=RES, yRes=RES, dstSRS="EPSG:3006",
                  resampleAlg="near", srcNodata=0, dstNodata=0, outputType=gdal.GDT_Byte)
    a = m.ReadAsArray()
    if a.ndim == 3:
        a = np.floor(a[:3].astype(np.float32).mean(0)).astype(np.uint8)
    d = gdal.GetDriverByName("GTiff").Create(path, a.shape[1], a.shape[0], 1, gdal.GDT_Byte,
                                             options=["COMPRESS=DEFLATE", "TILED=YES"])
    d.SetGeoTransform(m.GetGeoTransform())
    d.SetProjection(m.GetProjection())
    d.GetRasterBand(1).WriteArray(a)
    d.GetRasterBand(1).SetNoDataValue(0)
    d = None
    return a


def stretch(values):
    v = values[values > 0]
    return tuple(float(x) for x in np.percentile(v, (2, 98))) if v.size else (0.0, 255.0)


# ── vectors ──────────────────────────────────────────────────────────────────
def box_poly(b):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]), (b[0], b[1])):
        ring.AddPoint_2D(x, y)
    p = ogr.Geometry(ogr.wkbPolygon)
    p.AddGeometry(ring)
    return p


def labels_gpkg(path, wid, box):
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    lyr = ds.CreateLayer("extent", SRS, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("id", ogr.OFTString))
    f = ogr.Feature(lyr.GetLayerDefn())
    f.SetField("id", wid)
    f.SetGeometry(box_poly(box))
    lyr.CreateFeature(f)
    for name in ("canopy", "ignore"):
        ds.CreateLayer(name, SRS, ogr.wkbPolygon)
    ds = None


def points_gpkg(path, pts):
    """pts: [(pid, E, N)] in pid order."""
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    lyr = ds.CreateLayer("points", SRS, ogr.wkbPoint)
    for name, t in (("pid", ogr.OFTInteger), ("label", ogr.OFTInteger), ("conf", ogr.OFTInteger),
                    ("issue", ogr.OFTString)):
        lyr.CreateField(ogr.FieldDefn(name, t))
    for pid, e, n in pts:
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetField("pid", pid)
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(e, n)
        f.SetGeometry(g)
        lyr.CreateFeature(f)
    ds = None


def burn(ds, box, res, all_touched=False, layers=None):
    """Rasterise the vector dataset ds (reprojected on the fly if needed) on the box grid -> bool."""
    nx, ny = int(round((box[2] - box[0]) / res)), int(round((box[3] - box[1]) / res))
    m = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    m.SetGeoTransform((box[0], res, 0, box[3], 0, -res))
    m.SetProjection(SRS.ExportToWkt())
    layers = layers or [ds.GetLayer(i).GetName() for i in range(ds.GetLayerCount())]
    gdal.Rasterize(m, ds, burnValues=[1], allTouched=all_touched, layers=layers)
    return m.ReadAsArray().astype(bool)


def box_mask(box, res, boxes):
    """True where any of boxes (N x 4 array: xmin ymin xmax ymax) covers the box grid."""
    nx, ny = int(round((box[2] - box[0]) / res)), int(round((box[3] - box[1]) / res))
    m = np.zeros((ny, nx), bool)
    if not len(boxes):
        return m
    hit = boxes[(boxes[:, 2] > box[0]) & (boxes[:, 0] < box[2]) & (boxes[:, 3] > box[1]) & (boxes[:, 1] < box[3])]
    for x0, y0, x1, y1 in hit:
        c0, c1 = max(0, int(np.floor((x0 - box[0]) / res))), min(nx, int(np.ceil((x1 - box[0]) / res)))
        r0, r1 = max(0, int(np.floor((box[3] - y1) / res))), min(ny, int(np.ceil((box[3] - y0) / res)))
        m[r0:r1, c0:c1] = True
    return m


# ── QGIS project (.qgz written directly; no PyQGIS on the cluster) ───────────
def srs_el(parent, tag):
    e = ET.SubElement(parent, tag)
    s = ET.SubElement(e, "spatialrefsys", nativeFormat="Wkt")
    for k, v in (("wkt", SRS.ExportToWkt(["FORMAT=WKT2_2019"])), ("proj4", SRS.ExportToProj4().strip()),
                 ("srid", "3006"), ("authid", "EPSG:3006"), ("description", "SWEREF99 TM"),
                 ("projectionacronym", "utm"), ("ellipsoidacronym", "EPSG:7019"), ("geographicflag", "false")):
        ET.SubElement(s, k).text = v
    return e


def extent_el(parent, box, tag="extent"):
    e = ET.SubElement(parent, tag)
    for k, v in zip(("xmin", "ymin", "xmax", "ymax"), box):
        ET.SubElement(e, k).text = repr(float(v))


def qml(name):
    return list(ET.parse(os.path.join(STYLES, name)).getroot())


def gray_pipe(lo, hi):
    pipe = ET.Element("pipe")
    r = ET.SubElement(pipe, "rasterrenderer", type="singlebandgray", grayBand="1", gradient="BlackToWhite",
                      opacity="1", alphaBand="-1", nodataColor="")
    ET.SubElement(r, "rasterTransparency")
    mmo = ET.SubElement(r, "minMaxOrigin")
    for k, v in (("limits", "None"), ("extent", "WholeRaster"), ("statAccuracy", "Estimated"),
                 ("cumulativeCutLower", "0.02"), ("cumulativeCutUpper", "0.98"), ("stdDevFactor", "2")):
        ET.SubElement(mmo, k).text = v
    ce = ET.SubElement(r, "contrastEnhancement")
    ET.SubElement(ce, "minValue").text = f"{lo:g}"
    ET.SubElement(ce, "maxValue").text = f"{hi:g}"
    ET.SubElement(ce, "algorithm").text = "StretchToMinimumMaximum"
    ET.SubElement(pipe, "brightnesscontrast", brightness="0", contrast="0", gamma="1")
    ET.SubElement(pipe, "huesaturation", saturation="0", grayscaleMode="0", colorizeOn="0", colorizeRed="255",
                  colorizeGreen="128", colorizeBlue="128", colorizeStrength="100", invertColors="0")
    ET.SubElement(pipe, "rasterresampler", maxOversample="2")
    return [pipe]


def write_qgz(path, title, layers, box):
    """layers, top to bottom: dict(name, source relative to the project, kind vector|raster, geom, style)."""
    q = ET.Element("qgis", projectname=title, version="3.34.0-Prizren")
    ET.SubElement(q, "homePath", path="")
    ET.SubElement(q, "title").text = title
    srs_el(q, "projectCrs")
    tree = ET.SubElement(q, "layer-tree-group")
    ET.SubElement(tree, "customproperties")
    mc = ET.SubElement(q, "mapcanvas", name="theMapCanvas", annotationsVisible="1")
    ET.SubElement(mc, "units").text = "meters"
    extent_el(mc, box)
    ET.SubElement(mc, "rotation").text = "0"
    srs_el(mc, "destinationsrs")
    pl = ET.SubElement(q, "projectlayers")
    for L in layers:
        lid = f"{L['name']}_{uuid.uuid4().hex}"
        prov = "ogr" if L["kind"] == "vector" else "gdal"
        ET.SubElement(tree, "layer-tree-layer", id=lid, name=L["name"], source=L["source"], providerKey=prov,
                      checked="Qt::Checked", expanded="1")
        attrs = dict(type=L["kind"])
        if L["kind"] == "vector":
            attrs.update(geometry=L["geom"], wkbType=L["geom"])
        m = ET.SubElement(pl, "maplayer", **attrs)
        extent_el(m, box)
        ET.SubElement(m, "id").text = lid
        ET.SubElement(m, "datasource").text = L["source"]
        ET.SubElement(m, "layername").text = L["name"]
        srs_el(m, "srs")
        if L["kind"] == "vector":
            ET.SubElement(m, "provider", encoding="UTF-8").text = prov
        else:
            ET.SubElement(m, "provider").text = prov
        m.extend(L["style"])
    props = ET.SubElement(q, "properties")
    paths = ET.SubElement(props, "Paths")
    ET.SubElement(paths, "Absolute", type="bool").text = "false"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(os.path.splitext(os.path.basename(path))[0] + ".qgs",
                   DOCTYPE + ET.tostring(q, encoding="unicode"))


# ── windows ──────────────────────────────────────────────────────────────────
def read_windows(path):
    """-> (rows with coordinates, rows to build, skipped rows with the reason)"""
    located, todo, skipped = [], [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r = {k: (v or "").strip() for k, v in r.items()}
            if r["E"] and r["N"]:
                h = float(r["size_m"] or 250) / 2
                e, n = float(r["E"]), float(r["N"])
                r["box"] = (e - h, n - h, e + h, n + h)
                located.append(r)
            missing = [k for k in ("E", "N", "era", "city") if not r[k]]
            if missing:
                skipped.append((r, "empty " + "/".join(missing)))
            else:
                todo.append(r)
    return located, todo, skipped


def build_window(r, a):
    city, era = r["city"], r["era"]
    pat = f"{r['sheet'] or '*'}_{r['flight'] or '*'}.tif"
    own = sorted(glob.glob(os.path.join(cf.src_dir(city, era), pat)))
    if not own:
        return f"SKIPPED: no imagery {cf.src_dir(city, era)}/{pat}"
    note = ""
    if r["sheet"]:
        sb = cf.sheet_box(r["sheet"])
        e, n = float(r["E"]), float(r["N"])
        if not (sb[0] <= e < sb[2] and sb[1] <= n < sb[3]):
            note = f"; WARNING: centre is outside sheet {r['sheet']} (E{sb[0]}-{sb[2]} N{sb[1]}-{sb[3]})"
    dst = os.path.join(a.out, r["folder"], r["id"])
    if os.path.exists(dst) and not a.overwrite:
        return "exists, kept" + note
    tmp = dst + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    b, mg = r["box"], a.margin
    cbox = (b[0] - mg, b[1] - mg, b[2] + mg, b[3] + mg)
    img = clip(era_vrt(city, era, own[-1]), cbox, os.path.join(tmp, "image.tif"))
    labels_gpkg(os.path.join(tmp, "labels.gpkg"), r["id"], b)
    vec = lambda name: dict(name=name, source=f"./labels.gpkg|layername={name}", kind="vector", geom="Polygon",
                            style=qml(f"{name}.qml"))
    write_qgz(os.path.join(tmp, "window.qgz"), r["id"],
              [vec("extent"), vec("ignore"), vec("canopy"),
               dict(name=f"image {era} ({os.path.basename(own[-1])})", source="./image.tif", kind="raster",
                    style=gray_pipe(*stretch(img)))], cbox)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    return f"built from {os.path.basename(own[-1])}, imagery on {100 * (img > 0).mean():.0f}% of the clip" + note


# ── test points ──────────────────────────────────────────────────────────────
def epsg(s):
    if s is None:
        return None
    if s.GetAuthorityCode(None) is None:
        try:
            s.AutoIdentifyEPSG()
        except RuntimeError:
            return None
    return s.GetAuthorityCode(None)


def train_boxes():
    boxes = []
    for g in TRAIN_GLOBS:
        files, bad = sorted(glob.glob(g)), []
        for f in files:
            try:
                d = gdal.Open(f)
            except RuntimeError:
                bad.append((f, "unreadable")); continue
            gt = d.GetGeoTransform()
            if gt == (0, 1, 0, 0, 0, 1) or epsg(d.GetSpatialRef()) != "3006":
                bad.append((f, "no EPSG:3006 georeferencing")); continue
            x = (gt[0], gt[0] + gt[1] * d.RasterXSize)
            y = (gt[3], gt[3] + gt[5] * d.RasterYSize)
            boxes.append((min(x), min(y), max(x), max(y)))
        print(f"  training footprints {g}: {len(files)} files, {len(files) - len(bad)} used")
        for f, why in bad[:5]:
            print(f"    NOT USED ({why}): {f}")
        if len(bad) > 5:
            print(f"    ... and {len(bad) - 5} more not used")
    return boxes


def sample_city(city, a, excl, tat_ds, rng):
    """-> [dict(uid, E, N, stratum, nmd)] in uid order (shuffled, so uid order says nothing about strata)."""
    if not os.path.exists(HYDRO[city]):
        print(f"  {city}: SKIPPED, no hydro_v2 water at {HYDRO[city]}"); return []
    water_ds = gdal.OpenEx(HYDRO[city], gdal.OF_VECTOR)
    idx = {e: era_index(city, e) for e in a.point_eras}
    empty = [e for e in a.point_eras if not idx[e]]
    if empty:
        print(f"  {city}: SKIPPED, no imagery for era(s) {empty}"); return []
    sheets = sorted(set.intersection(*[set(idx[e]) for e in a.point_eras]))
    names = [tat_ds.GetLayer(i).GetName() for i in range(tat_ds.GetLayerCount())]
    tat_layers = [TAT_LAYER] if TAT_LAYER in names else None
    cells, area = [], np.zeros(len(STRATA) + 1)    # cells: (sheet, flat index, stratum, nmd) per candidate
    for sh in sheets:
        box = cf.sheet_box(sh)
        ok = burn(tat_ds, box, a.grid, layers=tat_layers)
        if not ok.any():
            continue
        ok &= ~burn(water_ds, box, a.grid, all_touched=True)
        for e in a.point_eras:
            ok &= cf.warp(idx[e][sh], box, a.grid, "near") > 0
        ok &= ~box_mask(box, a.grid, excl)
        if not ok.any():
            continue
        nmd = cf.warp(a.nmd, box, a.grid, "near")
        st = np.zeros(ok.shape, np.uint8)
        for k, codes in enumerate(STRATA.values(), 1):
            st[np.isin(nmd, codes) & ok] = k
        flat = np.flatnonzero(st)
        area += np.bincount(st.ravel()[flat], minlength=len(STRATA) + 1)
        cells.append((sh, ok.shape[1], flat, st.ravel()[flat], nmd.ravel()[flat]))
    print(f"  {city}: {len(sheets)} sheets with imagery in every era; candidate area by stratum (km2): " +
          ", ".join(f"{s} {area[k] * a.grid ** 2 / 1e6:.2f}" for k, s in enumerate(STRATA, 1)))
    pts = []
    for k, s in enumerate(STRATA, 1):
        sel = [(sh, nx, f[v == k], c[v == k]) for sh, nx, f, v, c in cells]
        counts = np.array([len(x[2]) for x in sel])
        n = min(a.per_stratum, int(counts.sum()))
        if n < a.per_stratum:
            print(f"    WARNING: {s} has only {n} candidate cells, fewer than {a.per_stratum}")
        if not n:
            continue
        cum = np.cumsum(counts)
        for g in np.sort(rng.choice(int(counts.sum()), n, replace=False)):
            i = int(np.searchsorted(cum, g, side="right"))
            sh, nx, f, c = sel[i]
            j = g - (cum[i] - counts[i])
            row, col = divmod(int(f[j]), nx)
            box = cf.sheet_box(sh)
            pts.append(dict(E=box[0] + (col + 0.5) * a.grid, N=box[3] - (row + 0.5) * a.grid,
                            stratum=s, nmd=int(c[j])))
    rng.shuffle(pts)
    for i, p in enumerate(pts, 1):
        p["uid"] = f"{city}_{i:03d}"
    print(f"    {len(pts)} points: " + ", ".join(f"{s} {sum(p['stratum'] == s for p in pts)}" for s in STRATA))
    return pts


def build_set(city, era, pts, perm, a):
    name = f"{city}_{era}"
    dst = os.path.join(a.out, POINTS_DIR, name)
    key = os.path.join(a.key_dir, f"{name}.csv")
    if os.path.exists(dst) and not a.overwrite:
        if os.path.exists(key):
            with open(key, newline="") as fh:
                old = sorted((round(float(r["E"]), 2), round(float(r["N"]), 2)) for r in csv.DictReader(fh))
            if old != sorted((round(p["E"], 2), round(p["N"], 2)) for p in pts):
                return "exists, kept, BUT its points differ from this run's (inputs changed); rerun with --overwrite"
        return "exists, kept (key untouched)"
    order = [pts[i] for i in perm]                      # pid = position in this set's shuffled order
    tmp = dst + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "chips"))
    vrt, h = era_vrt(city, era), a.chip_m / 2
    vals = []
    for pid, p in enumerate(order, 1):
        c = clip(vrt, (p["E"] - h, p["N"] - h, p["E"] + h, p["N"] + h), os.path.join(tmp, "chips", f"{pid:03d}.tif"))
        vals.append(c.ravel())
    cwd = os.getcwd()
    try:                                                # relative sources, so the folder can be moved
        os.chdir(tmp)
        gdal.BuildVRT("chips.vrt", [f"chips/{pid:03d}.tif" for pid in range(1, len(order) + 1)], srcNodata=0, VRTNodata=0)
    finally:
        os.chdir(cwd)
    points_gpkg(os.path.join(tmp, "points.gpkg"), [(pid, p["E"], p["N"]) for pid, p in enumerate(order, 1)])
    E, N = [p["E"] for p in order], [p["N"] for p in order]
    write_qgz(os.path.join(tmp, "points.qgz"), name,
              [dict(name="points", source="./points.gpkg|layername=points", kind="vector", geom="Point",
                    style=qml("points.qml")),
               dict(name=f"image {era}", source="./chips.vrt", kind="raster", style=gray_pipe(*stretch(np.concatenate(vals))))],
              (min(E) - h, min(N) - h, max(E) + h, max(N) + h))
    os.makedirs(a.key_dir, exist_ok=True)
    with open(key + ".part", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "pid", "uid", "city", "era", "E", "N", "stratum", "nmd_class"],
                           lineterminator="\n")
        w.writeheader()
        for pid, p in enumerate(order, 1):
            w.writerow(dict(set=name, pid=pid, uid=p["uid"], city=city, era=era, E=p["E"], N=p["N"],
                            stratum=p["stratum"], nmd_class=p["nmd"]))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    os.replace(key + ".part", key)
    return f"built, {len(order)} points"


def test_points(a, located):
    print(f"\n== {POINTS_DIR}")
    if not a.nmd or not os.path.exists(a.nmd):
        print(f"SKIPPED test points: NMD 2018 raster not found ({a.nmd or 'no --nmd given'})"); return
    if not os.path.exists(cf.TAT):
        print(f"SKIPPED test points: 2023 tätort not found at {cf.TAT}"); return
    boxes = train_boxes()
    if not boxes:
        print("SKIPPED test points: no training footprints found, so the 500 m exclusion would be empty"); return
    print(f"  windows excluded: {len(located)} CSV rows with coordinates")
    boxes += [r["box"] for r in located]
    d = a.min_dist
    excl = np.array([(b[0] - d, b[1] - d, b[2] + d, b[3] + d) for b in boxes])
    tat_ds = gdal.OpenEx(cf.TAT, gdal.OF_VECTOR)
    for ci, city in enumerate(a.cities):
        rng = np.random.default_rng([a.seed, ci])
        pts = sample_city(city, a, excl, tat_ds, rng)
        if not pts:
            continue
        for era in a.point_eras:
            perm = rng.permutation(len(pts))            # drawn even for kept sets, so later orders do not shift
            print(f"    {city}_{era}: {build_set(city, era, pts, perm, a)}", flush=True)


def zip_out(out):
    part = out + ".zip.part"
    root = os.path.dirname(os.path.abspath(out))
    n = 0
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, dns, fns in os.walk(out):
            dns[:] = sorted(x for x in dns if not x.endswith(".part"))
            for f in sorted(fns):
                p = os.path.join(dp, f)
                z.write(p, os.path.relpath(os.path.abspath(p), root))
                n += 1
    os.replace(part, out + ".zip")
    print(f"\nwrote {out}.zip ({n} files, {os.path.getsize(out + '.zip') / 1e6:.1f} MB)")


def main(a):
    located, todo, skipped = read_windows(a.csv)
    print(f"== windows from {a.csv}: {len(todo)} to build, {len(skipped)} skipped")
    for r, why in skipped:
        print(f"  skipped {r['id']} ({r['folder']}): {why}")
    for r in todo:
        print(f"  {r['folder']}/{r['id']} {r['city']} {r['era']}: {build_window(r, a)}", flush=True)
    if not a.no_points:
        test_points(a, located)
    if os.path.isdir(a.out):
        zip_out(a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="label_windows.csv")
    ap.add_argument("--nmd", default=None, help="NMD 2018 basskikt raster (EPSG:3006, 10 m)")
    ap.add_argument("--out", default="results/anna_labelling")
    ap.add_argument("--key-dir", default="results/anna_labelling_key", help="strata keys; kept out of the zip")
    ap.add_argument("--margin", type=float, default=25.0, help="context around each window, m")
    ap.add_argument("--cities", nargs="+", default=cf.CITIES)
    ap.add_argument("--point-eras", nargs="+", default=POINT_ERAS)
    ap.add_argument("--per-stratum", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=500.0, help="from training footprints and windows, m")
    ap.add_argument("--grid", type=float, default=2.0, help="sampling grid, m")
    ap.add_argument("--chip-m", type=float, default=100.0, help="image chip side per test point, m")
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--no-points", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing folders (replaces labels in them)")
    main(ap.parse_args())
