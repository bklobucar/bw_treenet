#!/usr/bin/env python3
"""53_build_label_queue.py -- labelling package for the research assistant: windows to digitise, test points.

Windows (label_windows.csv: id, folder, city, era, flight, sheet, E, N, size_m; E/N = window centre).
Per row, in <out>/<folder>/<id>/:
  image.tif    production imagery of the era (data/raw/<city>/<era>_OF_gray_*), window + --margin m,
               0.5 m, EPSG:3006, Byte, 0 = nodata. The row's sheet/flight file is drawn on top; the era's
               other sheets fill the margin where the window is near a sheet edge.
  labels.gpkg  layers extent (the window square), canopy and ignore (empty polygons, to digitise)
  window.qgz   QGIS project, top to bottom: extent, ignore, canopy, image; styles from styles/
Rows with an empty E, N or era are skipped and listed. Rows with folder 00_calibration (K01-K04) are the
calibration windows; they are built like the others.

Test points (<out>/01_test_points), per city: stratified random points from NMD2023 basskikt v2.1 (--nmd),
STRATA x --per-stratum. Codes are checked against the raster attribute table (NMD2023bas_v2_1.tif.vat.dbf;
the product description's Bilaga 1 is a figure). Forest share = share of the NMD 10 m cells within 50 m
that are forest (111-117, 121-127; temporarily non-forest 118/128 is not forest); a pixel takes the first
stratum it matches, so open vegetated land with 20-60 % forest share is "mixed", not "open_shrub". The
code-to-stratum table is printed. Points sit at the cell centres of a --grid m grid that are inside the
2023 tätort, not hydro_v2 water (all_touched, as 19), have imagery in every --point-eras era, and are
  >= 500 m (HIST_BUFFER, fixed) from every historical training tile (data/processed/*/tiles_<year>*,
     year < 2000), every labelling clip (data/raw/clips_for_labelling) and every window in the CSV with
     coordinates (placeholder rows cannot be excluded, so rerun once they are filled in), and
  >= --modern-buffer m (100) from every modern training tile (data/processed/*/tiles, or tiles_<year>,
     year >= 2000); 0 = the footprint only, negative = modern tiles not excluded. --modern-buffer-city
     (e.g. gtb=-1,malmo=-1,sth=100; the job script's default) overrides it per city.
Each footprint is grown by its distance on every side, so corners are slightly over-excluded. Tiles without
georeferencing (the 9 malmo/tiles_1959) are placed by 54 (own geotransform, offsets in the name, or image
matching in the 1959 validation clip / frames); a tile that cannot be placed is reported with the reason and
left out. Per city and stratum the log gives the candidate area before the exclusion, the area each rule
removes on its own (historical tiles, clips, windows, modern footprint, modern + 100 m, modern + the chosen
buffer), and the area left after all rules; before the spacing. The settings of the run (buffers,
footprint counts, spacing, strata, eras, NMD, seed) go to <key-dir>/settings.txt. Points are drawn per
stratum (rarest first) in random order and kept only if >= --min-spacing m (100) from every test point already
kept in the city, so the sample is spatially spread rather than strictly uniform; a stratum short of
--per-stratum is reported. The same locations go into every era. Per set (city_era):
  points.gpkg  layer points: pid (1..n, shuffled per set), label, conf, issue (empty, to fill in)
  chips/       one --chip-m image chip per point (named by pid); chips.vrt mosaics them
  points.qgz   QGIS project: points (styles/points.qml, value-map form), chips
The strata never go into the package: <key-dir>/<city>_<era>.csv maps pid -> point uid, E, N, stratum,
NMD class, forest share. Same --seed, CSV and inputs -> same points and orders.

Calibration points (<out>/00_calibration/K05_points): --cal-per-stratum (20) per stratum, split over the
cities (7/7/6, rotating) and spread evenly over the --point-eras (4 per stratum and era; one era per point).
Drawn per city after the test points, from the same candidate cells, never at a test point and >= --min-spacing
m from every test point. Calibration points are also >= --chip-m / 2 apart in E or N, so no point lies on
another point's chip (chips of different eras overlap in chips.vrt). Built like a test set (each chip from
its point's era); the era and stratum are only in <key-dir>/calibration.csv.

Existing window/set folders (and their keys) are kept, so a rerun resumes; --overwrite rebuilds them,
which REPLACES any labels already digitised into them. <out>.zip is rebuilt from <out> on every run.
<out>/log.csv is the work log (one row per window, K05_points and test point set, in working order, status
todo); it is written only if it does not exist, even with --overwrite, so the assistant's entries survive a
rerun. styles/ is copied to <out>/styles if not there.

--windows-csv CSV --out-dir DIR builds only CSV's windows (same window code: image, labels.gpkg, window.qgz with
snapping) into DIR and zips it to DIR.zip: no test points, no keys, no log.csv, no styles/. DIR may not be or
overlap the package (results/anna_labelling) or --key-dir. Package windows and key points within 500 m of each
window are listed, as these windows are not in the test-point exclusion.

    python scripts/53_build_label_queue.py                 # NMD from data/ext/nmd2023/nmd2023.vrt
    python scripts/53_build_label_queue.py --windows-csv demo_windows.csv --out-dir results/anna_demo
gis env (GDAL Python bindings).
"""
import argparse, csv, datetime, glob, importlib, os, re, shutil, sys, uuid, zipfile
import xml.etree.ElementTree as ET
import numpy as np
from osgeo import gdal, ogr, osr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cf = importlib.import_module("39_common_footprint")    # src_dir, sheet_box, warp, SHEET, TAT, CITIES
georef = importlib.import_module("54_georef_1959_tiles")  # locate: footprints of ungeoreferenced tiles

gdal.UseExceptions()
ogr.UseExceptions()
RES = 0.5
HYDRO = {c: f"data/hydro_v2/{c}_water.gpkg" for c in cf.CITIES}
TAT_LAYER = "Tatorter_2023"
POINT_ERAS = ["1960", "1970", "1990r05", "2005", "2010"]
# every tile or clip the production model was trained on; modern = data/processed/*/tiles or tiles_<year>
# with year >= 2000, the rest (tiles_<year>* before 2000, labelling clips) historical
TRAIN_GLOBS = ["data/processed/*/tiles*/images/*.tif", "data/raw/clips_for_labelling/*_clip.tif"]
TILE_YEAR = re.compile(r"^tiles_(\d{4})")
HIST_BUFFER = 500.0                             # m around historical tiles, labelling clips and windows (fixed)
# per-city modern buffers of job_label_queue.sh and why; written to settings.txt when the run uses exactly these
MODERN_CITY_DEFAULT = {"gtb": -1.0, "malmo": -1.0, "sth": 100.0}
MODERN_CITY_REASON = (
    "Modern training imagery is Gothenburg 2016 + 2024 and Malmo 2024, which are not test flights, so those "
    "locations stay eligible (gtb, malmo: modern tiles not excluded). Stockholm modern training imagery includes "
    "the 2011 flight, which is the Stockholm ~2010 test era, so Stockholm modern tiles stay excluded (+100 m).")
STYLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "styles")
DEFAULT_OUT = "results/anna_labelling"
POINTS_DIR = "01_test_points"
LOG_COLS = ["seq", "item", "folder", "type", "expected_h", "date", "hours", "status", "quality", "ignore_pct", "note"]
EXPECTED_H = {"00_calibration": 1.5, "02_failure_cases": 2, "03_closed_stands_film": 2, "04_shrub_noncanopy": 1.25,
              POINTS_DIR: 2}
# test point sets in working order: the city changes every row and the era never repeats back to back
SET_ORDER = ["malmo_2010", "gtb_1970", "sth_2005", "malmo_1960", "gtb_2010", "sth_1990r05", "malmo_1970", "gtb_2005",
             "sth_1960", "malmo_1990r05", "gtb_1960", "sth_2010", "malmo_2005", "gtb_1990r05", "sth_1970"]
CAL_DIR = "00_calibration"
CAL_SET = "K05_points"
NMD = "data/ext/nmd2023/nmd2023.vrt"           # NMD2023 basskikt v2.1, mosaic of the per-city clips (UInt16)
NMD_RES = 10.0
SHARE_R = 50.0                                  # radius of the forest share, m
# NMD2023 v2.1 codes (raster attribute table); fastmark 11x with their våtmark twins 12x
CONIFER = [111, 112, 113, 114, 121, 122, 123, 124]               # pine, spruce, mixed conifer, conifer with deciduous
DECIDUOUS = [115, 117, 116, 125, 127, 126]                       # deciduous, + hardwood, hardwood
FOREST = CONIFER + DECIDUOUS
ARTIFICIAL = [51, 53, 52]                                        # buildings, roads/railways, other artificial
OPEN_VEG = [4211, 4212, 4213, 4221, 4222, 4223, 4231, 4232, 4233]   # other open land with vegetation
OPEN_WET = [200] + list(range(211, 219)) + list(range(221, 229))   # open wetland (all subclasses)
TEMP_NONFOREST = [118, 128]
# stratum -> (codes, rule on the forest share or None, rule as text); a pixel takes the first match.
# Not in any stratum: arable 3, bare 411, peat extraction 54, water 61/62, low mountain forest 23/43,
# and artificial land with a forest share of 10-20 % or above 60 %.
STRATA = {
    "closed_conifer":   (CONIFER, None, ""),
    "closed_deciduous": (DECIDUOUS, None, ""),
    "built_up":         (ARTIFICIAL, lambda s: s < 0.10, "forest share within 50 m < 10 %"),
    "mixed":            (ARTIFICIAL + OPEN_VEG, lambda s: (s >= 0.20) & (s <= 0.60), "forest share within 50 m 20-60 %"),
    "open_shrub":       (OPEN_VEG + OPEN_WET + TEMP_NONFOREST, None, ""),
}
NMD_NAMES = {
    51: "Byggnad", 52: "Anlagd mark, ej byggnad eller väg/järnväg", 53: "Väg eller järnväg",
    111: "Tallskog på fastmark", 112: "Granskog på fastmark", 113: "Barrblandskog på fastmark",
    114: "Lövblandad barrskog på fastmark", 115: "Triviallövskog på fastmark", 116: "Ädellövskog på fastmark",
    117: "Triviallövskog med ädellövinslag på fastmark", 118: "Temporärt ej skog på fastmark",
    121: "Tallskog på våtmark", 122: "Granskog på våtmark", 123: "Barrblandskog på våtmark",
    124: "Lövblandad barrskog på våtmark", 125: "Triviallövskog på våtmark", 126: "Ädellövskog på våtmark",
    127: "Triviallövskog med ädellövinslag på våtmark", 128: "Temporärt ej skog på våtmark",
    200: "Öppen våtmark (underindelning saknas)", 211: "Buskmyr", 212: "Ristuvemyr", 213: "Fastmattemyr, mager",
    214: "Fastmattemyr, frodig", 215: "Sumpkärr", 216: "Mjukmattemyr", 217: "Lösbottenmyr", 218: "Övrig öppen myr",
    221: "Våtmark med buskar", 222: "Risdominerad våtmark", 223: "Gräsdominerad våtmark, mager",
    224: "Gräsdominerad våtmark, frodvuxen", 225: "Gräsdominerad våtmark, högvuxen", 226: "Mossdominerad våtmark",
    227: "Våtmark utan växttäcke", 228: "Övrig öppen våtmark",
    4211: "Torr buskdominerad mark", 4212: "Frisk buskdominerad mark", 4213: "Frisk-fuktig buskdominerad mark",
    4221: "Torr risdominerad mark", 4222: "Frisk risdominerad mark", 4223: "Frisk-fuktig risdominerad mark",
    4231: "Torr gräsdominerad mark", 4232: "Frisk gräsdominerad mark", 4233: "Frisk-fuktig gräsdominerad mark",
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


# snapping for the window projects, as QGIS 3.40.5 saves it: enabled, all layers (mode 2), vertex and segment
# (type 3), 10 px (unit 1), no self-/intersection snapping; avoid overlap on the active layer
# (AvoidIntersectionsMode 1), no topological editing. Tracing is not a project setting (canvas toggle, T).
SNAP = dict(enabled="1", mode="2", type="3", tolerance="10", unit="1", minScale="0", maxScale="0",
            scaleDependencyMode="0", **{"self-snapping": "0", "intersection-snapping": "0"})


def write_qgz(path, title, layers, box, snapping=False):
    """layers, top to bottom: dict(name, source relative to the project, kind vector|raster, geom, style).
    snapping: SNAP settings (window projects)."""
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
    if snapping:
        indiv = ET.SubElement(ET.SubElement(q, "snapping-settings", **SNAP), "individual-layer-settings")
    for L in layers:
        lid = f"{L['name']}_{uuid.uuid4().hex}"
        prov = "ogr" if L["kind"] == "vector" else "gdal"
        ET.SubElement(tree, "layer-tree-layer", id=lid, name=L["name"], source=L["source"], providerKey=prov,
                      checked="Qt::Checked", expanded="1")
        attrs = dict(type=L["kind"])
        if L["kind"] == "vector":
            attrs.update(geometry=L["geom"], wkbType=L["geom"])
            if snapping:
                ET.SubElement(indiv, "layer-setting", id=lid, enabled="1", type=SNAP["type"], tolerance=SNAP["tolerance"],
                              units=SNAP["unit"], minScale="0", maxScale="0")
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
    if snapping:
        dig = ET.SubElement(props, "Digitizing")
        ET.SubElement(dig, "AvoidIntersectionsMode", type="int").text = "1"
        ET.SubElement(dig, "TopologicalEditing", type="int").text = "0"
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
                    style=gray_pipe(*stretch(img)))], cbox, snapping=True)
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


def is_modern(f):
    """data/processed/<city>/tiles or tiles_<year> with year >= 2000; everything else is historical."""
    d = os.path.basename(os.path.dirname(os.path.dirname(f)))
    m = TILE_YEAR.match(d)
    return "/processed/" in f.replace("\\", "/") and (d == "tiles" or (m is not None and int(m.group(1)) >= 2000))


def train_boxes():
    """-> (historical tile boxes, labelling clip boxes, modern tile boxes, notes for settings.txt)"""
    hist, clips, modern, sets, notes = [], [], [], {}, []
    put = lambda f, b: (modern if is_modern(f) else hist if "/processed/" in f.replace("\\", "/") else clips).append(b)
    for g in TRAIN_GLOBS:
        files, bad, placed = sorted(glob.glob(g)), [], 0
        for f in files:
            sets.setdefault(os.path.dirname(os.path.dirname(f)) if "processed" in f else os.path.dirname(f), []).append(f)
            try:
                d = gdal.Open(f)
            except RuntimeError:
                bad.append((f, "unreadable")); continue
            gt = d.GetGeoTransform()
            if gt == (0, 1, 0, 0, 0, 1) or epsg(d.GetSpatialRef()) != "3006":
                bad.append((f, "no EPSG:3006 georeferencing")); continue
            x = (gt[0], gt[0] + gt[1] * d.RasterXSize)
            y = (gt[3], gt[3] + gt[5] * d.RasterYSize)
            put(f, (min(x), min(y), max(x), max(y)))
        nogeo = [f for f, why in bad if why.startswith("no EPSG") and "tiles_1959" in f.replace("\\", "/")]
        if nogeo:
            print(f"  {len(nogeo)} tiles_1959 tiles without georeferencing: placing them with 54")
            try:
                rows = georef.locate(nogeo, log=print)
            except Exception as e:                      # print why and continue without them
                print(f"    could not place them ({type(e).__name__}: {e}); continuing without them")
                rows = []
            notes.append(f"tiles_1959 without georeferencing: {sum(r['ok'] for r in rows)}/{len(nogeo)} placed by 54 "
                         + "; ".join(f"{os.path.basename(r['file'])} {r.get('method', '')} NCC {r.get('ncc', '')}"
                                     for r in rows if r["ok"]))
            for r in rows:
                if r["ok"]:
                    put(r["file"], tuple(r[k] for k in ("xmin", "ymin", "xmax", "ymax")))
                    bad = [b for b in bad if b[0] != r["file"]]
                    placed += 1
                else:
                    bad = [(f, r["why"]) if f == r["file"] else (f, w) for f, w in bad]
        print(f"  training footprints {g}: {len(files)} files, {len(files) - len(bad)} used ({placed} placed by 54)")
        notes.append(f"{g}: {len(files)} files, {len(files) - len(bad)} used, {len(bad)} not used")
        for f, why in bad[:10]:
            print(f"    NOT USED ({why}): {f}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more not used")
    print("  training sets, buffer class:")
    for s, fs in sorted(sets.items()):
        print(f"    {s}: {len(fs)} files, {'modern' if is_modern(fs[0]) else 'historical'}")
        notes.append(f"  {s}: {len(fs)} files, {'modern' if is_modern(fs[0]) else 'historical'}")
    return hist, clips, modern, notes


def nmd_sheet(path, box):
    """NMD codes on the box at 10 m, and per cell the forest share of the NMD cells within SHARE_R m."""
    r = int(round(SHARE_R / NMD_RES))
    m = r * NMD_RES
    c = gdal.Warp("", path, format="MEM", outputBounds=(box[0] - m, box[1] - m, box[2] + m, box[3] + m),
                  xRes=NMD_RES, yRes=NMD_RES, resampleAlg="near", srcNodata="None",
                  outputType=gdal.GDT_UInt16).ReadAsArray()        # UInt16: codes go up to 4233
    forest, known = np.isin(c, FOREST).astype(np.float32), (c > 0).astype(np.float32)
    H, W = c.shape[0] - 2 * r, c.shape[1] - 2 * r
    fs, ks = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                fs += forest[r + dy:r + dy + H, r + dx:r + dx + W]
                ks += known[r + dy:r + dy + H, r + dx:r + dx + W]
    return c[r:r + H, r:r + W], fs / np.maximum(ks, 1)


def print_strata():
    print(f"  NMD2023 v2.1 code -> stratum (a pixel takes the first stratum it matches; forest = {FOREST})")
    for s, (codes, _, rule) in STRATA.items():
        print(f"    {s}" + (f"   [{rule}]" if rule else ""))
        for c in codes:
            print(f"      {c:5d}  {NMD_NAMES[c]}")


def cal_alloc(a):
    """(city, stratum) -> calibration points: --cal-per-stratum split over the cities, the remainder rotating."""
    nc, out = len(a.cities), {}
    for k, s in enumerate(STRATA):
        for j, city in enumerate(a.cities):
            out[city, s] = a.cal_per_stratum // nc + ((j - k) % nc < a.cal_per_stratum % nc)
    return out


def spaced(rng, total, need, xy, taken, d, peers=None, cheb=None):
    """Up to `need` of `total` candidates, visited in random order, each kept only if >= d m from `taken`
    (n x 2 array) and from those kept before it. With cheb: kept only if >= d m from `taken`, and >= cheb m in
    E or N from `peers` and those kept before it. xy(indices) -> n x 2 coordinates. -> kept indices"""
    kept, done = [], np.zeros(0, np.int64)
    taken = np.asarray(taken, float).reshape(-1, 2)
    peers = np.asarray(peers if peers is not None else [], float).reshape(-1, 2)
    for size in (min(total, max(50 * need, 2000)), total):    # a sample first; all candidates only if needed
        cand = rng.choice(total, size, replace=False) if size < total else rng.permutation(total)
        cand = cand[~np.isin(cand, done)]
        done = np.concatenate([done, cand])
        for c0 in range(0, len(cand), 5000):
            chunk = cand[c0:c0 + 5000]
            for g, p in zip(chunk, xy(chunk)):
                if len(taken) and ((taken - p) ** 2).sum(1).min() < d * d:
                    continue
                if cheb is None:
                    taken = np.vstack([taken, p])
                elif len(peers) and np.abs(peers - p).max(1).min() < cheb:
                    continue
                else:
                    peers = np.vstack([peers, p])
                kept.append(int(g))
                if len(kept) == need:
                    return kept
        if size == total:
            break
    return kept


def sample_city(city, a, rules, excl, tat_ds, rng, n_cal):
    """-> (test points, calibration points), each [dict(uid, city, E, N, stratum, nmd, share)] in uid order
    (shuffled, so uid order says nothing about strata). n_cal: stratum -> calibration points for this city.
    rules: [(label, grown boxes)] whose removed area is reported one by one; excl: all rules combined."""
    if not os.path.exists(HYDRO[city]):
        print(f"  {city}: SKIPPED, no hydro_v2 water at {HYDRO[city]}"); return [], []
    water_ds = gdal.OpenEx(HYDRO[city], gdal.OF_VECTOR)
    idx = {e: era_index(city, e) for e in a.point_eras}
    empty = [e for e in a.point_eras if not idx[e]]
    if empty:
        print(f"  {city}: SKIPPED, no imagery for era(s) {empty}"); return [], []
    sheets = sorted(set.intersection(*[set(idx[e]) for e in a.point_eras]))
    names = [tat_ds.GetLayer(i).GetName() for i in range(tat_ds.GetLayerCount())]
    tat_layers = [TAT_LAYER] if TAT_LAYER in names else None
    f = int(round(NMD_RES / a.grid))                # sampling cells per NMD cell, per axis
    cells = []                                      # per sheet: (sheet, nx, flat index, stratum, nmd, share)
    area0, area = np.zeros(len(STRATA) + 1), np.zeros(len(STRATA) + 1)    # before / after the exclusion
    removed = np.zeros((len(rules), len(STRATA) + 1))                      # by each rule on its own
    for sh in sheets:
        box = cf.sheet_box(sh)
        ok = burn(tat_ds, box, a.grid, layers=tat_layers)
        if not ok.any():
            continue
        ok &= ~burn(water_ds, box, a.grid, all_touched=True)
        for e in a.point_eras:
            ok &= cf.warp(idx[e][sh], box, a.grid, "near") > 0
        if not ok.any():
            continue
        nmd, share = (np.repeat(np.repeat(x, f, 0), f, 1) for x in nmd_sheet(a.nmd, box))
        st = np.zeros(ok.shape, np.uint8)
        for k, (codes, rule, _) in enumerate(STRATA.values(), 1):
            m = ok & (st == 0) & np.isin(nmd, codes)
            if rule is not None:
                m &= rule(share)
            st[m] = k
        area0 += np.bincount(st.ravel(), minlength=len(STRATA) + 1)
        for i, (_, boxes) in enumerate(rules):
            removed[i] += np.bincount(st[box_mask(box, a.grid, boxes)], minlength=len(STRATA) + 1)
        st[box_mask(box, a.grid, excl)] = 0
        flat = np.flatnonzero(st)
        area += np.bincount(st.ravel()[flat], minlength=len(STRATA) + 1)
        cells.append((sh, ok.shape[1], flat, st.ravel()[flat], nmd.ravel()[flat], share.ravel()[flat]))
    km2 = a.grid ** 2 / 1e6
    print(f"  {city}: {len(sheets)} sheets with imagery in every era; candidate area by stratum (km2), before the "
          f"spacing: before exclusion, removed by each rule on its own (R1-R{len(rules)}), left after all rules")
    for i, (label, _) in enumerate(rules, 1):
        print(f"    R{i} = {label}")
    print(f"    {'stratum':18s} {'before':>8s}" + "".join(f"{'R%d' % i:>8s}" for i in range(1, len(rules) + 1))
          + f"{'after':>8s}")
    for k, s in list(enumerate(STRATA, 1)) + [(slice(1, None), "total")]:
        v = lambda x: float(np.sum(x[k])) * km2
        print(f"    {s:18s} {v(area0):8.2f}" + "".join(f"{v(r):8.2f}" for r in removed) + f"{v(area):8.2f}")
    by_stratum = {}
    for k in range(1, len(STRATA) + 1):
        sel = [(sh, nx, fl[v == k], c[v == k], sv[v == k]) for sh, nx, fl, v, c, sv in cells]
        counts = np.array([len(x[2]) for x in sel], np.int64)
        by_stratum[k] = (sel, counts, np.cumsum(counts))

    def locate(k, gs):
        sel, counts, cum = by_stratum[k]
        out = []
        for g in np.atleast_1d(gs):
            i = int(np.searchsorted(cum, g, side="right"))
            sh, nx, fl, c, sv = sel[i]
            j = int(g - (cum[i] - counts[i]))
            row, col = divmod(int(fl[j]), nx)
            box = cf.sheet_box(sh)
            out.append((box[0] + (col + 0.5) * a.grid, box[3] - (row + 0.5) * a.grid, int(c[j]), float(sv[j])))
        return out

    def draw(k, need, taken, peers=None, cheb=None):
        total = int(by_stratum[k][1].sum())
        xy = lambda gs: np.array([p[:2] for p in locate(k, gs)]).reshape(-1, 2)
        return locate(k, spaced(rng, total, need, xy, taken, a.min_spacing, peers, cheb)) if total and need else []

    order = np.argsort(area[1:], kind="stable") + 1             # rarest stratum first
    test, taken = [], np.zeros((0, 2))
    for k in order:                                             # test points: >= --min-spacing m apart
        s = list(STRATA)[k - 1]
        got = draw(k, a.per_stratum, taken)
        if len(got) < a.per_stratum:
            print(f"    WARNING: {s}: only {len(got)} test points with {a.min_spacing:g} m spacing "
                  f"(candidate area {area[k] * km2:.2f} km2), fewer than {a.per_stratum}")
        taken = np.vstack([taken, np.array([p[:2] for p in got]).reshape(-1, 2)])
        test += [dict(city=city, E=e, N=n, stratum=s, nmd=c, share=round(sv, 3)) for e, n, c, sv in got]
    cal, peers = [], np.zeros((0, 2))
    for k in order:              # calibration: >= --min-spacing m from every test point, not on each other's chips
        s = list(STRATA)[k - 1]
        got = draw(k, n_cal[s], taken, peers, a.chip_m / 2)
        if len(got) < n_cal[s]:
            print(f"    WARNING: {s}: only {len(got)} calibration points, wanted {n_cal[s]}")
        peers = np.vstack([peers, np.array([p[:2] for p in got]).reshape(-1, 2)])
        cal += [dict(city=city, E=e, N=n, stratum=s, nmd=c, share=round(sv, 3)) for e, n, c, sv in got]
    for pts, prefix in ((test, f"{city}_"), (cal, f"cal_{city}_")):
        rng.shuffle(pts)
        for i, p in enumerate(pts, 1):
            p["uid"] = f"{prefix}{i:03d}"
    print(f"    {len(test)} test points: " + ", ".join(f"{s} {sum(p['stratum'] == s for p in test)}" for s in STRATA))
    print(f"    {len(cal)} calibration points: " + ", ".join(f"{s} {sum(p['stratum'] == s for p in cal)}" for s in STRATA))
    return test, cal


def build_set(name, folder, era, pts, perm, a):
    """One point set in <out>/<folder>; key <key-dir>/<name>.csv. Points may come from several cities; with
    era None each point's chip comes from its own era (p["era"])."""
    dst = os.path.join(a.out, folder)
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
    ce = lambda p: (p["city"], era or p["era"])
    vrts, h = {k: era_vrt(*k) for k in sorted({ce(p) for p in order})}, a.chip_m / 2
    vals = []
    for pid, p in enumerate(order, 1):
        c = clip(vrts[ce(p)], (p["E"] - h, p["N"] - h, p["E"] + h, p["N"] + h),
                 os.path.join(tmp, "chips", f"{pid:03d}.tif"))
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
               dict(name=f"image {era or '(era per point)'}", source="./chips.vrt", kind="raster", style=gray_pipe(*stretch(np.concatenate(vals))))],
              (min(E) - h, min(N) - h, max(E) + h, max(N) + h))
    os.makedirs(a.key_dir, exist_ok=True)
    with open(key + ".part", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "pid", "uid", "city", "era", "E", "N", "stratum", "nmd_class",
                                           "forest_share_50m"], lineterminator="\n")
        w.writeheader()
        for pid, p in enumerate(order, 1):
            w.writerow(dict(set=name, pid=pid, uid=p["uid"], city=p["city"], era=era or p["era"], E=p["E"], N=p["N"],
                            stratum=p["stratum"], nmd_class=p["nmd"], forest_share_50m=p["share"]))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    os.replace(key + ".part", key)
    return f"built, {len(order)} points"


def test_points(a, located):
    print(f"\n== {POINTS_DIR}")
    if not a.nmd or not os.path.exists(a.nmd):
        print(f"SKIPPED test points: NMD raster not found ({a.nmd or 'no --nmd given'})"); return
    if abs(NMD_RES / a.grid - round(NMD_RES / a.grid)) > 1e-9:
        print(f"SKIPPED test points: --grid {a.grid} does not divide the {NMD_RES:g} m NMD cell"); return
    print_strata()
    if not os.path.exists(cf.TAT):
        print(f"SKIPPED test points: 2023 tätort not found at {cf.TAT}"); return
    mb = {c: a.modern_buffer_city.get(c, a.modern_buffer) for c in a.cities}          # per-city modern buffer
    hist, clips, modern, notes = train_boxes()
    if not hist or (max(mb.values()) >= 0 and not modern):
        print("SKIPPED test points: no historical" + (" or no modern" if max(mb.values()) >= 0 else "") +
              " training footprints found, so the exclusion would be incomplete"); return
    wins = [r["box"] for r in located]
    print(f"  windows excluded: {len(wins)} CSV rows with coordinates")
    grow = lambda bs, d: np.array([(b[0] - d, b[1] - d, b[2] + d, b[3] + d) for b in bs]).reshape(-1, 4)
    H = f"+ {HIST_BUFFER:g} m"
    rules = [(f"historical training tiles {H} ({len(hist)})", grow(hist, HIST_BUFFER)),
             (f"clips_for_labelling {H} ({len(clips)})", grow(clips, HIST_BUFFER)),
             (f"label windows {H} ({len(wins)})", grow(wins, HIST_BUFFER)),
             (f"modern training tiles, footprint only ({len(modern)})", grow(modern, 0)),
             (f"modern training tiles + 100 m ({len(modern)})", grow(modern, 100))]
    base = grow(hist + clips + wins, HIST_BUFFER)
    chosen = {c: f"+ {d:g} m" if d > 0 else "footprint only" if d == 0 else "NOT excluded" for c, d in mb.items()}
    print(f"  exclusion applied: historical tiles, clips and windows {H} in every city; modern tiles: "
          + ", ".join(f"{c} {chosen[c]}" for c in a.cities))
    write_settings(a, notes, located, mb, chosen)
    tat_ds = gdal.OpenEx(cf.TAT, gdal.OF_VECTOR)
    alloc, cal = cal_alloc(a), []
    for ci, city in enumerate(a.cities):
        rng = np.random.default_rng([a.seed, ci])
        d = mb[city]
        crules = rules + ([(f"modern training tiles + {d:g} m, chosen ({len(modern)})", grow(modern, d))]
                          if d >= 0 and d not in (0, 100) else [])
        excl = np.vstack([base] + ([grow(modern, d)] if d >= 0 else []))
        src = "--modern-buffer-city" if city in a.modern_buffer_city else "--modern-buffer"
        print(f"  {city}: modern rule applied: modern training tiles {chosen[city]} ({src} {d:g})")
        pts, c = sample_city(city, a, crules, excl, tat_ds, rng, {s: alloc[city, s] for s in STRATA})
        cal += c
        if not pts:
            continue
        for era in a.point_eras:
            perm = rng.permutation(len(pts))            # drawn even for kept sets, so later orders do not shift
            name = f"{city}_{era}"
            print(f"    {name}: {build_set(name, os.path.join(POINTS_DIR, name), era, pts, perm, a)}", flush=True)
    calibration_points(a, cal)


def write_settings(a, notes, located, mb, chosen):
    """The test design of this run -> <key-dir>/settings.txt (outside the zip)."""
    os.makedirs(a.key_dir, exist_ok=True)
    path = os.path.join(a.key_dir, "settings.txt")
    L = [f"53_build_label_queue.py settings, {datetime.datetime.now().isoformat(timespec='seconds')}",
         f"command: {' '.join(sys.argv)}", "",
         "exclusion of test and calibration points:",
         f"  historical training tiles (tiles_<year>, year < 2000): + {HIST_BUFFER:g} m (fixed)",
         f"  clips_for_labelling: + {HIST_BUFFER:g} m (fixed)",
         f"  label windows ({a.csv}, rows with coordinates): + {HIST_BUFFER:g} m (fixed); "
         + ", ".join(f"{r['id']}" for r in located),
         f"  modern training tiles (tiles, tiles_<year> >= 2000): --modern-buffer {a.modern_buffer:g}, "
         f"--modern-buffer-city {','.join(f'{c}={d:g}' for c, d in a.modern_buffer_city.items()) or '(none)'}",
         *[f"    {c}: {chosen[c]}" for c in a.cities],
         "    reason: " + (MODERN_CITY_REASON if all(mb[c] == MODERN_CITY_DEFAULT.get(c) for c in a.cities)
                          else "per-city values differ from the documented default "
                               f"({','.join(f'{c}={d:g}' for c, d in MODERN_CITY_DEFAULT.items())}); none recorded"),
         "  footprints grown on every side (square buffers)",
         "training footprints:"] + [f"  {n}" for n in notes] + [
         "", "sampling:",
         f"  cities {' '.join(a.cities)}; eras {' '.join(a.point_eras)} (same test locations in every era)",
         f"  test points: {a.per_stratum} per stratum and city, >= {a.min_spacing:g} m apart within a city, rarest stratum first",
         f"  calibration points ({CAL_DIR}/{CAL_SET}): {a.cal_per_stratum} per stratum over all cities, one era each, "
         f">= {a.min_spacing:g} m from every test point, >= {a.chip_m / 2:g} m apart in E or N",
         f"  candidates: cell centres of a {a.grid:g} m grid in the 2023 tätort ({cf.TAT}), not hydro_v2 water, "
         f"imagery in every era",
         f"  chips {a.chip_m:g} m; seed {a.seed}; NMD {a.nmd}",
         "strata (NMD2023 v2.1, first match wins; forest share within %g m):" % SHARE_R]
    for st, (codes, _, rule) in STRATA.items():
        L.append(f"  {st}: {codes}" + (f" [{rule}]" if rule else ""))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  settings -> {path}")


def calibration_points(a, cal):
    """One era per calibration point, spread evenly over the eras within each stratum; one set, K05_points."""
    print(f"\n== {CAL_DIR}/{CAL_SET}: {len(cal)} points (wanted {a.cal_per_stratum * len(STRATA)})")
    rng = np.random.default_rng([a.seed, 1000])
    for s in STRATA:
        ps = sorted((p for p in cal if p["stratum"] == s), key=lambda p: p["uid"])
        for i, j in enumerate(rng.permutation(len(ps))):
            ps[j]["era"] = a.point_eras[i % len(a.point_eras)]
    for s in STRATA:
        print(f"    {s:18s} " + ", ".join(f"{c} {sum(p['city'] == c and p['stratum'] == s for p in cal)}"
                                          for c in a.cities) + " | " +
              ", ".join(f"{e} {sum(p['era'] == e and p['stratum'] == s for p in cal)}" for e in a.point_eras))
    if not cal:
        return
    perm = rng.permutation(len(cal))
    print(f"    {CAL_SET}: {build_set('calibration', os.path.join(CAL_DIR, CAL_SET), None, cal, perm, a)}", flush=True)


def work_log(a, windows):
    """<out>/log.csv, pre-filled, only if it does not exist yet (never overwritten); styles/ into <out>."""
    os.makedirs(a.out, exist_ok=True)
    st = os.path.join(a.out, "styles")
    if os.path.isdir(st):
        print(f"\n{st}/ exists, kept")
    else:
        shutil.copytree(STYLES, st)
        print(f"\n{st}/ copied from {STYLES}")
    path = os.path.join(a.out, "log.csv")
    if os.path.exists(path):
        print(f"{path} exists, KEPT (never overwritten, also not with --overwrite)"); return
    by_folder = lambda f: [(r["id"], f, "window") for r in sorted(windows, key=lambda r: r["id"]) if r["folder"] == f]
    sets = [f"{c}_{e}" for c in a.cities for e in a.point_eras]
    sets = [x for x in SET_ORDER if x in sets] + [x for x in sets if x not in SET_ORDER]
    items = (by_folder(CAL_DIR) + [(CAL_SET, CAL_DIR, "points")] + [(x, POINTS_DIR, "points") for x in sets]
             + by_folder("02_failure_cases") + by_folder("03_closed_stands_film") + by_folder("04_shrub_noncanopy"))
    items += [(r["id"], r["folder"], "window") for r in sorted(windows, key=lambda r: r["id"])
              if (r["id"], r["folder"], "window") not in items]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(LOG_COLS)
        for i, (item, folder, kind) in enumerate(items, 1):
            h = 1.5 if item == CAL_SET else EXPECTED_H.get(folder, "")
            w.writerow([i, item, folder, kind, h, "", "", "todo", "", "", ""])
    print(f"{path} created ({len(items)} rows, status todo)")


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


def windows_only(a):
    """--windows-csv: build only that CSV's windows into --out-dir and zip it; no test points, keys, log or styles.
    Refuses the package and key folders. Package windows (--csv) and key points within HIST_BUFFER m of a window
    are listed (read only), since those windows are not in the test-point exclusion."""
    out = os.path.abspath(a.out)
    for d in (DEFAULT_OUT, a.key_dir):
        d = os.path.abspath(d)
        if out == d or out.startswith(d + os.sep) or d.startswith(out + os.sep):
            sys.exit(f"[STOP] --windows-csv writes outside the package: --out-dir {a.out} overlaps {d}")
    located, todo, skipped = read_windows(a.windows_csv)
    print(f"== windows only, from {a.windows_csv} -> {a.out}: {len(todo)} to build, {len(skipped)} skipped")
    for r, why in skipped:
        print(f"  skipped {r['id']} ({r['folder']}): {why}")
    for r in todo:
        print(f"  {r['folder']}/{r['id']} {r['city']} {r['era']}: {build_window(r, a)}", flush=True)
    pkg = read_windows(a.csv)[0] if os.path.exists(a.csv) else []
    keys = []
    for k in sorted(glob.glob(os.path.join(a.key_dir, "*.csv"))):
        with open(k, newline="") as fh:
            keys += [(os.path.basename(k), p["pid"], float(p["E"]), float(p["N"])) for p in csv.DictReader(fh)]
    near = lambda b, x0, y0, x1, y1: x0 < b[2] + HIST_BUFFER and x1 > b[0] - HIST_BUFFER and \
        y0 < b[3] + HIST_BUFFER and y1 > b[1] - HIST_BUFFER
    print(f"\n  within {HIST_BUFFER:g} m of each window ({a.csv} windows; points in {len(keys)} key rows of {a.key_dir}):")
    for r in todo:
        w = [p["id"] for p in pkg if near(r["box"], *p["box"])]
        k = [f"{s}:{pid}" for s, pid, e, n in keys if near(r["box"], e, n, e, n)]
        print(f"    {r['id']}: windows {', '.join(w) or 'none'}; points {', '.join(k) or 'none'}")
    if os.path.isdir(a.out):
        zip_out(a.out)


def main(a):
    if a.windows_csv:
        return windows_only(a)
    located, todo, skipped = read_windows(a.csv)
    print(f"== windows from {a.csv}: {len(todo)} to build, {len(skipped)} skipped")
    for r, why in skipped:
        print(f"  skipped {r['id']} ({r['folder']}): {why}")
    for r in todo:
        print(f"  {r['folder']}/{r['id']} {r['city']} {r['era']}: {build_window(r, a)}", flush=True)
    if not a.no_points:
        test_points(a, located)
    work_log(a, todo + [r for r, _ in skipped])
    if os.path.isdir(a.out):
        zip_out(a.out)


def city_buffers(text):
    """'gtb=-1,malmo=-1,sth=100' -> {city: metres}"""
    out = {}
    for item in filter(None, (x.strip() for x in text.split(","))):
        c, _, d = item.partition("=")
        if c not in cf.CITIES or not d:
            raise argparse.ArgumentTypeError(f"bad item {item!r}: use CITY=METRES with CITY in {cf.CITIES}")
        out[c] = float(d)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="label_windows.csv")
    ap.add_argument("--nmd", default=NMD, help="NMD2023 basskikt v2.1 raster or VRT (EPSG:3006, 10 m, UInt16)")
    ap.add_argument("--out", "--out-dir", dest="out", default=DEFAULT_OUT)
    ap.add_argument("--windows-csv", help="build only these windows into --out-dir (not the package) and zip it; "
                                          "no test points, keys, log.csv or styles (e.g. demo_windows.csv)")
    ap.add_argument("--key-dir", default="results/anna_labelling_key", help="strata keys; kept out of the zip")
    ap.add_argument("--margin", type=float, default=25.0, help="context around each window, m")
    ap.add_argument("--cities", nargs="+", default=cf.CITIES)
    ap.add_argument("--point-eras", nargs="+", default=POINT_ERAS)
    ap.add_argument("--per-stratum", type=int, default=30)
    ap.add_argument("--modern-buffer-city", type=city_buffers, default={}, metavar="CITY=M,...",
                    help="per-city --modern-buffer, e.g. gtb=-1,malmo=-1,sth=100 (-1 not excluded, 0 footprint only)")
    ap.add_argument("--cal-per-stratum", type=int, default=20, help="calibration points per stratum, all cities")
    ap.add_argument("--modern-buffer", type=float, default=100.0, metavar="METRES",
                    help="around modern training tiles; 0 = footprint only, negative = not excluded "
                         "(historical tiles, clips and windows: always %g m)" % HIST_BUFFER)
    ap.add_argument("--min-spacing", type=float, default=100.0,
                    help="between test points of a city, and from calibration to test points, m")
    ap.add_argument("--grid", type=float, default=2.0, help="sampling grid, m")
    ap.add_argument("--chip-m", type=float, default=300.0, help="image chip side per test point, m")
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--no-points", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing folders (replaces labels in them)")
    main(ap.parse_args())
