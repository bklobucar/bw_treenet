"""
11_score_sth.py -- validate v12 vs v18d (and the rest) on the STH 1970 clip.

Second-city confirmation of the Malmo 1970 result. Rasterizes labels_sth_1970.shp
onto sth_1970_clip.tif's grid (same approach as 09_accuracy_assessment.py), builds
a tatort mask on that same grid from the national geopackage, runs taper-blended
inference, and scores with the SAME compute_metrics used everywhere else.

The point is NOT to reproduce a remembered number -- it is to compare v12 and v18d
under identical conditions on a different city. If v18d wins here too, the ranking
generalises.

Run on a GPU node with the env active:
    python scripts/11_score_sth.py
"""
import os, sys
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "BWTreeNet", "GuiTest"))
from model.BWTreeNet import BWTreeNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATCH, OVERLAP, N_CLASSES = 1000, 200, 2
THRESHOLDS = [0.4, 0.5, 0.6]

VAL_DIR   = os.path.join(HERE, "data/processed/sth/validation")
VAL_CLIP  = os.path.join(VAL_DIR, "sth_1970_clip.tif")
VAL_SHP   = os.path.join(VAL_DIR, "labels_sth_1970.shp")
TATORT    = os.path.join(HERE, "data/Tatorter_2023.gpkg")
MODELS = {
    "v12":  os.path.join(HERE, "models/bwtreenet_v12_best.pt"),
    "v18a": os.path.join(HERE, "models/bwtreenet_v18a_best.pt"),
    "v18c": os.path.join(HERE, "models/bwtreenet_v18c_best.pt"),
    "v18d": os.path.join(HERE, "models/bwtreenet_v18d_best.pt"),
    "v19": os.path.join(HERE, "models/bwtreenet_v19_best.pt"),
    "v20": os.path.join(HERE, "models/bwtreenet_v20_best.pt"),
}
DROP_LE = {"v18b", "v18c"}


def compute_metrics(pred_binary, gt_binary):
    pred = pred_binary.astype(bool); gt = gt_binary.astype(bool)
    TP=(pred&gt).sum(); FP=(pred&~gt).sum(); FN=(~pred&gt).sum(); TN=(~pred&~gt).sum()
    total = TP+FP+FN+TN
    iou=TP/(TP+FP+FN+1e-10); p=TP/(TP+FP+1e-10); r=TP/(TP+FN+1e-10)
    return dict(TP=int(TP),FP=int(FP),FN=int(FN),TN=int(TN),
                IoU=float(iou), F1=float(2*p*r/(p+r+1e-10)),
                Precision=float(p), Recall=float(r),
                OA=float((TP+TN)/(total+1e-10)))


def rasterize_labels(shp_path, ref_raster):
    """Same approach as 09_accuracy_assessment.rasterize_labels."""
    gdf = gpd.read_file(shp_path)
    with rasterio.open(ref_raster) as src:
        transform = src.transform; shape = (src.height, src.width); crs = src.crs
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, np.uint8)
    return rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8)


def tatort_mask(ref_raster, gpkg):
    if not os.path.exists(gpkg):
        print(f"[warn] no tatort geopackage at {gpkg}; scoring WITHOUT urban mask")
        return None
    with rasterio.open(ref_raster) as src:
        transform=src.transform; shape=(src.height,src.width); crs=src.crs; b=src.bounds
    t = gpd.read_file(gpkg, layer="Tatorter_2023")
    if t.crs != crs: t = t.to_crs(crs)
    sub = t.cx[b.left:b.right, b.bottom:b.top]
    shapes = [(g,1) for g in sub.geometry if g is not None and not g.is_empty]
    print(f"[info] {len(shapes)} tatort polygons intersect the STH clip")
    if not shapes:
        return None
    return rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8)


def taper_window(size, taper):
    taper=int(max(1,min(taper,size//2)))
    ramp=np.sin(np.linspace(0,np.pi/2,taper,dtype=np.float32))**2
    line=np.ones(size,np.float32); line[:taper]=ramp; line[-taper:]=ramp[::-1]
    return np.maximum(np.outer(line,line).astype(np.float32),1e-4)

def positions(total,patch,stride):
    if total<=patch: return [0]
    pos=list(range(0,total-patch+1,stride))
    if pos[-1]!=total-patch: pos.append(total-patch)
    return pos

def to_prob(chw):
    s=chw.sum(0)
    if chw.min()>=-1e-3 and chw.max()<=1+1e-3 and np.allclose(s,1,atol=5e-2): return chw
    z=chw-chw.max(0,keepdims=True); e=np.exp(z); return e/e.sum(0,keepdims=True)

def predict_full(model, image):
    H,W=image.shape; pin=image
    ph,pw=max(0,PATCH-H),max(0,PATCH-W)
    if ph or pw: pin=np.pad(image,((0,ph),(0,pw)),mode="reflect")
    pH,pW=pin.shape; win=taper_window(PATCH,OVERLAP); stride=PATCH-OVERLAP
    acc=np.zeros((N_CLASSES,pH,pW),np.float32); wsum=np.zeros((pH,pW),np.float32)
    with torch.no_grad():
        for y in positions(pH,PATCH,stride):
            for x in positions(pW,PATCH,stride):
                p=pin[y:y+PATCH,x:x+PATCH]
                t=torch.from_numpy(np.ascontiguousarray(p)).float()[None,None].to(DEVICE)
                pr=to_prob(model(t)[0].cpu().numpy())
                acc[:,y:y+PATCH,x:x+PATCH]+=pr*win[None]
                wsum[y:y+PATCH,x:x+PATCH]+=win
    acc/=np.maximum(wsum[None],1e-8)
    return acc[1][:H,:W]

def load_model(path, drop_le):
    m=BWTreeNet(n_class=N_CLASSES).to(DEVICE)
    sd=torch.load(path,map_location=DEVICE,weights_only=False)
    sd={k:v for k,v in sd.items()
        if 'SharpFeature.weight' not in k and 'SharpOri.weight' not in k}
    m.load_state_dict(sd,strict=False)
    if drop_le:
        import torch.nn as nn
        class _P(nn.Module):
            def forward(self,x): return x,None
        m.le=_P()
    m.eval(); return m


def main():
    with rasterio.open(VAL_CLIP) as s:
        img = s.read(1).astype(np.float32); nod = s.nodata
    print(f"[info] STH clip {img.shape}, range {img.min():.0f}-{img.max():.0f}")

    gt_all = rasterize_labels(VAL_SHP, VAL_CLIP)
    print(f"[info] GT tree pixels: {gt_all.sum():,} ({100*gt_all.mean():.1f}%)")

    tat = tatort_mask(VAL_CLIP, TATORT)
    nod = 0.0 if nod is None else nod
    valid = (img != nod)
    if tat is not None:
        valid &= (tat > 0)
    print(f"[info] valid (in-urban, non-nodata) pixels: {100*valid.mean():.1f}%\n")

    results={}
    for name,path in MODELS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}"); continue
        model=load_model(path, name in DROP_LE)
        prob=predict_full(model, img)
        results[name]={t: compute_metrics((prob>=t)&valid, (gt_all>0)&valid)
                       for t in THRESHOLDS}
        del model; torch.cuda.empty_cache()
        print(f"[done] {name}")

    print("\n"+"="*72)
    print("STH 1970 (masked to tatort) -- second-city validation")
    print("="*72)
    for t in THRESHOLDS:
        print(f"\n--- threshold {t} ---")
        print(f"{'model':>6} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'FP':>12}")
        for n in MODELS:
            if n not in results: continue
            m=results[n][t]
            print(f"{n:>6} {m['IoU']:>8.4f} {m['F1']:>8.4f} {m['Precision']:>8.4f} "
                  f"{m['Recall']:>8.4f} {m['FP']:>12,}")

    print("\n"+"="*72)
    print("VERDICT (threshold 0.5) -- does v18d beat v12 on a SECOND city?")
    print("="*72)
    if "v12" in results:
        ref=results["v12"][0.5]['IoU']
        for n in MODELS:
            if n not in results: continue
            m=results[n][0.5]; d=m['IoU']-ref
            tag="(reference)" if n=="v12" else f"({d:+.4f} vs v12)"
            print(f"  {n:>5}: IoU {m['IoU']:.4f}  Prec {m['Precision']:.4f}  "
                  f"Rec {m['Recall']:.4f}   {tag}")
        print("\nIf v18d > v12 here as well, the ranking holds across two cities.")


if __name__ == "__main__":
    main()
