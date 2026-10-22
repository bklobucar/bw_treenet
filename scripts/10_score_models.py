"""
10_score_models.py -- score v12/v18a/v18b/v18c on the SAME masked benchmark.

Evaluates each checkpoint on the held-out Malmo 1970 tiles, with water excluded
via the tatort mask (inside-urban) -- the same masking that lifted v12's measured
precision from 0.53 to 0.72. Uses compute_metrics copied verbatim from
scripts/09_accuracy_assessment.py so numbers are directly comparable.

Run (on a GPU node, env active):
    python scripts/10_score_models.py
"""
import os, sys, glob
import numpy as np
import rasterio
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "BWTreeNet", "GuiTest"))
from model.BWTreeNet import BWTreeNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATCH, OVERLAP, N_CLASSES = 1000, 200, 2
THRESHOLDS = [0.4, 0.5, 0.6]

IMG_DIR = os.path.join(HERE, "data/processed/malmo/tiles_1970/images")
LBL_DIR = os.path.join(HERE, "data/processed/malmo/tiles_1970/labels")
TAT_DIR = os.path.join(HERE, "data/processed/malmo/tiles_1970/tatortmask")
MODELS = {
    "v12":  os.path.join(HERE, "models/bwtreenet_v12_best.pt"),
    "v18a": os.path.join(HERE, "models/bwtreenet_v18a_best.pt"),
    "v18b": os.path.join(HERE, "models/bwtreenet_v18b_best.pt"),
    "v18c": os.path.join(HERE, "models/bwtreenet_v18c_best.pt"),
    "v18d": os.path.join(HERE, "models/bwtreenet_v18d_best.pt"),
    "v19": os.path.join(HERE, "models/bwtreenet_v19_best.pt"),
    "v20": os.path.join(HERE, "models/bwtreenet_v20_best.pt"),
}
DROP_LE = {"v18b", "v18c"}      # these were trained with the LE bypassed


# ---- metrics: verbatim from 09_accuracy_assessment.py ----------------------
def compute_metrics(pred_binary, gt_binary):
    pred = pred_binary.astype(bool); gt = gt_binary.astype(bool)
    TP = (pred & gt).sum(); FP = (pred & ~gt).sum()
    FN = (~pred & gt).sum(); TN = (~pred & ~gt).sum()
    total = TP + FP + FN + TN
    iou = TP/(TP+FP+FN+1e-10); precision = TP/(TP+FP+1e-10)
    recall = TP/(TP+FN+1e-10)
    f1 = 2*precision*recall/(precision+recall+1e-10)
    return dict(TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
                IoU=float(iou), F1=float(f1),
                Precision=float(precision), Recall=float(recall),
                OA=float((TP+TN)/(total+1e-10)))


# ---- taper-blended sliding window ------------------------------------------
def taper_window(size, taper):
    taper = int(max(1, min(taper, size//2)))
    ramp = np.sin(np.linspace(0, np.pi/2, taper, dtype=np.float32))**2
    line = np.ones(size, np.float32); line[:taper] = ramp; line[-taper:] = ramp[::-1]
    return np.maximum(np.outer(line, line).astype(np.float32), 1e-4)

def positions(total, patch, stride):
    if total <= patch: return [0]
    pos = list(range(0, total-patch+1, stride))
    if pos[-1] != total-patch: pos.append(total-patch)
    return pos

def to_prob(chw):
    s = chw.sum(0)
    if chw.min() >= -1e-3 and chw.max() <= 1+1e-3 and np.allclose(s, 1, atol=5e-2):
        return chw
    z = chw - chw.max(0, keepdims=True); e = np.exp(z); return e/e.sum(0, keepdims=True)

def predict_full(model, image):
    H, W = image.shape
    pin = image
    ph, pw = max(0, PATCH-H), max(0, PATCH-W)
    if ph or pw: pin = np.pad(image, ((0,ph),(0,pw)), mode="reflect")
    pH, pW = pin.shape
    win = taper_window(PATCH, OVERLAP); stride = PATCH - OVERLAP
    acc = np.zeros((N_CLASSES, pH, pW), np.float32); wsum = np.zeros((pH, pW), np.float32)
    with torch.no_grad():
        for y in positions(pH, PATCH, stride):
            for x in positions(pW, PATCH, stride):
                p = pin[y:y+PATCH, x:x+PATCH]
                t = torch.from_numpy(np.ascontiguousarray(p)).float()[None,None].to(DEVICE)
                pr = to_prob(model(t)[0].cpu().numpy())
                acc[:, y:y+PATCH, x:x+PATCH] += pr * win[None]
                wsum[y:y+PATCH, x:x+PATCH] += win
    acc /= np.maximum(wsum[None], 1e-8)
    return acc[1][:H, :W]


def load_model(path, drop_le):
    m = BWTreeNet(n_class=N_CLASSES).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = {k: v for k, v in sd.items()
          if 'SharpFeature.weight' not in k and 'SharpOri.weight' not in k}
    m.load_state_dict(sd, strict=False)
    if drop_le:
        import torch.nn as nn
        class _P(nn.Module):
            def forward(self, x): return x, None
        m.le = _P()
    m.eval()
    return m


def main():
    tiles = sorted(glob.glob(os.path.join(IMG_DIR, "*.tif")))
    print(f"[info] {len(tiles)} Malmo 1970 tiles; device {DEVICE}\n")

    results = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: no checkpoint at {path}"); continue
        model = load_model(path, name in DROP_LE)
        agg = {t: dict(TP=0,FP=0,FN=0,TN=0) for t in THRESHOLDS}
        for ip in tiles:
            base = os.path.basename(ip)
            with rasterio.open(ip) as s:
                img = s.read(1).astype(np.float32); nod = s.nodata
            with rasterio.open(os.path.join(LBL_DIR, base)) as s:
                lbl = s.read(1)
            tp_ = os.path.join(TAT_DIR, base)
            tat = None
            if os.path.exists(tp_):
                with rasterio.open(tp_) as s: tat = s.read(1)
            nod = 0.0 if nod is None else nod
            valid = (img != nod)
            if tat is not None:
                valid &= (tat > 0)          # inside urban area (excludes sea)
            prob = predict_full(model, img)
            gt = (lbl > 0) & valid
            for t in THRESHOLDS:
                m_ = compute_metrics((prob >= t) & valid, gt)
                for k in ("TP","FP","FN","TN"): agg[t][k] += m_[k]
        results[name] = agg
        del model; torch.cuda.empty_cache()
        best = max(THRESHOLDS, key=lambda t: agg[t]['TP']/(agg[t]['TP']+agg[t]['FP']+agg[t]['FN']+1e-10))
        print(f"[done] {name}")

    def m_from(c):
        TP,FP,FN = c['TP'],c['FP'],c['FN']
        p = TP/(TP+FP+1e-10); r = TP/(TP+FN+1e-10)
        return TP/(TP+FP+FN+1e-10), 2*p*r/(p+r+1e-10), p, r

    print("\n" + "="*72)
    print("MASKED (inside tatort) Malmo 1970 -- held out from all v18 training")
    print("="*72)
    for t in THRESHOLDS:
        print(f"\n--- threshold {t} ---")
        print(f"{'model':>6} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'FP':>12}")
        for name in MODELS:
            if name not in results: continue
            c = results[name][t]; iou,f1,p,r = m_from(c)
            print(f"{name:>6} {iou:>8.4f} {f1:>8.4f} {p:>8.4f} {r:>8.4f} {c['FP']:>12,}")

    print("\n" + "="*72)
    print("VERDICT (threshold 0.5)")
    print("="*72)
    ref = None
    for name in MODELS:
        if name not in results: continue
        iou,f1,p,r = m_from(results[name][0.5])
        if ref is None: ref = iou
        delta = iou - ref
        tag = "(reference)" if abs(delta) < 1e-9 else f"({delta:+.4f} vs v12)"
        print(f"  {name:>5}: IoU {iou:.4f}  Prec {p:.4f}  Rec {r:.4f}   {tag}")
    print("\nHigher IoU than v12 = the change helped. Compare v18a vs v18b to judge")
    print("the LE removal, and v18b vs v18c to judge degradation augmentation.")


if __name__ == "__main__":
    main()
