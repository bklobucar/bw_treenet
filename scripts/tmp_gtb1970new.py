import importlib.util, glob, os, numpy as np, rasterio
spec = importlib.util.spec_from_file_location("b24", "scripts/24_baseline_swiss.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
D = "data/processed/gtb/tiles_1970new"
for cfg in ["swiss46_eq_05m", "v19"]:
    m, _ = b.build(cfg); tot = np.zeros(4, np.int64)
    for ip in sorted(glob.glob(f"{D}/images/*.tif")):
        lp = b._match(f"{D}/labels", os.path.basename(ip))
        img = rasterio.open(ip).read(1).astype(np.float32); gt = rasterio.open(lp).read(1) > 0
        mm = b.S10.compute_metrics(b.predict(m, img, 1) >= 0.5, gt)
        tot += [mm["TP"], mm["FP"], mm["FN"], mm["TN"]]
    print(f"==> {cfg} pooled GTB 1970new (10 tiles) IoU@0.5: {b.pooled(tot)['IoU']:.3f}")
