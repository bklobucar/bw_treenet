"""
09_accuracy_assessment.py
--------------------------
Accuracy assessment of BWTreeNet inference against manually digitised
tree crown polygons for Malmö 1959.

Inputs:
  - Inference probability/binary raster (from 08_inference.py)
  - Manual label shapefile (labels_malmo_1959.shp)
  - Clipped historical image (mmo_clip.tif) — defines assessment extent

Outputs:
  - Printed metrics: IoU, F1, precision, recall, OA
  - assessment_results.csv with per-threshold metrics
  - Optional: difference raster (TP/FP/FN/TN)

Usage:
    python 09_accuracy_assessment.py
    python 09_accuracy_assessment.py --pred_dir <path> --val_dir <path>
"""

import os
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from pathlib import Path
import csv

# ── defaults ───────────────────────────────────────────────────────────────
BASE = "/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet"

DEFAULT_PRED_DIR  = f"{BASE}/results/inference/malmo_1960s_v12"
DEFAULT_VAL_DIR   = f"{BASE}/data/processed/malmo/validation"
DEFAULT_OUTPUT    = f"{BASE}/results/accuracy_assessment_v12.csv"
DEFAULT_THRESHOLD = 0.5


def rasterize_labels(shp_path, reference_raster_path):
    """Rasterize shapefile to match reference raster extent and resolution."""
    gdf = gpd.read_file(shp_path)
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
        crs = src.crs

    # reproject labels if needed
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if len(shapes) == 0:
        return np.zeros(shape, dtype=np.uint8)

    burned = rasterize(shapes, out_shape=shape, transform=transform,
                       fill=0, dtype=np.uint8)
    return burned


def compute_metrics(pred_binary, gt_binary):
    """Compute segmentation metrics for tree class."""
    pred = pred_binary.astype(bool)
    gt   = gt_binary.astype(bool)

    TP = (pred & gt).sum()
    FP = (pred & ~gt).sum()
    FN = (~pred & gt).sum()
    TN = (~pred & ~gt).sum()

    total = TP + FP + FN + TN

    iou       = TP / (TP + FP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    recall    = TP / (TP + FN + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    oa        = (TP + TN) / (total + 1e-10)

    return {
        'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN),
        'IoU': round(float(iou), 4),
        'F1': round(float(f1), 4),
        'Precision': round(float(precision), 4),
        'Recall': round(float(recall), 4),
        'OA': round(float(oa), 4),
        'GT_tree_pixels': int(gt.sum()),
        'Pred_tree_pixels': int(pred.sum()),
    }


def find_matching_pred(val_clip_path, pred_dir):
    """Find inference tile(s) that overlap with the validation clip extent."""
    with rasterio.open(val_clip_path) as val_src:
        val_bounds = val_src.bounds
        val_crs    = val_src.crs

    pred_files = sorted(Path(pred_dir).glob("*_pred_prob.tif"))
    matching = []
    for f in pred_files:
        with rasterio.open(f) as src:
            b = src.bounds
            # check overlap
            if (b.left < val_bounds.right and b.right > val_bounds.left and
                b.bottom < val_bounds.top and b.top > val_bounds.bottom):
                matching.append(f)
    return matching


def crop_pred_to_val(pred_path, val_path):
    """Crop and resample prediction raster to match validation raster extent."""
    with rasterio.open(val_path) as val_src:
        val_profile = val_src.profile.copy()
        val_transform = val_src.transform
        val_shape = (val_src.height, val_src.width)
        val_crs = val_src.crs

    with rasterio.open(pred_path) as pred_src:
        pred_data = pred_src.read(1)
        pred_transform = pred_src.transform
        pred_crs = pred_src.crs

    # reproject prediction to match validation extent/resolution
    dest = np.zeros(val_shape, dtype=np.float32)
    reproject(
        source=pred_data,
        destination=dest,
        src_transform=pred_transform,
        src_crs=pred_crs,
        dst_transform=val_transform,
        dst_crs=val_crs,
        resampling=Resampling.bilinear
    )
    return dest


def main(args):
    pred_dir  = args.pred_dir
    val_dir   = args.val_dir
    output    = args.output
    threshold = args.threshold

    val_clip = os.path.join(val_dir, "sth_1970_clip.tif")
    val_shp  = os.path.join(val_dir, "labels_sth_1970.shp")

    print(f"Validation clip: {val_clip}")
    print(f"Validation labels: {val_shp}")
    print(f"Prediction dir: {pred_dir}")
    print(f"Threshold: {threshold}")
    print()

    # rasterize ground truth labels
    print("Rasterizing ground truth labels...")
    gt_binary = rasterize_labels(val_shp, val_clip)
    print(f"  GT tree pixels: {gt_binary.sum()} / {gt_binary.size} "
          f"({gt_binary.mean()*100:.1f}%)")

    # find matching prediction tiles
    matching = find_matching_pred(val_clip, pred_dir)
    print(f"Found {len(matching)} overlapping prediction tiles")

    if len(matching) == 0:
        print("ERROR: No prediction tiles overlap with validation extent")
        print("Check that inference has been run and covers the validation area")
        return

    # merge overlapping tiles if multiple
    if len(matching) == 1:
        pred_prob = crop_pred_to_val(matching[0], val_clip)
    else:
        # average overlapping predictions
        pred_sum   = np.zeros_like(gt_binary, dtype=np.float32)
        pred_count = np.zeros_like(gt_binary, dtype=np.float32)
        for f in matching:
            p = crop_pred_to_val(f, val_clip)
            mask = p > 0
            pred_sum[mask]   += p[mask]
            pred_count[mask] += 1
        pred_prob = np.where(pred_count > 0, pred_sum / pred_count, 0)

    print(f"Prediction prob min/max/mean: {pred_prob.min():.4f} / "
          f"{pred_prob.max():.4f} / {pred_prob.mean():.4f}")

    # evaluate at multiple thresholds
    results = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print()
    print(f"{'Threshold':>10} {'IoU':>8} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'OA':>8}")
    print("-" * 60)

    for t in thresholds:
        pred_binary = (pred_prob >= t).astype(np.uint8)
        metrics = compute_metrics(pred_binary, gt_binary)
        metrics['threshold'] = t
        results.append(metrics)
        print(f"{t:>10.1f} {metrics['IoU']:>8.4f} {metrics['F1']:>8.4f} "
              f"{metrics['Precision']:>10.4f} {metrics['Recall']:>8.4f} "
              f"{metrics['OA']:>8.4f}")

    # save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output}")

    # highlight best threshold
    best = max(results, key=lambda x: x['IoU'])
    print(f"\nBest threshold: {best['threshold']} → IoU={best['IoU']}, "
          f"F1={best['F1']}, Precision={best['Precision']}, "
          f"Recall={best['Recall']}")

    # save difference raster at best threshold
    pred_binary = (pred_prob >= best['threshold']).astype(np.uint8)
    diff = np.zeros_like(gt_binary, dtype=np.uint8)
    diff[(pred_binary == 1) & (gt_binary == 1)] = 1  # TP — green
    diff[(pred_binary == 1) & (gt_binary == 0)] = 2  # FP — red
    diff[(pred_binary == 0) & (gt_binary == 1)] = 3  # FN — orange
    diff[(pred_binary == 0) & (gt_binary == 0)] = 0  # TN — transparent

    with rasterio.open(val_clip) as src:
        profile = src.profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=255)

    diff_path = output.replace('.csv', '_diff.tif')
    with rasterio.open(diff_path, 'w', **profile) as dst:
        dst.write(diff, 1)
    print(f"Difference raster saved to {diff_path}")
    print("  1=TP (correct tree), 2=FP (false positive), "
          "3=FN (missed tree), 0=TN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",  default=DEFAULT_PRED_DIR)
    parser.add_argument("--val_dir",   default=DEFAULT_VAL_DIR)
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    main(args)
