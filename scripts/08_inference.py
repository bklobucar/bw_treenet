# LEGACY (v17): kept exactly as it was run on Arrhenius for v17 evaluation.
# Paths are hardcoded to the cluster project dir. Production inference is
# scripts/19_production_inference.py.
"""
08_inference.py
---------------
Run trained BWTreeNet on historical panchromatic tiles.
Inputs : directory of GeoTiff tiles (single-band, uint8, 0.5m, EPSG:3006)
Outputs: matching directory of binary prediction GeoTiffs + soft probability maps

Strategy: sliding 1000x1000 window with 50px overlap; overlapping regions
averaged before thresholding to reduce edge artefacts.

Usage:
    python 08_inference.py                        # defaults to Malmo 1960s
    python 08_inference.py --input_dir <path> --output_dir <path>
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm

# ── BWTreeNet import ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/BWTreeNet/GuiTest"))
from model.BWTreeNet import BWTreeNet

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR  = os.path.expanduser(
    "/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/raw/malmo/1960_OF_gray_mmo/")
DEFAULT_OUTPUT_DIR = os.path.expanduser(
    "/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/inference/malmo_1960s_v12/")
DEFAULT_WEIGHTS    = os.path.expanduser(
    "/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/models/bwtreenet_v17_best.pt")

PATCH_SIZE  = 512    # pixels — must match training
OVERLAP     = 100    # pixels overlap on each side
THRESHOLD   = 0.5    # probability threshold for binary map
N_CLASSES   = 2
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_model(weights_path):
    model = BWTreeNet(n_class=N_CLASSES).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded weights from {weights_path}")
    return model


def normalise(arr):
    """Normalise uint8 image to [0, 1] float32."""
    return arr.astype(np.float32)  # keep raw 0-255 range for LE module


def pad_to_multiple(arr, patch_size, overlap):
    """Pad image so dimensions are exact multiples of (patch_size - overlap)."""
    step = patch_size - overlap
    h, w = arr.shape
    pad_h = (step - (h - overlap) % step) % step
    pad_w = (step - (w - overlap) % step) % step
    arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
    return arr_padded, pad_h, pad_w


def run_inference_on_tile(model, image_arr):
    """
    Sliding window inference with overlap averaging.
    image_arr : 2D numpy array, float32, [0,1]
    returns   : prob_map (float32, same shape), binary_map (uint8, same shape)
    """
    H, W       = image_arr.shape
    step       = PATCH_SIZE - OVERLAP

    # pad so we cover the full image
    img_pad, pad_h, pad_w = pad_to_multiple(image_arr, PATCH_SIZE, OVERLAP)
    pH, pW = img_pad.shape

    prob_acc   = np.zeros((pH, pW), dtype=np.float32)   # accumulated probabilities
    count_acc  = np.zeros((pH, pW), dtype=np.float32)   # overlap count

    ys = list(range(0, pH - PATCH_SIZE + 1, step))
    xs = list(range(0, pW - PATCH_SIZE + 1, step))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img_pad[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                # model expects N x C x H x W
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)

                output = model(tensor)                        # N x 2 x H x W
                prob   = output  # model already applies softmax internally            # N x 2 x H x W
                prob_tree = prob[0, 1].cpu().numpy()          # H x W, tree class

                prob_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE]  += prob_tree
                count_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0

    # average overlapping regions
    prob_map = prob_acc / np.maximum(count_acc, 1e-6)

    # crop back to original size
    prob_map = prob_map[:H, :W]

    binary_map = (prob_map >= THRESHOLD).astype(np.uint8)

    return prob_map, binary_map


def process_tile(model, tif_path, output_dir):
    """Run inference on a single GeoTiff tile and save outputs."""
    tif_path = Path(tif_path)
    stem     = tif_path.stem

    out_binary = Path(output_dir) / f"{stem}_pred_binary.tif"
    out_prob   = Path(output_dir) / f"{stem}_pred_prob.tif"

    if out_binary.exists():
        print(f"  Skipping {stem} (output already exists)")
        return

    with rasterio.open(tif_path) as src:
        image    = src.read(1).astype(np.float32)   # H x W
        profile  = src.profile.copy()
        transform = src.transform
        crs      = src.crs

    print(f"  {stem}: {image.shape[1]}x{image.shape[0]}px  "
          f"res={profile['transform'].a:.2f}m")

    image_norm = normalise(image)

    prob_map, binary_map = run_inference_on_tile(model, image_norm)

    # ── save binary prediction ─────────────────────────────────────────────
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw',
                   nodata=255)
    with rasterio.open(out_binary, 'w', **profile) as dst:
        dst.write(binary_map, 1)

    # ── save probability map ───────────────────────────────────────────────
    profile.update(dtype=rasterio.float32, count=1, compress='lzw',
                   nodata=-1.0)
    with rasterio.open(out_prob, 'w', **profile) as dst:
        dst.write(prob_map, 1)

    tree_pct = binary_map.mean() * 100
    print(f"    → tree cover: {tree_pct:.1f}%  saved to {output_dir}")


def main(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(input_dir.glob("*.tif"))
    # exclude aux files that rasterio can't open
    tif_files = [f for f in tif_files if not f.name.endswith(".aux.xml")]

    print(f"Found {len(tif_files)} tiles in {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Patch: {PATCH_SIZE}px  Overlap: {OVERLAP}px  Threshold: {THRESHOLD}")
    print()

    model = load_model(args.weights)

    for i, tif_path in enumerate(tif_files, 1):
        print(f"[{i}/{len(tif_files)}] {tif_path.name}")
        process_tile(model, tif_path, output_dir)

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BWTreeNet inference on historical tiles")
    parser.add_argument("--input_dir",  default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights",    default=DEFAULT_WEIGHTS)
    parser.add_argument("--threshold",  type=float, default=THRESHOLD)
    args = parser.parse_args()

    THRESHOLD = args.threshold
    main(args)