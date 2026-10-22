# LEGACY (v17): kept exactly as it was run on Arrhenius to build the v17
# 512px tiles (data/processed_512, since deleted). BASE is hardcoded to the
# cluster project dir. v18 onwards trains on the 1000px data/processed tiles.
"""
retile_512.py
--------------
Retiles existing training data at 512x512px with positive sample filtering.
Keeps only tiles with >= MIN_TREE_COVER % tree coverage.
Applies random flipping augmentation to positive tiles to increase diversity.

Based on advice from BWTreeNet author (Yuanyuan Gui):
- Crop to 512x512 to increase positive sample proportion per tile
- Filter background-heavy tiles
- Target 30-50% tree coverage across training set
- Duplicate/augment positive tiles

Usage:
    python retile_512.py           # process all tile directories
    python retile_512.py --test    # process malmo/tiles only
    python retile_512.py --min_cover 0.10  # minimum 10% tree cover (default)
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────
BASE = "/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet"

TILE_SIZE    = 512    # new tile size
STEP         = 256    # 50% overlap
MIN_COVER    = 0.10   # minimum tree cover fraction to keep tile
AUGMENT      = True   # flip augmentation on positive tiles

# source tile directories (1000x1000 tiles)
# tuple: (src_dir, prefix, min_cover_override or None)
SOURCE_DIRS = [
    # modern — use default MIN_COVER (0.10)
    (f"{BASE}/data/processed/malmo/tiles",   "malmo",      None),
    (f"{BASE}/data/processed/gtb/tiles",     "gtb",        None),
    (f"{BASE}/data/processed/sth/tiles",     "sth",        None),
    # historical — use lower threshold (0.03) to keep more tiles
    (f"{BASE}/data/processed/malmo/tiles_1959", "malmo_1959", 0.03),
    (f"{BASE}/data/processed/malmo/tiles_1960", "malmo_1960", 0.03),
    (f"{BASE}/data/processed/malmo/tiles_1970", "malmo_1970", 0.03),
    (f"{BASE}/data/processed/malmo/tiles_1990", "malmo_1990", 0.03),
    (f"{BASE}/data/processed/gtb/tiles_1960",   "gtb_1960",   0.03),
    (f"{BASE}/data/processed/gtb/tiles_1970",   "gtb_1970",   0.03),
    (f"{BASE}/data/processed/gtb/tiles_1990",   "gtb_1990",   0.03),
    (f"{BASE}/data/processed/sth/tiles_1960",   "sth_1960",   0.03),
    (f"{BASE}/data/processed/sth/tiles_1990",   "sth_1990",   0.03),
]

# output base directory
OUT_BASE = f"{BASE}/data/processed_512"


def flip_variants(img, lbl):
    """Generate flip augmentation variants of an image/label pair."""
    variants = []
    # horizontal flip
    variants.append((np.flip(img, axis=-1).copy(), np.flip(lbl, axis=-1).copy()))
    # vertical flip
    variants.append((np.flip(img, axis=-2).copy(), np.flip(lbl, axis=-2).copy()))
    # both
    variants.append((np.flip(img, axis=(-1,-2)).copy(), np.flip(lbl, axis=(-1,-2)).copy()))
    return variants


def process_source_dir(src_dir, prefix, min_cover, augment, test=False):
    img_src = Path(src_dir) / 'images'
    lbl_src = Path(src_dir) / 'labels'

    if not img_src.exists():
        print(f"  SKIP — no images/ dir: {src_dir}")
        return 0, 0

    # output dirs
    img_out = Path(OUT_BASE) / prefix / 'images'
    lbl_out = Path(OUT_BASE) / prefix / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    source_tiles = sorted(img_src.glob('*.tif'))
    if test:
        source_tiles = source_tiles[:5]  # only first 5 tiles in test mode

    kept = 0
    skipped = 0
    augmented = 0

    for tile_path in source_tiles:
        lbl_path = lbl_src / tile_path.name
        if not lbl_path.exists():
            continue

        with rasterio.open(tile_path) as src:
            img = src.read(1)  # H x W
            profile = src.profile.copy()
            transform = src.transform

        with rasterio.open(lbl_path) as src:
            lbl = src.read(1)
            lbl_profile = src.profile.copy()

        H, W = img.shape

        # extract 512x512 sub-tiles with 50% overlap
        for y in range(0, H - TILE_SIZE + 1, STEP):
            for x in range(0, W - TILE_SIZE + 1, STEP):
                img_tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                lbl_tile = lbl[y:y+TILE_SIZE, x:x+TILE_SIZE]

                if img_tile.shape != (TILE_SIZE, TILE_SIZE):
                    continue

                tree_cover = lbl_tile.mean()

                # skip background-heavy tiles
                if tree_cover < min_cover:
                    skipped += 1
                    continue

                # compute tile transform
                tile_transform = rasterio.transform.from_bounds(
                    transform.c + x * transform.a,
                    transform.f + (y + TILE_SIZE) * transform.e,
                    transform.c + (x + TILE_SIZE) * transform.a,
                    transform.f + y * transform.e,
                    TILE_SIZE, TILE_SIZE)

                stem = tile_path.stem
                tile_name = f"{stem}_{y:05d}_{x:05d}.tif"

                # save original
                prof_img = profile.copy()
                prof_img.update(width=TILE_SIZE, height=TILE_SIZE,
                               transform=tile_transform, count=1,
                               compress='deflate')
                prof_lbl = lbl_profile.copy()
                prof_lbl.update(width=TILE_SIZE, height=TILE_SIZE,
                               transform=tile_transform, count=1,
                               compress='deflate', dtype=rasterio.uint8,
                               nodata=255)

                with rasterio.open(img_out / tile_name, 'w', **prof_img) as dst:
                    dst.write(img_tile, 1)
                with rasterio.open(lbl_out / tile_name, 'w', **prof_lbl) as dst:
                    dst.write(lbl_tile.astype(np.uint8), 1)
                kept += 1

                # augment with flips
                if augment:
                    for i, (img_aug, lbl_aug) in enumerate(
                            flip_variants(img_tile[np.newaxis], lbl_tile[np.newaxis])):
                        aug_name = f"{stem}_{y:05d}_{x:05d}_aug{i}.tif"
                        with rasterio.open(img_out / aug_name, 'w', **prof_img) as dst:
                            dst.write(img_aug[0], 1)
                        with rasterio.open(lbl_out / aug_name, 'w', **prof_lbl) as dst:
                            dst.write(lbl_aug[0].astype(np.uint8), 1)
                        augmented += 1

    total = kept + augmented
    print(f"  {prefix}: {kept} kept, {skipped} skipped, {augmented} augmented → {total} total tiles")
    return kept, augmented


def main(args):
    dirs = SOURCE_DIRS[:1] if args.test else SOURCE_DIRS
    min_cover = args.min_cover

    print(f"Tile size: {TILE_SIZE}px  Step: {STEP}px  Min cover: {min_cover*100:.0f}%  Augment: {AUGMENT}")
    print(f"Output: {OUT_BASE}\n")

    total_kept = 0
    total_aug  = 0

    for entry in dirs:
        src_dir, prefix, cover_override = entry
        effective_cover = cover_override if cover_override is not None else min_cover
        print(f"Processing {prefix} (min_cover={effective_cover*100:.0f}%)...")
        k, a = process_source_dir(src_dir, prefix, effective_cover, AUGMENT, args.test)
        total_kept += k
        total_aug  += a

    print(f"\nTotal: {total_kept} kept + {total_aug} augmented = {total_kept + total_aug} tiles")
    print(f"\nNext step: update scripts/06_train.py IMAGES_DIRS to point to {OUT_BASE}/<city>/images/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',      action='store_true',
                        help='Process only malmo/tiles (first 5 source tiles)')
    parser.add_argument('--min_cover', type=float, default=0.10,
                        help='Minimum tree cover fraction to keep tile (default 0.10)')
    args = parser.parse_args()
    main(args)
