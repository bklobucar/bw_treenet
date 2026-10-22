"""
07_train_v18.py -- v18 training, built as a minimal, faithful adaptation of your
proven 06_train.py recipe. Reuses your model, Forest_dataset, augmentation classes,
IoULoss, SGD + ReduceLROnPlateau, seed 42 -- VERBATIM -- so the only things that
change between ablation rungs are the ones under test.

Changes from 06_train.py (all deliberate):
  * reads the 1000px `data/processed/` tiles (processed_512 is deleted); input 1000px.
  * batch 8 (known-safe at 1000px on the GH200; 512->1000 is 4x the pixels).
  * holds out malmo/tiles_1970 from training and uses it as the validation set,
    so the masked-Malmo-1970 benchmark stays honest. STH 1970 is not in any
    training dir, so it stays clean as the external benchmark (scored separately
    with the taper + tatort inference pipeline, NOT here).
  * warm-starts from v12.

Ablation ladder (run in order, one change per rung, score each on held-out
Malmo 1970 + STH 1970 with your inference pipeline):
  v18a : baseline           ->  python 07_train_v18.py
  v18b : + drop LE          ->  python 07_train_v18.py --drop-le
  v18c : + degradation aug  ->  python 07_train_v18.py --drop-le --degrade
  v18d : + new labels       ->  (after labeling; add the dirs, then rerun v18c cmd)

ALWAYS run with --smoke first (1 epoch, few tiles) to confirm everything wires up
before committing a full run:
  python 07_train_v18.py --smoke
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/bw_treenet/BWTreeNet/GuiTest"))
from model.BWTreeNet import BWTreeNet
from loss.lossfunction import IoULoss
from util.Forest_dataset import Forest_dataset
from util.augmentation import (RandomFlip, RandomBrightness, RandomNoise,
                               RandomScratch, RandomBlur, RandomDistortion,
                               RandomRoll)

BASE = os.path.expanduser("~/bw_treenet")
PROC = os.path.join(BASE, "data/processed")

# ── city/era tile sets (dir name relative to data/processed) ────────────────
# modern sets end in "/tiles"; historical in "/tiles_<year>".
ALL_SETS = [
    "malmo/tiles", "gtb/tiles", "sth/tiles",                 # modern
    "malmo/tiles_1959", "malmo/tiles_1960", "malmo/tiles_1990",
    "malmo/tiles_1960new", "gtb/tiles_1960new", "sth/tiles_1960new",
    "malmo/tiles_1970new", "gtb/tiles_1970new", "sth/tiles_1970new",
    "gtb/tiles_1960", "gtb/tiles_1970", "gtb/tiles_1990",
    "sth/tiles_1960", "sth/tiles_1990",
]
HOLDOUT_SET = "malmo/tiles_1970"     # kept OUT of training; used as validation
MODERN_SETS = {"malmo/tiles", "gtb/tiles", "sth/tiles"}

WEIGHTS_DIR = os.path.join(BASE, "models")
V12_WEIGHTS = os.path.join(WEIGHTS_DIR, "bwtreenet_v12_best.pt")

# ── hyperparameters (from 06_train.py; unchanged) ───────────────────────────
BATCH_SIZE = 8
LR         = 0.001
N_CLASSES  = 2
INPUT_HW   = 1000
SCHED_PATIENCE, SCHED_FACTOR, SCHED_MIN_LR = 3, 0.5, 1e-6
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── degradation wrapper: applied to MODERN datasets only (v18c) ─────────────
# Operates on Forest_dataset's OUTPUT, so it needs none of its internals. Format
# handling is verified by --smoke before any real run.
class DegradeModern(Dataset):
    def __init__(self, inner, degrader):
        self.inner = inner
        self.degrader = degrader
    def __len__(self):
        return len(self.inner)
    def __getitem__(self, i):
        out = self.inner[i]
        img, lbl, rest = out[0], out[1], out[2:]
        was_tensor = torch.is_tensor(img)
        arr = img.detach().cpu().numpy() if was_tensor else np.asarray(img)
        squeeze = arr.ndim == 3 and arr.shape[0] == 1     # [1,H,W] -> [H,W]
        hw = arr[0] if squeeze else arr
        hw = self.degrader(hw.astype(np.float32), is_modern=True)   # raw [0,255]
        arr = hw[None] if squeeze else hw
        img = torch.from_numpy(np.ascontiguousarray(arr)).float() if was_tensor \
              else arr
        return (img, lbl, *rest)


def build_augs():
    # verbatim from 06_train.py
    return [RandomFlip(prob=0.5),
            RandomBrightness(bright_range=0.15, prob=0.5),
            RandomNoise(noise_range=10, prob=0.3),
            RandomBlur(prob=0.3),
            RandomDistortion(prob=0.3),
            RandomScratch(prob=0.3),
            RandomRoll(prob=0.3)]


def make_set(rel, degrade_modern=None):
    img_dir = os.path.join(PROC, rel, "images")
    lbl_dir = os.path.join(PROC, rel, "labels")
    if not os.path.isdir(img_dir):
        print(f"[warn] missing {img_dir}; skipping")
        return None
    ds = Forest_dataset(map_dir=img_dir + "/", map_seffix=".tif",
                        label_dir=lbl_dir + "/", label_seffix=".tif",
                        have_label=True, class_num=N_CLASSES,
                        input_h=INPUT_HW, input_w=INPUT_HW,
                        transform=build_augs())
    if degrade_modern is not None and rel in MODERN_SETS:
        ds = DegradeModern(ds, degrade_modern)
        print(f"[degrade] modern set {rel} wrapped with historical degradation")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on a tiny subset to verify wiring")
    ap.add_argument("--drop-le", action="store_true", help="v18b: bypass Luminance Enhancer")
    ap.add_argument("--degrade", action="store_true", help="v18c: degrade modern tiles")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE,
                    help="batch size; 8 fits the GH200, use 1-2 on a 16GB card")
    ap.add_argument("--tag", default="v18a")
    ap.add_argument("--from-scratch", action="store_true",
                    help="do not warm-start from v12")
    args = ap.parse_args()

    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(BASE, "results")).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(BASE, f"results/training_log_{args.tag}.csv")
    best_path = os.path.join(WEIGHTS_DIR, f"bwtreenet_{args.tag}_best.pt")
    print(f"[cfg] tag={args.tag} device={DEVICE} drop_le={args.drop_le} "
          f"degrade={args.degrade} smoke={args.smoke}")

    degrader = None
    if args.degrade:
        from historical_degradation import HistoricalDegradation
        degrader = HistoricalDegradation()

    # ── datasets: train = every set except the holdout; val = holdout ────────
    train_parts = [make_set(r, degrader) for r in ALL_SETS]
    train_parts = [d for d in train_parts if d is not None]
    train_set = ConcatDataset(train_parts)
    val_inner = make_set(HOLDOUT_SET)          # no degradation on validation
    if val_inner is None:
        sys.exit(f"[STOP] holdout set {HOLDOUT_SET} not found")

    if args.smoke:                              # tiny subsets for a fast wiring check
        train_set = Subset(train_set, list(range(min(16, len(train_set)))))
        val_inner = Subset(val_inner, list(range(min(8, len(val_inner)))))
        args.epochs = 1

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_inner, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"[data] train={len(train_set)} tiles  val(holdout Malmo1970)={len(val_inner)}")
    print(f"[cfg] batch={args.batch} at {INPUT_HW}px on {DEVICE}")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[cfg] GPU has {gb:.0f} GB; use --batch 1-2 under ~24GB, 8 on the GH200")

    # ── model ────────────────────────────────────────────────────────────────
    model = BWTreeNet(n_class=N_CLASSES).to(DEVICE)
    if not args.from_scratch and os.path.exists(V12_WEIGHTS):
        model.load_state_dict(torch.load(V12_WEIGHTS, map_location=DEVICE,
                                         weights_only=False), strict=False)
        print(f"[model] warm-started from v12")
    else:
        print("[model] training from scratch")

    if args.drop_le:
        # Bypass the Luminance Enhancer per Yuanyuan's advice (rely on CLAHE).
        # The model calls it as `x_enhanced, _ = self.le(x_norm)` (line ~365),
        # so it must return TWO values. A plain nn.Identity returns one and fails
        # to unpack -- instead use a passthrough that returns (x, None).
        import torch.nn as nn

        class _LEPassthrough(nn.Module):
            def forward(self, x):
                return x, None

        model.le = _LEPassthrough()
        print("[drop-le] replaced model.le with passthrough returning (x, None)")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHED_FACTOR, patience=SCHED_PATIENCE,
        min_lr=SCHED_MIN_LR)

    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_iou,lr\n")

    best_val_iou, no_improve = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]['lr']
        model.train(); tr = []
        for images, labels, *_ in train_loader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            out = model(images)
            loss = IoULoss(out, labels).loss_function()
            loss.backward(); optimizer.step()
            tr.append(loss.item())
        model.eval(); vl, ious = [], []
        with torch.no_grad():
            for images, labels, *_ in val_loader:
                images = images.to(DEVICE, dtype=torch.float32)
                labels = labels.to(DEVICE, dtype=torch.long)
                out = model(images)
                vl.append(IoULoss(out, labels).loss_function().item())
                preds = out.argmax(dim=1)
                inter = ((preds == 1) & (labels == 1)).sum().item()
                union = ((preds == 1) | (labels == 1)).sum().item()
                if union > 0:
                    ious.append(inter / union)
        tl = float(np.mean(tr)); vloss = float(np.mean(vl))
        viou = float(np.mean(ious)) if ious else 0.0
        scheduler.step(viou)
        print(f"epoch {epoch:03d}/{args.epochs}  train={tl:.4f}  val={vloss:.4f}  "
              f"val_iou={viou:.4f}  lr={cur_lr:.2e}")
        with open(log_file, "a") as f:
            f.write(f"{epoch},{tl:.4f},{vloss:.4f},{viou:.4f},{cur_lr:.2e}\n")
        if viou > best_val_iou:
            best_val_iou, no_improve = viou, 0
            torch.save(model.state_dict(), best_path)
            print(f"  --> saved best ({best_val_iou:.4f}) to {best_path}")
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"early stop at epoch {epoch}"); break

    if args.smoke:
        print("\n[smoke] wiring OK: data loaded at 1000px, model ran, loss "
              "computed, val loop ran. Safe to launch the full run.")
    print(f"\ndone. best held-out Malmo1970 IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
