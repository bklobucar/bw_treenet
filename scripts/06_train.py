# LEGACY (v17): kept exactly as it was run on Arrhenius for v17 training.
# Paths are hardcoded to the cluster project dir. v18 onwards (incl. v19)
# trains with scripts/07_train_v18.py instead.
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
from pathlib import Path

# add BWTreeNet to path
sys.path.insert(0, os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/BWTreeNet/GuiTest"))

from model.BWTreeNet import BWTreeNet
from loss.lossfunction import IoULoss
from util.Forest_dataset import Forest_dataset
from util.augmentation import (RandomFlip, RandomBrightness, RandomNoise,
                                RandomScratch, RandomBlur, RandomDistortion,
                                RandomRoll)

# ── config ─────────────────────────────────────────────────────────────────
IMAGES_DIRS = [
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/sth/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb_1960/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/sth_1960/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1959/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1960/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1970/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1990/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb_1970/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb_1990/images/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/sth_1990/images/"),
]
LABELS_DIRS = [
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/sth/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb_1960/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/sth_1960/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1959/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1960/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1970/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/malmo_1990/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb_1970/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/gtb_1990/labels/"),
    os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/data/processed_512/sth_1990/labels/"),
]
WEIGHTS_DIR = os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/models/")
LOG_FILE    = os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/training_log_v17.csv")
PRETRAINED  = os.path.expanduser(
    "/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/models/NONE_train_from_scratch")  # resume from v3 best

Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.expanduser("/nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results")).mkdir(parents=True, exist_ok=True)

EPOCHS      = 80           # cap at 40; ReduceLROnPlateau will manage descent
BATCH_SIZE  = 16
LR          = 0.001       # 10x lower than v3 — fine-tuning from checkpoint
VAL_SPLIT   = 0.2
N_CLASSES   = 2
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_EVERY  = 10

# ReduceLROnPlateau config
SCHED_PATIENCE = 3         # epochs without improvement before LR halves
SCHED_FACTOR   = 0.5       # halve the LR
SCHED_MIN_LR   = 1e-6      # floor
EARLY_STOP_PATIENCE = 10   # stop if no improvement for this many epochs

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"Using device: {DEVICE}")
print(f"LR={LR}  patience={SCHED_PATIENCE}  early_stop={EARLY_STOP_PATIENCE}")

# ── augmentation ───────────────────────────────────────────────────────────
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomBrightness(bright_range=0.15, prob=0.5),
    RandomNoise(noise_range=10, prob=0.3),
    RandomBlur(prob=0.3),
    RandomDistortion(prob=0.3),
    RandomScratch(prob=0.3),
    RandomRoll(prob=0.3),
]

# ── dataset ────────────────────────────────────────────────────────────────
datasets = []
for img_dir, lbl_dir in zip(IMAGES_DIRS, LABELS_DIRS):
    ds = Forest_dataset(
        map_dir=img_dir,
        map_seffix=".tif",
        label_dir=lbl_dir,
        label_seffix=".tif",
        have_label=True,
        class_num=N_CLASSES,
        input_h=512,
        input_w=512,
        transform=augmentation_methods
    )
    datasets.append(ds)

full_dataset = ConcatDataset(datasets)
print(f"Total dataset size: {len(full_dataset)}")

n_val   = int(len(full_dataset) * VAL_SPLIT)
n_train = len(full_dataset) - n_val
train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {n_train}  Val: {n_val}")

# ── model ──────────────────────────────────────────────────────────────────
model = BWTreeNet(n_class=N_CLASSES).to(DEVICE)

if os.path.exists(PRETRAINED):
    state = torch.load(PRETRAINED, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained weights from {PRETRAINED}")
else:
    print("WARNING: no pretrained weights found, training from scratch")

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                             weight_decay=1e-4)

# ReduceLROnPlateau: watches val IoU (mode='max'), reduces LR when it stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=SCHED_FACTOR,
    patience=SCHED_PATIENCE,
    min_lr=SCHED_MIN_LR,
)

# ── logging ────────────────────────────────────────────────────────────────
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,val_loss,val_iou,lr\n")

# ── training loop ──────────────────────────────────────────────────────────
best_val_iou        = 0.0
epochs_no_improve   = 0

for epoch in range(1, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]['lr']

    # ── train ──────────────────────────────────────────────────────────────
    model.train()
    train_losses = []

    for images, labels, _ in train_loader:
        images = images.to(DEVICE, dtype=torch.float32)
        labels = labels.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)

        loss_fn = IoULoss(outputs, labels)
        loss = loss_fn.loss_function()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # ── validate ───────────────────────────────────────────────────────────
    model.eval()
    val_losses  = []
    iou_scores  = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)

            outputs = model(images)

            loss_fn = IoULoss(outputs, labels)
            val_losses.append(loss_fn.loss_function().item())

            preds = outputs.argmax(dim=1)
            inter = ((preds == 1) & (labels == 1)).sum().item()
            union = ((preds == 1) | (labels == 1)).sum().item()
            if union > 0:
                iou_scores.append(inter / union)

    avg_val_loss = np.mean(val_losses)
    avg_val_iou  = np.mean(iou_scores) if iou_scores else 0.0

    # step scheduler on val IoU
    scheduler.step(avg_val_iou)

    print(f"Epoch {epoch:03d}/{EPOCHS}  "
          f"train_loss={avg_train_loss:.4f}  "
          f"val_loss={avg_val_loss:.4f}  "
          f"val_iou={avg_val_iou:.4f}  "
          f"lr={current_lr:.2e}")

    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_train_loss:.4f},"
                f"{avg_val_loss:.4f},{avg_val_iou:.4f},{current_lr:.2e}\n")

    # ── save best ──────────────────────────────────────────────────────────
    if avg_val_iou > best_val_iou:
        best_val_iou      = avg_val_iou
        epochs_no_improve = 0
        torch.save(model.state_dict(),
                   os.path.join(WEIGHTS_DIR, "bwtreenet_v10b_best.pt"))
        print(f"  --> saved best model (val_iou={best_val_iou:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  (no improvement: {epochs_no_improve}/{EARLY_STOP_PATIENCE})")

    if epoch % SAVE_EVERY == 0:
        torch.save(model.state_dict(),
                   os.path.join(WEIGHTS_DIR,
                                f"bwtreenet_all_cities_epoch{epoch}.pt"))

    # ── early stopping ─────────────────────────────────────────────────────
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
        break

print(f"\nTraining complete. Best val IoU: {best_val_iou:.4f}")
print(f"Log saved to {LOG_FILE}")