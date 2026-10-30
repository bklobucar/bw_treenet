"""Label value 255 = ignore in training (Forest_dataset, IoULoss, validation IoU in 06, 07 and 25).

(a) tiles without 255: loader output, loss and validation IoU are identical to the code before the change;
(b) pixels labelled 255 contribute nothing to the loss (value or gradient), the IoU term or the validation IoU,
    whatever the prediction there.
    python -m pytest tests/test_ignore_255.py        (CPU; needs torch, numpy, tifffile, cv2, albumentations)
"""
import os, re, sys
import numpy as np
import pytest
import tifffile
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "BWTreeNet", "GuiTest"))
from loss.lossfunction import IoULoss                       # noqa: E402
from util.Forest_dataset import Forest_dataset              # noqa: E402


# ── the code before the change, verbatim, as the reference ──────────────────
def old_read_label(path):
    image = tifffile.imread(path)
    image = np.squeeze(image)
    image[image > 1] = 1
    return image


def old_loss(input, target):
    N = len(input)
    ce_loss = F.nll_loss(torch.log(input + 1e-16), target)
    target_onehot = IoULoss.to_one_hot(target, 2)
    pred_tree = input[:, 1, :, :].contiguous().view(N, -1)
    target_tree = target_onehot[:, 1, :, :].contiguous().view(N, -1)
    inter = (pred_tree * target_tree).sum(1)
    union = (pred_tree + target_tree - pred_tree * target_tree).sum(1)
    iou_tree = inter / (union + 1e-16)
    iou_loss = 1 - iou_tree.mean()
    return 0.5 * ce_loss + 0.5 * iou_loss


def old_val_iou(preds, labels):
    inter = ((preds == 1) & (labels == 1)).sum().item()
    union = ((preds == 1) | (labels == 1)).sum().item()
    return inter, union


# ── the validation IoU lines as they stand in the training scripts ───────────
TRAIN_SCRIPTS = ["06_train.py", "07_train_v18.py", "25_train_finetune.py"]


def new_val_iou(preds, labels, script="07_train_v18.py"):
    src = open(os.path.join(ROOT, "scripts", script), encoding="utf-8").read()
    lines = []
    for v in ("inter", "union"):
        found = re.findall(rf"^\s*({v} = .*)$", src, re.M)
        assert len(found) == 1, f"{script}: expected one '{v} = ' line, found {len(found)}"
        lines.append(found[0])
    ns = dict(preds=preds, labels=labels)
    for line in lines:
        exec(line, {}, ns)
    return ns["inter"], ns["union"]


def new_loss(input, target):
    return IoULoss(input, target).loss_function()


def softmaxed(n=2, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, 2, h, w, generator=g), dim=1)


def labels01(n=2, h=32, w=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, h, w, generator=g) > 0.6).long()


# ── (a) no 255: identical to before ──────────────────────────────────────────
def test_loader_without_255_unchanged(tmp_path):
    a = np.random.default_rng(0).choice([0, 1, 2, 7], size=(40, 40)).astype(np.uint8)
    p = str(tmp_path / "t.tif")
    tifffile.imwrite(p, a)
    assert np.array_equal(Forest_dataset.read_image(None, p, "labels"), old_read_label(p))


def test_loss_without_255_unchanged():
    for seed in range(5):
        x, y = softmaxed(seed=seed), labels01(seed=seed + 10)
        assert torch.equal(new_loss(x, y), old_loss(x, y))


@pytest.mark.parametrize("script", TRAIN_SCRIPTS)
def test_val_iou_without_255_unchanged(script):
    for seed in range(3):
        preds, labels = labels01(seed=3 + seed), labels01(seed=13 + seed)
        assert new_val_iou(preds, labels, script) == old_val_iou(preds, labels)


# ── (b) 255 contributes nothing ──────────────────────────────────────────────
def test_loader_keeps_255(tmp_path):
    a = np.array([[0, 1, 2, 255], [7, 255, 1, 0]], np.uint8)
    p = str(tmp_path / "t.tif")
    tifffile.imwrite(p, a)
    assert Forest_dataset.read_image(None, p, "labels").tolist() == [[0, 1, 1, 255], [1, 255, 1, 0]]


def test_loss_ignores_255():
    y = labels01(seed=5)
    ign = torch.zeros_like(y, dtype=torch.bool)
    ign[:, 4:20, 8:30] = True
    y255 = y.clone()
    y255[ign] = 255
    losses = []
    for seed in range(4):                        # predictions that differ only on the 255 pixels
        x = softmaxed(seed=0).clone()
        other = softmaxed(seed=100 + seed)
        x[ign.unsqueeze(1).expand_as(x)] = other[ign.unsqueeze(1).expand_as(x)]
        x.requires_grad_(True)
        loss = new_loss(x, y255)
        loss.backward()
        assert torch.all(x.grad[ign.unsqueeze(1).expand_as(x)] == 0)          # no gradient from 255 pixels
        losses.append(loss.item())
    assert max(losses) - min(losses) < 1e-6

    # equals the loss computed on the valid pixels only
    x = softmaxed(seed=0)
    N = len(x)
    v = ~ign
    ce = -torch.log(x + 1e-16).gather(1, (y * v).unsqueeze(1)).squeeze(1)[v].mean()
    p1, t1 = x[:, 1] * v, (y == 1) & v
    iou = torch.stack([(p1[i] * t1[i]).sum() / ((p1[i] + t1[i] - p1[i] * t1[i]).sum() + 1e-16) for i in range(N)])
    ref = 0.5 * ce + 0.5 * (1 - iou.mean())
    assert abs(new_loss(x, y255).item() - ref.item()) < 1e-6


@pytest.mark.parametrize("script", TRAIN_SCRIPTS)
def test_val_iou_ignores_255(script):
    labels = labels01(seed=6)
    ign = torch.zeros_like(labels, dtype=torch.bool)
    ign[:, :10, :] = True
    lab255 = labels.clone()
    lab255[ign] = 255
    base = labels01(seed=7)
    counts = set()
    for fill in (0, 1):                          # prediction on the 255 pixels all 0, then all 1
        preds = base.clone()
        preds[ign] = fill
        counts.add(new_val_iou(preds, lab255, script))
    assert len(counts) == 1
    # equals the IoU over the valid pixels only
    v = ~ign
    inter = ((base == 1) & (labels == 1) & v).sum().item()
    union = (((base == 1) | (labels == 1)) & v).sum().item()
    assert counts == {(inter, union)}
