"""
42_compare_sharp_weights.py
---------------------------
Compares the SharpFeature/SharpOri parameter tensors of two checkpoints
(default: bwtreenet_v12_best.pt vs bwtreenet_v19_best.pt).

Checks whether v19 carries v12's Sharp kernels (warm-start loaded them) or
different ones (the load filter in 07_train_v18.py dropped them). For each
tensor it prints shape, Pearson r, relative difference ||b - a|| / ||a||
and max |b - a|.

Run on a GPU node (the bwtreenet env is GH200-only); CPU tensors are used.
    python scripts/42_compare_sharp_weights.py [--a A.pt] [--b B.pt]
"""
import os, argparse
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")
KEYS = ("SharpFeature", "SharpOri")


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in sd.items() if any(s in k for s in KEYS)}


def main(args):
    a, b = load(args.a), load(args.b)
    na, nb = os.path.basename(args.a), os.path.basename(args.b)
    print(f"a = {na}  ({len(a)} Sharp tensors)")
    print(f"b = {nb}  ({len(b)} Sharp tensors)\n")
    print(f"{'tensor':<40} {'shape':<18} {'pearson_r':>10} {'rel_diff':>10} {'max_abs':>10}")
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            print(f"{k:<40} only in {'a' if k in a else 'b'}")
            continue
        ta, tb = a[k].double().flatten(), b[k].double().flatten()
        shape = "x".join(map(str, a[k].shape))
        if a[k].shape != b[k].shape:
            print(f"{k:<40} shape mismatch {tuple(a[k].shape)} vs {tuple(b[k].shape)}")
            continue
        if ta.std() == 0 or tb.std() == 0:
            r = float("nan")   # constant tensor: correlation undefined
        else:
            r = torch.corrcoef(torch.stack([ta, tb]))[0, 1].item()
        na_ = ta.norm().item()
        rel = (tb - ta).norm().item() / na_ if na_ > 0 else float("nan")
        mx = (tb - ta).abs().max().item()
        print(f"{k:<40} {shape:<18} {r:>10.4f} {rel:>10.4f} {mx:>10.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(MODELS, "bwtreenet_v12_best.pt"))
    ap.add_argument("--b", default=os.path.join(MODELS, "bwtreenet_v19_best.pt"))
    main(ap.parse_args())
