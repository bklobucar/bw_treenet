import sys, torch
def sd(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            if k in ck and isinstance(ck[k], dict): return ck[k], list(ck.keys())
        return ck, None
    return ck.state_dict(), None
a, ta = sd(sys.argv[1]); b, tb = sd(sys.argv[2])
print("swiss top-level:", ta, "| v19 top-level:", tb)
print("n keys swiss/v19:", len(a), len(b))
strip = lambda k: k.replace("module.", "")
A = {strip(k): tuple(v.shape) for k, v in a.items()}
B = {strip(k): tuple(v.shape) for k, v in b.items()}
print("only in swiss:", [k for k in A if k not in B][:20])
print("only in v19:", [k for k in B if k not in A][:20])
print("shape mismatches:", [(k, A[k], B[k]) for k in A if k in B and A[k] != B[k]][:20])
print("LE-like keys in swiss:", [k for k in A if "lum" in k.lower() or "enhanc" in k.lower() or k.startswith("le.")][:10])
print("first 5 swiss keys:", list(A)[:5])
