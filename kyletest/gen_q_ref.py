import numpy as np
import sys

# Usage:
# python gen_q_ref.py x.txt Wq.npy bq.npy q_ref.npy B C H W

if len(sys.argv) != 9:
    print("Usage:")
    print("  python gen_q_ref.py x.txt Wq.npy bq.npy q_ref.npy B C H W")
    sys.exit(1)

x_path   = sys.argv[1]
Wq_path  = sys.argv[2]
bq_path  = sys.argv[3]
out_path = sys.argv[4]
B = int(sys.argv[5])
C = int(sys.argv[6])
H = int(sys.argv[7])
W = int(sys.argv[8])

# ---- Load x from custom .txt format ----
def load_x_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the 'mat' line
    try:
        mat_idx = next(i for i, l in enumerate(lines) if l.strip() == "mat")
    except StopIteration:
        raise RuntimeError(f"'mat' not found in {path}")

    # Everything after 'mat' is numeric (with brackets)
    data_lines = lines[mat_idx + 1 :]

    data_str = " ".join(data_lines)
    data_str = data_str.replace("[", " ").replace("]", " ")

    vals = np.fromstring(data_str, sep=" ", dtype=np.float32)
    return vals

# Load tensors
x_flat = load_x_txt(x_path)
Wq = np.load(Wq_path).astype(np.float32)
bq = np.load(bq_path).astype(np.float32)

# Validate sizes
assert x_flat.size == B * C * H * W, \
    f"x size mismatch: {x_flat.size} vs {B*C*H*W}"
assert Wq.size == C * C, \
    f"Wq size mismatch: {Wq.size} vs {C*C}"
assert bq.size == C, \
    f"bq size mismatch: {bq.size} vs {C}"

# Reshape
x  = x_flat.reshape(B, C, H, W)
Wq = Wq.reshape(C, C)
bq = bq.reshape(C)

# ---- Compute proj_q ----
q = np.zeros((B, C, H, W), dtype=np.float32)

for b in range(B):
    for h in range(H):
        for w in range(W):
            q[b, :, h, w] = Wq @ x[b, :, h, w] + bq

# ---- Save as .npy ----
np.save(out_path, q)

print(f"q_ref saved to {out_path}")
print(f"Shape: {q.shape}")
