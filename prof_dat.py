import torch
import torch.nn as nn
import time

from dat import DAttentionBaseline

B = 1
C = 64
H = 32
W = 32

device = "cpu"

attn = DAttentionBaseline(
    q_size=(H, W),
    kv_size=(H, W),
    n_heads=4,
    n_head_channels=C // 4,
    n_groups=4,
    attn_drop=0.0,
    proj_drop=0.0,
    stride=1,
    offset_range_factor=1.0,
    use_pe=True,
    dwc_pe=False,
    no_off=False,
    fixed_pe=False,
    ksize=3,
    log_cpb=False,
).to(device)


x = torch.randn(B, C, H, W, device=device)


print("Warming up...")
for _ in range(20):
    y = attn(x)


def time_forward(model, x, iters=100):
    s_to_ms = 1000

    start = time.time()
    for _ in range(iters):
        y = model(x)
    return ((time.time() - start) / iters) * s_to_ms


ms = time_forward(attn, x)
print(f"Forward Pass Time: {ms:.3f} ms/iter")