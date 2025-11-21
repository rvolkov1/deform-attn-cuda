import torch
import torch.nn as nn
import time

# ---------------------------------------------------------
# Import the DAttentionBaseline class
# ---------------------------------------------------------
from DAT_pytorch.models.dat_blocks import DAttentionBaseline

torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
B = 1
C = 64
H = 32
W = 32

device = "cuda"

# This is just an example configuration – match it to your model
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


# ---------------------------------------------------------
# Dummy Input
# ---------------------------------------------------------
x = torch.randn(B, C, H, W, device=device)


# ---------------------------------------------------------
# Warmup
# ---------------------------------------------------------
print("Warming up...")
for _ in range(20):
    y = attn(x)
torch.cuda.synchronize()


# ---------------------------------------------------------
# CUDA event-based timing (accurate)
# ---------------------------------------------------------
def time_forward(model, x, iters=100):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = model(x)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per iteration


ms = time_forward(attn, x)
print(f"Forward Pass Time: {ms:.3f} ms")


# ---------------------------------------------------------
# Optional: PyTorch Profiler
# Generates a Chrome trace you can open in chrome://tracing
# ---------------------------------------------------------
USE_PROFILER = False

if USE_PROFILER:
    print("Running profiler...")
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("DAttentionBaseline_forward"):
            y = attn(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_chrome_trace("dattn_trace.json")
    print("Profiler trace saved to dattn_trace.json")