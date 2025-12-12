import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time

from dat import DAttentionBaseline

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)

def save_testcase(x, y, testcase_no=1):
  x_txt = f"""
  Tensor x
  shape
  {list(x.shape)}
  mat
  {str(x.flatten().numpy())}
"""

  y_txt = f"""
  Tensor y
  shape
  {list(y[0].shape)}
  mat
  {str(y[0].flatten().detach().numpy())}
  """

  pos_txt = f"""
  Tensor pos
  shape
  {list(y[1].shape)}
  mat
  {str(y[1].flatten().detach().numpy())}
  """

  ref_txt = f"""
  Tensor ref
  shape
  {list(y[2].shape)}
  mat
  {str(y[2].flatten().detach().numpy())}
  """

  dirname = f"testcases/test_{testcase_no}"

  os.makedirs(dirname, exist_ok=True)

  with open(f"{dirname}/x.txt", "w") as f1:
      f1.write(x_txt)

  with open(f"{dirname}/y.txt", "w") as f2:
      f2.write(y_txt)

  with open(f"{dirname}/pos.txt", "w") as f2:
      f2.write(pos_txt)

  with open(f"{dirname}/ref.txt", "w") as f2:
      f2.write(ref_txt)
  

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
    return y, ((time.time() - start) / iters) * s_to_ms


y, ms = time_forward(attn, x)

save_testcase(x, y, testcase_no=1)

print(f"Forward Pass Time: {ms:.3f} ms/iter")