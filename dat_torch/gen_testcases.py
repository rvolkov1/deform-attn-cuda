import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time

from dat import DAttentionBaseline

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)

def save_weights_numpy(model, prefix):
    # conv_offset layers
    conv_offset0 = model.conv_offset[0]  # depthwise conv
    conv_offset1 = model.conv_offset[1].norm  # LayerNormProxy
    conv_offset3 = model.conv_offset[3]  # final conv in conv_offset

    # Save depthwise conv weight
    np.save(f"{prefix}/conv_offset0_weight.npy", conv_offset0.weight.detach().cpu().numpy())
    np.save(f"{prefix}/conv_offset0_bias.npy", conv_offset0.bias.detach().cpu().numpy())
    print(f"{prefix}/conv_offset0_weight.npy", conv_offset0.weight.detach().cpu().numpy().shape)
    print(f"{prefix}/conv_offset0_bias.npy", conv_offset0.bias.detach().cpu().numpy().shape)

    # Save LayerNorm weight and bias
    np.save(f"{prefix}/conv_offset1_weight.npy", conv_offset1.weight.detach().cpu().numpy())
    np.save(f"{prefix}/conv_offset1_bias.npy", conv_offset1.bias.detach().cpu().numpy())

    print(f"{prefix}/conv_offset1_weight.npy", conv_offset1.weight.detach().cpu().numpy().shape)
    print(f"{prefix}/conv_offset1_bias.npy", conv_offset1.bias.detach().cpu().numpy().shape)

    # Save conv_offset final conv weight
    np.save(f"{prefix}/conv_offset3_weight.npy", conv_offset3.weight.detach().cpu().numpy())

    print(f"{prefix}/conv_offset3_weight.npy", conv_offset3.weight.detach().cpu().numpy().shape)

    # Save proj_q, proj_k, proj_v, proj_out weights and biases
    for name, layer in [("proj_q", model.proj_q),
                        ("proj_k", model.proj_k),
                        ("proj_v", model.proj_v),
                        ("proj_out", model.proj_out)]:
        np.save(f"{prefix}/{name}_weight.npy", layer.weight.detach().cpu().numpy().flatten(order="C"))
        np.save(f"{prefix}/{name}_bias.npy", layer.bias.detach().cpu().numpy().flatten(order="C"))

        print(f"{prefix}/{name}_weight.npy", layer.weight.detach().cpu().numpy().shape)
        print(f"{prefix}/{name}_bias.npy", layer.bias.detach().cpu().numpy().shape)

    # Save rpe_table
    if model.rpe_table is not None:
        np.save(f"{prefix}/rpe_table.npy", model.rpe_table.detach().cpu().numpy())
        print(f"{prefix}/rpe_table.npy", model.rpe_table.detach().cpu().numpy().shape)

    print("All weights saved as .npy files.")


def save_testcase(x, y, attn, testcase_no=1):
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
  Tensor pos
  shape
  {list(y[2].shape)}
  mat
  {str(y[2].flatten().detach().numpy())}
  """

  rpe_table_txt = f"""
  Tensor ref
  shape
  {list(y[3].shape)}
  mat
  {str(y[3].flatten().detach().numpy())}
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

  with open(f"{dirname}/rpe_table.txt", "w") as f2:
      f2.write(rpe_table_txt)
  

for test_no in range(11, 21):
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
      stride=1,
      ksize=3,
      rpe_table=None
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

  save_testcase(x, y, attn, testcase_no=test_no)
  save_weights_numpy(attn, f"testcases/test_{test_no}")

  print(f"Forward Pass Time: {ms:.3f} ms/iter")