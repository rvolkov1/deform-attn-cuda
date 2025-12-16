import torch
import torch.nn as nn
import time

from dat import DAttentionBaseline

import re

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#torch.use_deterministic_algorithms(True)


def parse_shape_line(line):
    m = re.search(r"\[([^\]]+)\]", line)
    if not m:
        return None

    dims = re.findall(r"\d+", m.group(1))
    if not dims:
        return None

    return tuple(map(int, dims))


def read_tensor_txt(path):
    """
    Python equivalent of read_tensor_txt()

    Returns:
        data (list of float)
        out_count (int)
        B, C, H, W (int)
    """
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError:
        raise RuntimeError(f"Error: cannot open {path}")

    # -----------------------------------------------------
    # PASS 1: Find the shape line
    # -----------------------------------------------------

    shape = None

    for i, line in enumerate(lines):
        if "shape" in line:
            if i + 1 < len(lines):
                shape = parse_shape_line(lines[i + 1])
            break

    # -----------------------------------------------------
    # PASS 2: Count floating-point numbers
    # (skip first 5 lines, same as file_idx < 6 in C)
    # -----------------------------------------------------
    float_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
    count = 0

    for line in lines[5:]:
        matches = float_pattern.findall(line)
        count += len(matches)

    # -----------------------------------------------------
    # PASS 3: Parse floats into buffer
    # -----------------------------------------------------
    data = []
    for line in lines[5:]:
        for m in float_pattern.findall(line):
            data.append(float(m))

    # Safety check (mirrors C behavior implicitly)
    if len(data) != count:
        raise RuntimeError("Internal parsing error")

    tensor = torch.tensor(data).reshape(shape)
    print("dtype:", tensor.dtype)

    assert tensor.numel() == len(data)

    return tensor

device = "cpu"
testcase_no = 1

base_dir = f"testcases/test_{testcase_no}"

x = read_tensor_txt(base_dir + "/x.txt")
y_true = read_tensor_txt(base_dir + "/y.txt")
rpe_table = read_tensor_txt(base_dir + "/rpe_table.txt")

B, C, H, W = x.shape

attn = DAttentionBaseline(
    q_size=(H, W),
    kv_size=(H, W),
    n_heads=4,
    n_head_channels=C // 4,
    n_groups=4,
    stride=1,
    ksize=3,
    rpe_table = rpe_table
).to(device)

print("Warming up...")
for _ in range(20):
    y = attn(x)


def time_forward(model, x, iters=100):
    s_to_ms = 1000

    start = time.time()
    for _ in range(iters):
        y = model(x)

        y_pred = y[0]


    print("torch.close", torch.allclose(y_pred, y_true))
    return ((time.time() - start) / iters) * s_to_ms


ms = time_forward(attn, x)
print(f"Forward Pass Time: {ms:.3f} ms/iter")