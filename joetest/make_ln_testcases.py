import numpy as np
import torch
import einops

# Fixed seed for repeatability
torch.manual_seed(0)

B, Cg, H, W = 1, 16, 5, 7
eps = 1e-5

x = torch.randn(B, Cg, H, W, dtype=torch.float32)

# "No bias": use gamma, force beta=0
ln = torch.nn.LayerNorm(Cg, eps=eps, elementwise_affine=True)
with torch.no_grad():
    ln.bias.zero_()                 # beta = 0
    ln.weight.copy_(torch.randn(Cg))# gamma random

# Reference = LayerNorm over channels per (b,h,w)
x_nhwc = einops.rearrange(x, 'b c h w -> b h w c')
y_nhwc = ln(x_nhwc)
y_ref = einops.rearrange(y_nhwc, 'b h w c -> b c h w')

np.save("testx.npy", x.cpu().numpy())
np.save("testgamma.npy", ln.weight.detach().cpu().numpy())
np.save("testy_ref.npy", y_ref.detach().cpu().numpy())

print("Wrote x.npy, gamma.npy, y_ref.npy")
print("Shapes:", x.shape, ln.weight.shape, y_ref.shape)
