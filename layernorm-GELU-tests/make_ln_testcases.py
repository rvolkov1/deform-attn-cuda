import numpy as np
import torch
import einops

def main():
    torch.manual_seed(1111)

    # Match CUDA assumptions
    B, Cg, H, W = 1, 16, 32, 32
    eps = 1e-5

    # Input in NCHW
    x = torch.randn(B, Cg, H, W, dtype=torch.float32)

    # LayerNorm over last dim => we do NCHW->NHWC, LN(dim=Cg), NHWC->NCHW
    ln = torch.nn.LayerNorm(Cg, eps=eps, elementwise_affine=True)

    with torch.no_grad():
        ln.bias.zero_()                  # beta = 0, because we aren't using bias
        ln.weight.copy_(torch.randn(Cg)) # gamma

    # Stage 1: LayerNorm reference (NCHW output)
    x_nhwc = einops.rearrange(x, "b c h w -> b h w c")
    y_ln_nhwc = ln(x_nhwc)
    y_ln = einops.rearrange(y_ln_nhwc, "b h w c -> b c h w")

    # Stage 2: GELU(tanh) reference applied to LayerNorm output
    gelu = torch.nn.GELU()
    y_final = gelu(y_ln)

    testpath = "tests/" + "test10"

    # Save
    np.save(testpath + "/testx.npy", x.detach().cpu().numpy())
    np.save(testpath + "/testgamma.npy", ln.weight.detach().cpu().numpy())
    np.save(testpath + "/testy_ln_ref.npy", y_ln.detach().cpu().numpy())
    np.save(testpath + "/testy_ref.npy", y_final.detach().cpu().numpy())

    print("Wrote: testx.npy, testgamma.npy, testy_ln_ref.npy, testy_ref.npy")
    print("Shapes:")
    print("  x           :", tuple(x.shape))
    print("  gamma       :", tuple(ln.weight.shape))
    print("  y_ln_ref    :", tuple(y_ln.shape))
    print("  y_final_ref :", tuple(y_final.shape))

if __name__ == "__main__":
    main()
