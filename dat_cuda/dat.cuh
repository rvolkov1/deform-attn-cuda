#include <cufft.h>
#include <cublas.h>
#include <cudnn.h>

// 1x1 convolution + bias for batch=1, NCHW, device-callable
// x: [C, H, W], w: [K, C], bias: [K], y: [K, H, W]
__device__ void d_conv1x1_bias(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int C, int H, int W,
    int K
) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= H || w_out >= W) return;

    int HW = H * W;
    for (int k = 0; k < K; ++k) {
        float val = bias[k];
        for (int c = 0; c < C; ++c) {
            val += x[c*HW + h*W + w_out] * w[k*C + c];
        }
        y[k*HW + h*W + w_out] = val;
    }
}

__global__
void baseline_dat_forward(const float *x, int B, int C, int H, int W,
                            int q_size_h, int q_size_w,
                            int kv_size_h, int kv_size_w,
                            int n_heads,
                            int n_head_channels,
                            int n_groups,
                            int stride,
                            int ksize,
                            float* d_conv_offset_0_weight,
                            float* d_conv_offset_1_weight,
                            float* d_conv_offset_1_bias,
                            float* d_conv_offset_3_weight,
                            float* d_proj_q_weight,
                            float* d_proj_q_bias,
                            float* d_proj_k_weight,
                            float* d_proj_k_bias,
                            float* d_proj_v_weight,
                            float* d_proj_v_bias,
                            float* d_proj_out_weight,
                            float* d_proj_out_bias
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //if (row >= H || col >= W) return;

    printf("col: %d, row: %d\n", col, row);
    printf("hi\n");
}