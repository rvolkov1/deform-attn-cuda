#include <cufft.h>
#include <cublas.h>
#include <cudnn.h>

__device__ void d_norm2(float* buf, int len) {
  float sum = 0;
  for (int i = 0; i < len; ++i) {
    sum += buf[i] * buf[i];
  }

  printf("norm2: %f\n", sqrt(sum));
}

__device__ void d_print_first_n_elements(float* buf, int n) {
  for (int i = 0; i < n; ++i) {
    printf("buf[%d]: %f\n", i, buf[i]);
  }

}

__device__ void d_conv1x1_no_bias(
    const float* x,     // [B, C, H, W]
    const float* w,     // [K, C]
    float* y,           // [B, K, H, W]
    int b, int h, int w_out,
    int C, int H, int W,
    int K
) {
    int HW  = H * W;
    int CHW = C * HW;
    int KHW = K * HW;

    const float* x_ptr = x + b * CHW;
    float* y_ptr = y + b * KHW;

    for (int k = 0; k < K; ++k) {
        float val = 0;
        const float* w_ptr = w + k * C;

        for (int c = 0; c < C; ++c) {
            val += x_ptr[c * HW + h * W + w_out] * w_ptr[c];
        }

        y_ptr[k * HW + h * W + w_out] = val;
    }
}

__device__ void d_conv1x1(
    const float* x,     // [B, C, H, W]
    const float* w,     // [K, C]
    const float* bias,  // [K]
    float* y,           // [B, K, H, W]
    int b, int h, int w_out,
    int C, int H, int W,
    int K
) {
    int HW  = H * W;
    int CHW = C * HW;
    int KHW = K * HW;

    const float* x_ptr = x + b * CHW;
    float* y_ptr = y + b * KHW;

    for (int k = 0; k < K; ++k) {
        float val = bias[k];
        const float* w_ptr = w + k * C;

        for (int c = 0; c < C; ++c) {
            val += x_ptr[c * HW + h * W + w_out] * w_ptr[c];
        }

        y_ptr[k * HW + h * W + w_out] = val;
    }
}

__device__ void d_layernorm_C(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int b, int h, int w,
    int C, int H, int W,
    float eps
) {
    int HW  = H * W;
    int CHW = C * HW;
    int idx = h * W + w;

    // 1) mean
    float mean = 0.0f;
    for (int c = 0; c < C; ++c) {
        mean += x[b * CHW + c * HW + idx];
    }
    mean /= C;

    // 2) variance
    float var = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = x[b * CHW + c * HW + idx] - mean;
        var += v * v;
    }
    var /= C;

    float inv_std = rsqrtf(var + eps);

    // 3) normalize + affine
    for (int c = 0; c < C; ++c) {
        int off = b * CHW + c * HW + idx;
        y[off] = gamma[c] * (x[off] - mean) * inv_std + beta[c];
    }
}

//size_t conv_offset_0_weight_size = 16 * 1 * 3 * 3;
//size_t conv_offset_1_weight_size = 16;
//size_t conv_offset_1_bias_size = 16;
//size_t conv_offset_3_weight_size = 2 * 16 * 1 * 1;
//size_t proj_q_weight_size = 64 * 64 * 1 * 1;
//size_t proj_q_bias_size = 64;
//size_t proj_k_weight_size = 64 * 64 * 1 * 1;
//size_t proj_k_bias_size = 64;
//size_t proj_v_weight_size = 64 * 64 * 1 * 1;
//size_t proj_v_bias_size = 64;
//size_t proj_out_weight_size = 64 * 64 * 1 * 1;
//size_t proj_out_bias_size = 64;

__global__
void baseline_dat_forward(
                            float *x, int B, int C, int H, int W,
                            float *y, int By, int Cy, int Hy, int Wy,
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

    if (row != 0 || col != 0) return;

    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        for (int w_out = 0; w_out < W; ++w_out) {
          d_conv1x1(x, 
                        d_proj_q_weight, d_proj_q_bias, 
                        y, 
                        b, h, w_out,
                        C, H, W, 64);
        }
      }
    }

    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        for (int w_out = 0; w_out < W; ++w_out) {
          d_conv1x1(x, 
                        d_proj_q_weight, d_proj_q_bias, 
                        y, 
                        b, h, w_out,
                        C, H, W, 64);
        }
      }
    }


    d_print_first_n_elements(x, 10);

    printf("col: %d, row: %d\n", col, row);
    printf("hi\n");

    d_norm2(y, By * Cy * Hy * Wy);
}