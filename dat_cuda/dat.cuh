#include <cufft.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include "cnpy.h"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDNN_CHECK(call)                                                  \
do {                                                                       \
    cudnnStatus_t status = call;                                           \
    if (status != CUDNN_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuDNN Error: %s at %s:%d\n",                      \
                cudnnGetErrorString(status), __FILE__, __LINE__);          \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

__global__ void layernorm_nchw_over_c_nobias_f32(
    const float* __restrict__ x,    
    const float* __restrict__ gamma,
    float* __restrict__ y,          
    int B, int Cg, int H, int W,
    float eps)
{
    int idx = blockIdx.x;  
    int HW  = H * W;
    if (idx >= B * HW) return;

    int b = idx / HW;
    int s = idx - b * HW;  

    const float* xb = x + (b * Cg * HW) + s;
    float*       yb = y + (b * Cg * HW) + s;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        sum += xb[c * HW];
    }

    __shared__ float sh_sum;
    if (threadIdx.x == 0) sh_sum = 0.0f;
    __syncthreads();
    atomicAdd(&sh_sum, sum);
    __syncthreads();
    float mean = sh_sum / (float)Cg;

    float sq = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = xb[c * HW] - mean;
        sq += v * v;
    }

    __shared__ float sh_sq;
    if (threadIdx.x == 0) sh_sq = 0.0f;
    __syncthreads();
    atomicAdd(&sh_sq, sq);
    __syncthreads();
    float var = sh_sq / (float)Cg;

    float inv_std = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = (xb[c * HW] - mean) * inv_std;
        float g = (gamma ? gamma[c] : 1.0f);
        yb[c * HW] = v * g;
    }
}

__global__ void gelu_tanh_f32_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = x[i];
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f; // from pytorch impl

    float v3 = v * v * v;
    float t  = kAlpha * (v + kBeta * v3);
    y[i] = 0.5f * v * (1.0f + tanhf(t));
}

void launch_layernorm(const float* d_x, const float* d_gamma, float* d_y,
                      int B, int Cg, int H, int W)
{
    int blocks = B * H * W;
    int threads = 32;     // fine for Cg=16
    float eps = 1e-5f;

    layernorm_nchw_over_c_nobias_f32<<<blocks, threads>>>(
        d_x, d_gamma, d_y, B, Cg, H, W, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

void launch_gelu_tanh_f32(const float* d_x, float* d_y, int n, cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    gelu_tanh_f32_kernel<<<grid, block, 0, stream>>>(d_x, d_y, n);
}

__global__ void norm2(float* buf, int len) {
  float sum = 0;
  for (int i = 0; i < len; ++i) {
    sum += buf[i] * buf[i];
  }

  printf("norm2: %f\n", sqrt(sum));
}

__global__ void print_first_n_elements(float* buf, int n) {
  for (int i = 0; i < n; ++i) {
    printf("buf[%d]: %f\n", i, buf[i]);
  }

}

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

    float mean = 0.0f;
    for (int c = 0; c < C; ++c) {
        mean += x[b * CHW + c * HW + idx];
    }
    mean /= C;

    float var = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = x[b * CHW + c * HW + idx] - mean;
        var += v * v;
    }
    var /= C;

    float inv_std = rsqrtf(var + eps);

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

// grid sample utils

//(sizeof(float) * B * H * W * 2) is the size of the ref pointer
__global__ void get_ref_points_kernel(float *ref, int Hk, int Wk) {
    int b = blockIdx.x;
    int col = threadIdx.x;
    int row = threadIdx.y;

    if (row >= Hk || col >= Wk)
        return;

    int gridbase = b * 2 * Hk * Wk;
    int rowbase  = gridbase + 2 * row * Wk;
    int cellbase = rowbase + 2 * col;

    ref[cellbase] +=
        (row * 2.f + 1.f) / (Hk - 1) - 1.f;

    ref[cellbase + 1] +=
        (col * 2.f + 1.f) / (Wk - 1) - 1.f;
}

__device__ float get_value_device(
    const float* input,
    int b, int c, int y, int x,
    int C, int H, int W)
{
    if (x < 0 || x >= W || y < 0 || y >= H)
        return 0.0f;

    int idx = ((b * C + c) * H + y) * W + x;
    return input[idx];
}

__global__ void grid_sample_kernel(
    const float* input,
    const float* grid, 
    float* output, 
    int C,
    int H, int W,
    int H_out, int W_out)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= gridDim.z || h >= H_out || w >= W_out)
        return;

    int gidx = ((b * H_out + h) * W_out + w) * 2;
    float x_norm = grid[gidx];
    float y_norm = grid[gidx + 1];

    //aligning corners
    float x = (x_norm + 1.f) * 0.5f * (W - 1);
    float y = (y_norm + 1.f) * 0.5f * (H - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = x - x0;
    float wy = y - y0;

    for (int c = 0; c < C; ++c) {
        float v00 = get_value_device(input, b, c, y0, x0, C, H, W);
        float v01 = get_value_device(input, b, c, y0, x1, C, H, W);
        float v10 = get_value_device(input, b, c, y1, x0, C, H, W);
        float v11 = get_value_device(input, b, c, y1, x1, C, H, W);

        float val =
            (1.f - wx) * (1.f - wy) * v00 +
            wx         * (1.f - wy) * v01 +
            (1.f - wx) * wy         * v10 +
            wx         * wy         * v11;

        int out_idx =
            ((b * C + c) * H_out + h) * W_out + w;

        output[out_idx] = val;
    }
}