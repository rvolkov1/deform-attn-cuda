#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include "cnpy.h"

__global__ void layernorm_nchw_over_c_nobias_f32(
    const float* __restrict__ x,   // [B,Cg,H,W]
    const float* __restrict__ gamma,// [Cg] or nullptr
    float* __restrict__ y,         // [B,Cg,H,W]
    int B, int Cg, int H, int W,
    float eps)
{
    int idx = blockIdx.x;               // idx in [0, B*H*W)
    int HW  = H * W;
    if (idx >= B * HW) return;

    int b = idx / HW;
    int s = idx - b * HW;              // s = h*W + w

    // Base pointer to x/y for this (b, :, h, w)
    // x[c*HW + s] with batch offset b*Cg*HW
    const float* xb = x + (b * Cg * HW) + s;
    float*       yb = y + (b * Cg * HW) + s;

    // 1) Compute mean over channels
    float sum = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        sum += xb[c * HW];
    }

    // Reduce within block
    __shared__ float sh_sum;
    // simple reduction for small Cg: use atomic add into shared (fine for Cg=16)
    if (threadIdx.x == 0) sh_sum = 0.0f;
    __syncthreads();
    atomicAdd(&sh_sum, sum);
    __syncthreads();
    float mean = sh_sum / (float)Cg;

    // 2) Compute variance over channels
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

    // 3) Normalize and (optional) apply gamma, no beta
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = (xb[c * HW] - mean) * inv_std;
        float g = (gamma ? gamma[c] : 1.0f);
        yb[c * HW] = v * g;
    }
}

// // declare your launcher


void launch_layernorm(const float* d_x, const float* d_gamma, float* d_y,
                      int B, int Cg, int H, int W, cudaStream_t stream)
{
    int blocks = B * H * W;
    int threads = 32;      // fine for Cg=16
    float eps = 1e-5f;

    layernorm_nchw_over_c_nobias_f32<<<blocks, threads, 0, stream>>>(
        d_x, d_gamma, d_y, B, Cg, H, W, eps
    );
}

int main() {
  // Load npy
  cnpy::NpyArray xA = cnpy::npy_load("testx.npy");
  cnpy::NpyArray gA = cnpy::npy_load("testgamma.npy");
  cnpy::NpyArray yA = cnpy::npy_load("testy_ref.npy");

  float* x_h = xA.data<float>();
  float* g_h = gA.data<float>();
  float* y_ref_h = yA.data<float>();

  // Extract dims from x (assumes shape [B,Cg,H,W])
  auto xs = xA.shape;
  int B = (int)xs[0], Cg = (int)xs[1], H = (int)xs[2], W = (int)xs[3];
  size_t N = (size_t)B * Cg * H * W;

  // Device alloc
  float *d_x, *d_g, *d_y;
  cudaStream_t stream; cudaStreamCreate(&stream);
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_g, Cg * sizeof(float));

  cudaMemcpyAsync(d_x, x_h, N*sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_g, g_h, Cg*sizeof(float), cudaMemcpyHostToDevice, stream);

  // Run
  launch_layernorm(d_x, d_g, d_y, B, Cg, H, W, stream);

  // Copy back
  std::vector<float> y_out(N);
  cudaMemcpyAsync(y_out.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Compare
  double sum_abs = 0.0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < N; i++) {
    float diff = std::fabs(y_out[i] - y_ref_h[i]);
    sum_abs += diff;
    if (diff > max_abs) max_abs = diff;
  }
  printf("LayerNorm test: max_abs=%.6g mean_abs=%.6g\n",
         max_abs, (float)(sum_abs / (double)N));

  // Cleanup
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_g);
  cudaStreamDestroy(stream);
}
