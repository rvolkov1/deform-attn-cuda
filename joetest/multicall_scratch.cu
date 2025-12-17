#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

#define CUDNN_CHECK(call) do {                                      \
  cudnnStatus_t s__ = (call);                                       \
  if (s__ != CUDNN_STATUS_SUCCESS) {                                \
    fprintf(stderr, "cuDNN error %s:%d: %s\n",                      \
            __FILE__, __LINE__, cudnnGetErrorString(s__));          \
    std::exit(1);                                                   \
  }                                                                 \
} while (0)

#define CUDA_CHECK(call) do {                                       \
  cudaError_t e__ = (call);                                         \
  if (e__ != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
            __FILE__, __LINE__, cudaGetErrorString(e__));           \
    std::exit(1);                                                   \
  }                                                                 \
} while (0)


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


// // Runs y = Conv1x1(x, w), no bias.
// // x: [B,C,H,W], w: [K,C,1,1], y: [B,K,H,W] in NCHW float.
// void conv1x1_nobias_forward(cudnnHandle_t cudnn,
//                             const float* d_x,
//                             const float* d_w,
//                             float* d_y,
//                             int B, int C, int H, int W, int K,
//                             cudaStream_t stream,
//                             void* d_workspace = nullptr,
//                             size_t workspace_bytes = 0)
// {
//   // Ensure cudnn is on the right stream (safe to call repeatedly).
//   CUDNN_CHECK(cudnnSetStream(cudnn, stream));

//   cudnnTensorDescriptor_t xDesc, yDesc;
//   cudnnFilterDescriptor_t wDesc;
//   cudnnConvolutionDescriptor_t convDesc;

//   CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
//   CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
//   CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
//   CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

//   CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//                                         B, C, H, W));
//   CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
//                                         B, K, H, W));

//   CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
//                                         K, C, 1, 1));

//   CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
//                                              /*pad_h=*/0, /*pad_w=*/0,
//                                              /*stride_h=*/1, /*stride_w=*/1,
//                                              /*dilation_h=*/1, /*dilation_w=*/1,
//                                              CUDNN_CROSS_CORRELATION,
//                                              CUDNN_DATA_FLOAT));

//   // Choose an algorithm (simple default; you can switch to v7 "GetConvolutionForwardAlgorithm_v7" later)
//   cudnnConvolutionFwdAlgo_t algo;
//   CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
//       cudnn, xDesc, wDesc, convDesc, yDesc,
//       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//       /*memoryLimitInBytes=*/0,
//       &algo));

//   // Workspace sizing
//   size_t needed_ws = 0;
//   CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
//       cudnn, xDesc, wDesc, convDesc, yDesc, algo, &needed_ws));

//   bool allocated_here = false;
//   if (needed_ws > workspace_bytes) {
//     // If caller didn't provide enough workspace, allocate temporarily (fine for bring-up; avoid in perf path).
//     CUDA_CHECK(cudaMalloc(&d_workspace, needed_ws));
//     workspace_bytes = needed_ws;
//     allocated_here = true;
//   }

//   const float alpha = 1.0f;
//   const float beta  = 0.0f;
//   CUDNN_CHECK(cudnnConvolutionForward(
//       cudnn,
//       &alpha,
//       xDesc, d_x,
//       wDesc, d_w,
//       convDesc, algo,
//       d_workspace, workspace_bytes,
//       &beta,
//       yDesc, d_y));

//   if (allocated_here) {
//     CUDA_CHECK(cudaFree(d_workspace));
//   }

//   CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
//   CUDNN_CHECK(cudnnDestroyFilterDescriptor(wDesc));
//   CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
//   CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
// }
