#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstddef>

// void conv1x1_nobias_forward(cudnnHandle_t cudnn,
//                             const float* d_x,
//                             const float* d_w,
//                             float* d_y,
//                             int B, int C, int H, int W, int K,
//                             cudaStream_t stream,
//                             void* d_workspace = nullptr,
//                             size_t workspace_bytes = 0);