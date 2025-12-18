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

/*
dim3 block(16, 16);
dim3 grid(
    (W_out + block.x - 1) / block.x,
    (H_out + block.y - 1) / block.y,
    B
);
*/

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

__device__ float get_value_shared(
    const float* shared,   //C * H * W
    int c, int y, int x,
    int H, int W)
{
    if (x < 0 || x >= W || y < 0 || y >= H)
        return 0.0f;

    return shared[c * H * W + y * W + x];
}

__global__ void grid_sample_kernel_shared(
    const float* input,   // [B, C, H, W], H,W ≤ 32
    const float* grid,    // [B, H_out, W_out, 2]
    float* output,        // [B, C, H_out, W_out]
    int C,
    int H, int W,
    int H_out, int W_out)
{
    // Shared memory layout: [C][H][W]
    __shared__ float shared[];

    int b = blockIdx.z;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int elems_per_channel = H * W;
    int total_elems = C * elems_per_channel;

    for (int idx = tid; idx < total_elems; idx += threads) {
        int c = idx / elems_per_channel;
        int hw = idx % elems_per_channel;
        int y = hw / W;
        int x = hw % W;

        shared[idx] = input[((b * C + c) * H + y) * W + x];
    }

    __syncthreads();

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (h >= H_out || w >= W_out)
        return;

    int gidx = ((b * H_out + h) * W_out + w) * 2;
    float x_norm = grid[gidx];
    float y_norm = grid[gidx + 1];

    // align_corners = true
    float x = (x_norm + 1.f) * 0.5f * (W - 1);
    float y = (y_norm + 1.f) * 0.5f * (H - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = x - x0;
    float wy = y - y0;

    for (int c = 0; c < C; ++c) {
        float v00 = get_value_shared(shared, c, y0, x0, H, W);
        float v01 = get_value_shared(shared, c, y0, x1, H, W);
        float v10 = get_value_shared(shared, c, y1, x0, H, W);
        float v11 = get_value_shared(shared, c, y1, x1, H, W);

        float val =
            (1.f - wx) * (1.f - wy) * v00 +
            wx         * (1.f - wy) * v01 +
            (1.f - wx) * wy         * v10 +
            wx         * wy         * v11;

        output[((b * C + c) * H_out + h) * W_out + w] = val;
    }
}


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

    ref[cellbase] =
        (row * 2.f + 1.f) / (Hk - 1) - 1.f;

    ref[cellbase + 1] =
        (col * 2.f + 1.f) / (Wk - 1) - 1.f;
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