#include <cuda_runtime.h>

__device__ __forceinline__ float get_value_shmem(
    const float* shmem,
    int c, int y, int x,
    int H, int W)
{
    if (x < 0 || x >= W || y < 0 || y >= H)
        return 0.0f;

    return shmem[(c * H + y) * W + x];
}


__global__ void grid_sample_kernel_shared(
    const float* input,   // [B, C, H, W]
    const float* grid,    // [B, H_out, W_out, 2]
    float* output,        // [B, C, H_out, W_out]
    int C,
    int H, int W,
    int H_out, int W_out)
{
    extern __shared__ float shmem[];  
    /* layout: shmem[c][y][x] */

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int w = blockIdx.x * blockDim.x + tx;
    int h = blockIdx.y * blockDim.y + ty;
    int b = blockIdx.z;

    /* ------------------------------------------------ */
    /* Load input tile into shared memory               */
    /* ------------------------------------------------ */

    int t = ty * blockDim.x + tx;
    int threads = blockDim.x * blockDim.y;
    int elems = C * H * W;

    for (int idx = t; idx < elems; idx += threads) {
        int x = idx % W;
        int y = (idx / W) % H;
        int c = idx / (H * W);

        int in_idx = ((b * C + c) * H + y) * W + x;
        shmem[idx] = input[in_idx];
    }

    __syncthreads();

    /* ------------------------------------------------ */
    /* Bounds check for output                          */
    /* ------------------------------------------------ */

    if (h >= H_out || w >= W_out)
        return;

    /* ------------------------------------------------ */
    /* Grid lookup                                      */
    /* ------------------------------------------------ */

    int gidx = ((b * H_out + h) * W_out + w) * 2;
    float x_norm = grid[gidx];
    float y_norm = grid[gidx + 1];

    /* align_corners = true */
    float x = (x_norm + 1.0f) * 0.5f * (W - 1);
    float y = (y_norm + 1.0f) * 0.5f * (H - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = x - x0;
    float wy = y - y0;

    /* ------------------------------------------------ */
    /* Bilinear interpolation                           */
    /* ------------------------------------------------ */

    for (int c = 0; c < C; ++c) {
        float v00 = get_value_shmem(shmem, c, y0, x0, H, W);
        float v01 = get_value_shmem(shmem, c, y0, x1, H, W);
        float v10 = get_value_shmem(shmem, c, y1, x0, H, W);
        float v11 = get_value_shmem(shmem, c, y1, x1, H, W);

        float val =
            (1.0f - wx) * (1.0f - wy) * v00 +
            wx          * (1.0f - wy) * v01 +
            (1.0f - wx) * wy          * v10 +
            wx          * wy          * v11;

        int out_idx =
            ((b * C + c) * H_out + h) * W_out + w;

        output[out_idx] = val;
    }
}
