#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "naive_parallel.cuh"

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr,                                  \
                    "CUDA error %s:%d: %s\n",                \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                \
            exit(1);                                          \
        }                                                     \
    } while (0)

void run_iteration(int H_out, int W_out) {
    /* Problem sizes */
    int B = 4;
    int C = 32;
    int H = 32;
    int W = 32;

    int input_size  = B * C * H * W;
    int grid_size   = B * H_out * W_out * 2;
    int output_size = B * C * H_out * W_out;

    float *d_input = NULL;
    float *d_grid = NULL;
    float *d_output = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_input,
                           input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_grid,
                           grid_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output,
                           output_size * sizeof(float)));

    float* h_input = (float*)malloc(input_size * sizeof(float));
    float* h_grid  = (float*)malloc(grid_size * sizeof(float));

    for (int i = 0; i < input_size; ++i)
        h_input[i] = (float)(i % 127) / 127.0f;

    for (int b = 0; b < B; ++b) {
        for (int y = 0; y < H_out; ++y) {
            for (int x = 0; x < W_out; ++x) {
                int idx = ((b * H_out + y) * W_out + x) * 2;
                h_grid[idx + 0] = (2.0f * x) / (W_out - 1) - 1.0f;
                h_grid[idx + 1] = (2.0f * y) / (H_out - 1) - 1.0f;
            }
        }
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input,
                          input_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grid, h_grid,
                          grid_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block;
    block.x = 16;
    block.y = 16;
    block.z = 1;

    dim3 grid;
    grid.x = (W_out + block.x - 1) / block.x;
    grid.y = (H_out + block.y - 1) / block.y;
    grid.z = B;

    printf("========Grid Sample Naive Parallel. H_out: %d, W_out: % d ========\n", H_out, W_out);

    /* Warmup */
    for (int i = 0; i < 5; ++i)
        grid_sample_kernel<<<grid, block>>>(
            d_input, d_grid, d_output,
            C, H, W, H_out, W_out);

    CHECK_CUDA(cudaDeviceSynchronize());

    /* Timing */
    int iters = 50;
    cudaEvent_t start, stop;
    float ms = 0.0f;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iters; ++i) {
        grid_sample_kernel<<<grid, block>>>(
            d_input, d_grid, d_output,
            C, H, W, H_out, W_out);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    float avg_ms = ms / iters;

    double outputs = (double)B * C * H_out * W_out;

    printf("Average kernel time: %.4f ms\n", avg_ms);

    /* Cleanup */
    free(h_input);
    free(h_grid);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_grid));
    CHECK_CUDA(cudaFree(d_output));
}

int main(void)
{
    run_iteration(256, 256);
    run_iteration(128, 128);
    run_iteration(64, 64);
    run_iteration(256, 64);
    run_iteration(64, 256);

    return 0;

}