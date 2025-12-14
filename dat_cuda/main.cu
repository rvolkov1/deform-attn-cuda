#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "read_utils.h"
#include "dat.cuh"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


typedef void (*kernel_ptr)(const float*, int, int, int, int);

float run_kernel_once(const char* label,
                       kernel_ptr kernel,
                       dim3 grid, dim3 block,
                       const float* d_N,
                       int B, int C, int H, int W)
{
    cudaEvent_t start, stop;
    float total_ms = 0.0f;

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(d_N, B, C, H, W);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("%s: %.3f ms\n", label, total_ms);
    return total_ms;
}

float benchmark_kernel(const char* label,
                       kernel_ptr kernel,
                       dim3 grid, dim3 block,
                       const float* d_N,
                       int B, int C, int H, int W,
                       int warmup_iters,
                       int timed_iters)
{
    cudaEvent_t start, stop;
    float total_ms = 0.0f;
    int i;

    for (i = 0; i < warmup_iters; i++)
        kernel<<<grid, block>>>(d_N, B, C, H, W);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (i = 0; i < timed_iters; i++)
        kernel<<<grid, block>>>(d_N, B, C, H, W);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    total_ms /= timed_iters;

    printf("%s: %.3f ms\n", label, total_ms);
    return total_ms;
}

int main(void)
{
    int B_x, C_x, H_x, W_x;
    int B_y, C_y, H_y, W_y;
    size_t size_x, size_y, size_pos, size_ref;

    float *h_X = read_tensor_txt("testcases/test_1/x.txt", 
                                 &size_x,
                                 &B_x, &C_x, &H_x, &W_x);

    float *h_Y_true = read_tensor_txt("testcases/test_1/y.txt", 
                                 &size_y,
                                 &B_y, &C_y, &H_y, &W_y);

    printf("Input tensor: %d x %d x %d x %d  |  B x C x W x H\n\n", B_x, C_x, H_x, W_x);
    printf("Output tensor: %d x %d x %d x %d  |  B x C x W x H\n\n", B_y, C_y, H_y, W_y);

    if (!h_X) {
        printf("Host allocation failed.\n");
        return 1;
    }

    float *d_X;
    CUDA_CHECK(cudaMalloc(&d_X, size_x * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, size_x * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((W_x + block.x - 1) / block.x,
              (H_x + block.y - 1) / block.y);


    run_kernel_once("d_baseline_dat_forward",
                    d_baseline_dat_forward,
                    1, 1,
                    d_X,
                    B_x, C_x, H_x, W_x);

    //benchmark_kernel("d_baseline_dat_forward",
    //                d_baseline_dat_forward,
    //                1, 1,
    //                d_X,
    //                B_x, C_x, H_x, W_x,
    //                3, 20);

    CUDA_CHECK(cudaFree(d_X));
    free(h_X);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
