#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "read_utils.h"
#include "dat.cuh"
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

    // CONV 0
    cnpy::NpyArray conv_offset_0_weight_obj = cnpy::npy_load("testcases/test_1/conv_offset0_weight.npy");
    size_t conv_offset_0_weight_size = 16 * 1 * 3 * 3;
    float* conv_offset_0_weight = conv_offset_0_weight_obj.data<float>();

    // CONV 1
    cnpy::NpyArray conv_offset_1_weight_obj = cnpy::npy_load("testcases/test_1/conv_offset1_weight.npy");
    size_t conv_offset_1_weight_size = 16;
    float* conv_offset_1_weight = conv_offset_1_weight_obj.data<float>();

    cnpy::NpyArray conv_offset_1_bias_obj = cnpy::npy_load("testcases/test_1/conv_offset1_bias.npy");
    size_t conv_offset_1_bias_size = 16;
    float* conv_offset_1_bias = conv_offset_1_bias_obj.data<float>();
  

    // CONV 3
    cnpy::NpyArray conv_offset_3_weight_obj = cnpy::npy_load("testcases/test_1/conv_offset3_weight.npy");
    size_t conv_offset_3_weight_size = 2 * 16 * 1 * 1;
    float* conv_offset_3_weight = conv_offset_3_weight_obj.data<float>();

    // PROJ Q
    cnpy::NpyArray proj_q_weight_obj = cnpy::npy_load("testcases/test_1/proj_q_weight.npy");
    size_t proj_q_weight_size = 64 * 64 * 1 * 1;
    float* proj_q_weight = proj_q_weight_obj.data<float>();

    cnpy::NpyArray proj_q_bias_obj = cnpy::npy_load("testcases/test_1/proj_q_bias.npy");
    size_t proj_q_bias_size = 64;
    float* proj_q_bias = proj_q_bias_obj.data<float>();

    // PROJ K
    cnpy::NpyArray proj_k_weight_obj = cnpy::npy_load("testcases/test_1/proj_k_weight.npy");
    size_t proj_k_weight_size = 64 * 64 * 1 * 1;
    float* proj_k_weight = proj_k_weight_obj.data<float>();

    cnpy::NpyArray proj_k_bias_obj = cnpy::npy_load("testcases/test_1/proj_k_bias.npy");
    size_t proj_k_bias_size = 64;
    float* proj_k_bias = proj_k_bias_obj.data<float>();

    //PROJ V
    cnpy::NpyArray proj_v_weight_obj = cnpy::npy_load("testcases/test_1/proj_v_weight.npy");
    size_t proj_v_weight_size = 64 * 64 * 1 * 1;
    float* proj_v_weight = proj_v_weight_obj.data<float>();

    cnpy::NpyArray proj_v_bias_obj = cnpy::npy_load("testcases/test_1/proj_v_bias.npy");
    size_t proj_v_bias_size = 64;
    float* proj_v_bias = proj_v_bias_obj.data<float>();

    // PROJ OUT
    cnpy::NpyArray proj_out_weight_obj = cnpy::npy_load("testcases/test_1/proj_out_weight.npy");
    size_t proj_out_weight_size = 64 * 64 * 1 * 1;
    float* proj_out_weight = proj_out_weight_obj.data<float>();

    cnpy::NpyArray proj_out_bias_obj = cnpy::npy_load("testcases/test_1/proj_out_bias.npy");
    size_t proj_out_bias_size = 64;
    float* proj_out_bias = proj_out_bias_obj.data<float>();

    printf("Input tensor: %d x %d x %d x %d  |  B x C x W x H\n\n", B_x, C_x, H_x, W_x);
    printf("Output tensor: %d x %d x %d x %d  |  B x C x W x H\n\n", B_y, C_y, H_y, W_y);

    if (!h_X) {
        printf("Host allocation failed.\n");
        return 1;
    }


    // X
    float *d_X;
    CUDA_CHECK(cudaMalloc(&d_X, size_x * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, size_x * sizeof(float), cudaMemcpyHostToDevice));

    // CONV 0
    float *d_conv_offset_0_weight;
    CUDA_CHECK(cudaMalloc(&d_conv_offset_0_weight, conv_offset_0_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv_offset_0_weight, conv_offset_0_weight, conv_offset_0_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // CONV 1
    float *d_conv_offset_1_weight;
    CUDA_CHECK(cudaMalloc(&d_conv_offset_1_weight, conv_offset_1_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv_offset_1_weight, conv_offset_1_weight, conv_offset_1_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    float *d_conv_offset_1_bias;
    CUDA_CHECK(cudaMalloc(&d_conv_offset_1_bias, conv_offset_1_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv_offset_1_bias, conv_offset_1_bias, conv_offset_1_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    // CONV 3
    float *d_conv_offset_3_weight;
    CUDA_CHECK(cudaMalloc(&d_conv_offset_3_weight, conv_offset_3_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv_offset_3_weight, conv_offset_3_weight, conv_offset_3_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    // PROJ Q
    float *d_proj_q_weight;
    CUDA_CHECK(cudaMalloc(&d_proj_q_weight, proj_q_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_q_weight, proj_q_weight, proj_q_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    float *d_proj_q_bias;
    CUDA_CHECK(cudaMalloc(&d_proj_q_bias, proj_q_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_q_bias, proj_q_bias, proj_q_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    // PROJ K
    float *d_proj_k_weight;
    CUDA_CHECK(cudaMalloc(&d_proj_k_weight, proj_k_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_k_weight, proj_k_weight, proj_k_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    float *d_proj_k_bias;
    CUDA_CHECK(cudaMalloc(&d_proj_k_bias, proj_k_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_k_bias, proj_k_bias, proj_k_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    // PROJ V
    float *d_proj_v_weight;
    CUDA_CHECK(cudaMalloc(&d_proj_v_weight, proj_v_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_v_weight, proj_v_weight, proj_v_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    float *d_proj_v_bias;
    CUDA_CHECK(cudaMalloc(&d_proj_v_bias, proj_v_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_v_bias, proj_v_bias, proj_v_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    // PROJ OUT
    float *d_proj_out_weight;
    CUDA_CHECK(cudaMalloc(&d_proj_out_weight, proj_out_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_out_weight, proj_out_weight, proj_out_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    float *d_proj_out_bias;
    CUDA_CHECK(cudaMalloc(&d_proj_out_bias, proj_out_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_out_bias, proj_out_bias, proj_out_bias_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((W_x + block.x - 1) / block.x,
              (H_x + block.y - 1) / block.y);

    int q_size_h = H_x;
    int q_size_w = W_x;
    int kv_size_h = H_x;
    int kv_size_w = W_x;
    int n_heads = 4;
    int n_head_channels = C_x / 4;
    int n_groups = 4;
    int stride = 1;
    int ksize=3;

    baseline_dat_forward<<<1, 1>>>(d_X, B_x, C_x, H_x, W_x,
                                    q_size_h, q_size_w,
                                    kv_size_h, kv_size_w,
                                    n_heads, n_head_channels,
                                    n_groups,
                                    stride,
                                    ksize,
                                    d_conv_offset_0_weight,
                                    d_conv_offset_1_weight,
                                    d_conv_offset_1_bias,
                                    d_conv_offset_3_weight,
                                    d_proj_q_weight,
                                    d_proj_q_bias,
                                    d_proj_k_weight,
                                    d_proj_k_bias,
                                    d_proj_v_weight,
                                    d_proj_v_bias,
                                    d_proj_out_weight,
                                    d_proj_out_bias);

    //run_kernel_once("d_baseline_dat_forward",
    //                d_baseline_dat_forward,
    //                1, 1,
    //                d_X,
    //                B_x, C_x, H_x, W_x);

    //benchmark_kernel("d_baseline_dat_forward",
    //                d_baseline_dat_forward,
    //                1, 1,
    //                d_X,
    //                B_x, C_x, H_x, W_x,
    //                3, 20);

    CUDA_CHECK(cudaFree(d_X));
    free(h_X);

    CUDA_CHECK(cudaFree(d_conv_offset_0_weight));
    CUDA_CHECK(cudaFree(d_conv_offset_1_weight));
    CUDA_CHECK(cudaFree(d_conv_offset_1_bias));
    CUDA_CHECK(cudaFree(d_conv_offset_3_weight));
    CUDA_CHECK(cudaFree(d_proj_q_weight));
    CUDA_CHECK(cudaFree(d_proj_q_bias));
    CUDA_CHECK(cudaFree(d_proj_k_weight));
    CUDA_CHECK(cudaFree(d_proj_k_bias));
    CUDA_CHECK(cudaFree(d_proj_v_weight));
    CUDA_CHECK(cudaFree(d_proj_v_bias));
    CUDA_CHECK(cudaFree(d_proj_out_weight));
    CUDA_CHECK(cudaFree(d_proj_out_bias));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
