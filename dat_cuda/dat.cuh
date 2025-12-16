#include <cufft.h>
#include <cublas.h>

__device__ 
void d_conv_2d_stride_1(int inputs) {

}

__global__
void d_baseline_dat_forward(const float *N, int B, int C, int H, int W,
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