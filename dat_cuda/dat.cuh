__device__ 
void d_conv_2d_stride_1(int inputs) {

}

__global__
void d_baseline_dat_forward(const float *N, int B, int C, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    //if (row >= H || col >= W) return;

    printf("col: %d, row: %d\n", col, row);
}