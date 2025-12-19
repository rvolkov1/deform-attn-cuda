#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "read_utils.h"
#include "dat.cuh"
#include "cnpy.h"
#include <cudnn.h>
#include <cudnn_frontend.h>

typedef void (*kernel_ptr)(const float*, int, int, int, int);

#define BUILD_PATH(buf, base, file) \
    snprintf(buf, sizeof(buf), "%s/%s", base, file)

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

void conv2d_wbias_prev(
            float* d_X, int B_x, int C_x, int H_x, int W_x,
            float* d_Y, int B_y, int C_y, int H_y, int W_y,
            int B_weight, int C_weight, int H_weight, int W_weight,
            int B_bias, int C_bias, int H_bias, int W_bias,
            int kernel_size, int stride, int pad, int dilation, int groups,
            float* d_proj_q_weight,
            float* d_proj_q_bias,
            cudnnHandle_t handle)
{

  //q flat: tensor([-0.7177,  0.7654, -1.1706,  0.2137, -0.4887,  1.1410, -0.3336,  0.5294,
         //1.3686,  0.7021], grad_fn=<SliceBackward0>)

    int size_x = B_x * C_x * H_x * W_x;
    int size_y = B_y * C_y * H_y * W_y;

    cudnnTensorDescriptor_t x_desc, y_desc, bias_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,   // or FLOAT
        B_x, C_x, H_x, W_x
    ));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        B_y, C_y, H_y, W_y
    ));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        B_weight, C_weight, H_weight, W_weight
    ));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad, pad,     // pad_h, pad_w
        stride, stride,     // stride_h, stride_w
        dilation, dilation,     // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    cudnnSetConvolutionGroupCount(conv_desc, groups);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        B_bias, C_bias, H_bias, W_bias
    ));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        algo,
        &workspace_bytes
    ));

    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnActivationDescriptor_t activation_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc,
                                            CUDNN_ACTIVATION_IDENTITY,
                                            CUDNN_PROPAGATE_NAN,
                                            0.0));

    const cudnnTensorDescriptor_t z_desc = y_desc;
    const void* z = nullptr;

    //print_first_n_elements<<<1, 1>>>(d_X, 10);

    CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc, d_X,
        w_desc, d_proj_q_weight,
        conv_desc,
        algo,  // your IMPLICIT_GEMM
        workspace, workspace_bytes,
        &beta,
        y_desc, d_Y
    ));

    float alpha_add = 1.0f, beta_add = 1.0f;  // accumulate
    CUDNN_CHECK(cudnnAddTensor(
        handle,
        &alpha_add,
        bias_desc, d_proj_q_bias,
        &beta_add,
        y_desc, d_Y
    ));

    //printf("Weight first 10:\n");
    //print_first_n_elements<<<1,1>>>(d_proj_q_weight, 10);
    //fflush(stdout);

    //cudaDeviceSynchronize();

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc));

    //printf("bias first 10:\n");
    //print_first_n_elements<<<1,1>>>(d_proj_q_bias, 10);
    //fflush(stdout);

    //cudaDeviceSynchronize();

    //printf("y first 10:\n");
    //print_first_n_elements<<<1, 1>>>(d_Y, 10);
    //norm2<<<1, 1>>>(d_Y, size_y);
    //fflush(stdout);
}


void conv2d_wbias(
            float* d_X, int B_x, int C_x, int H_x, int W_x,
            int x_stride_b, int x_stride_c, int x_stride_h, int x_stride_w,
            float* d_Y, int B_y, int C_y, int H_y, int W_y,
            int B_weight, int C_weight, int H_weight, int W_weight,
            int B_bias, int C_bias, int H_bias, int W_bias,
            int kernel_size, int stride, int pad, int dilation, int groups,
            float* d_proj_q_weight,
            float* d_proj_q_bias,
            cudnnHandle_t handle)
{

    int size_x = B_x * C_x * H_x * W_x;
    int size_y = B_y * C_y * H_y * W_y;

    cudnnTensorDescriptor_t x_desc, y_desc, bias_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        x_desc,
        CUDNN_DATA_FLOAT,   // or FLOAT
        B_x, C_x, H_x, W_x,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w
    ));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        B_y, C_y, H_y, W_y
    ));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        B_weight, C_weight, H_weight, W_weight
    ));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad, pad,     // pad_h, pad_w
        stride, stride,     // stride_h, stride_w
        dilation, dilation,     // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    cudnnSetConvolutionGroupCount(conv_desc, groups);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        B_bias, C_bias, H_bias, W_bias
    ));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        algo,
        &workspace_bytes
    ));

    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnActivationDescriptor_t activation_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc,
                                            CUDNN_ACTIVATION_IDENTITY,
                                            CUDNN_PROPAGATE_NAN,
                                            0.0)); 

    const cudnnTensorDescriptor_t z_desc = y_desc; 
    const void* z = nullptr;

    CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc, d_X,
        w_desc, d_proj_q_weight,
        conv_desc,
        algo, 
        workspace, workspace_bytes,
        &beta,
        y_desc, d_Y
    ));

    float alpha_add = 1.0f, beta_add = 1.0f;  // accumulate
    CUDNN_CHECK(cudnnAddTensor(
        handle,
        &alpha_add,
        bias_desc, d_proj_q_bias,
        &beta_add,
        y_desc, d_Y
    ));

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc));

    //printf("first_10:\n");
    //print_first_n_elements<<<1, 1>>>(d_Y, 10);
    //norm2<<<1, 1>>>(d_Y, size_y);
    //fflush(stdout);
}

void conv2d_nobias(
            float* d_X, int B_x, int C_x, int H_x, int W_x,
            int x_stride_b, int x_stride_c, int x_stride_h, int x_stride_w,
            float* d_Y, int B_y, int C_y, int H_y, int W_y,
            int B_weight, int C_weight, int H_weight, int W_weight,
            int kernel_size, int stride, int pad, int dilation, int groups,
            float* d_proj_q_weight,
            cudnnHandle_t handle)
{

    int size_x = B_x * C_x * H_x * W_x;
    int size_y = B_y * C_y * H_y * W_y;

    cudnnTensorDescriptor_t x_desc, y_desc, bias_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        x_desc,
        CUDNN_DATA_FLOAT,   // or FLOAT
        B_x, C_x, H_x, W_x,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w
    ));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        B_y, C_y, H_y, W_y
    ));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        B_weight, C_weight, H_weight, W_weight
    ));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad, pad,     // pad_h, pad_w
        stride, stride,     // stride_h, stride_w
        dilation, dilation,     // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    cudnnSetConvolutionGroupCount(conv_desc, groups);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        algo,
        &workspace_bytes
    ));

    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc, d_X,
        w_desc, d_proj_q_weight,
        conv_desc,
        algo,  // your IMPLICIT_GEMM
        workspace, workspace_bytes,
        &beta,
        y_desc, d_Y
    ));

    //printf("first_10:\n");
    //print_first_n_elements<<<1, 1>>>(d_Y, 10);
    //norm2<<<1, 1>>>(d_Y, size_y);
    //fflush(stdout);
}

__global__ void scale_kernel(__half* S, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        S[idx] = __hmul(S[idx], __float2half(scale));
    }
}

void sdotprodattn_forward(
    cublasHandle_t cublas,
    cudnnHandle_t handle,
    const float* Q, 
    const float* K, 
    const float* V, 
    float* O,       
    int B, int H, int M, int N, int D,
    float scale
) {
    size_t S_bytes = B * H * M * N * sizeof(float);
    float* S;
    cudaMalloc(&S, S_bytes);

    int batch_count = B*H;
    float alpha = scale;
    float beta = 0.0f;

    cublasSgemmStridedBatched(
        cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        M, N, D,
        &alpha,
        Q, M, M*D,
        K, N, N*D,
        &beta,
        S, M, M*N,
        batch_count
    );

    cudnnTensorDescriptor_t S_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&S_desc));
    int dims[4] = {batch_count*M, N, 1, 1};
    int strides[4] = {N, 1, 1, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(S_desc, CUDNN_DATA_FLOAT, 4, dims, strides));

    float softmax_alpha = 1.0f;
    float softmax_beta  = 0.0f;

    CUDNN_CHECK(cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &softmax_alpha,
        S_desc, S,
        &softmax_beta,
        S_desc, S
    ));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(S_desc));

    // GEMM: O = S * V
    alpha = 1.0f;
    beta  = 0.0f;
    cublasSgemmStridedBatched(
        cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, D, N,
        &alpha,
        S, M, M*N,
        V, D, N*D,
        &beta,
        O, M, M*D,
        batch_count
    );

    cudaFree(S);
}

int main(int argc, char *argv[])
{
    //printf("cuDNN version: %lu\n", cudnnGetVersion());

    char *testcase = argv[1];
    char path[1024];

    int B_x, C_x, H_x, W_x;
    int B_y, C_y, H_y, W_y;
    size_t size_x, size_y, size_pos, size_ref;

    BUILD_PATH(path, testcase, "/x.txt");
    float *h_X = read_tensor_txt(path, 
                                 &size_x,
                                 &B_x, &C_x, &H_x, &W_x);

    BUILD_PATH(path, testcase, "/y.txt");
    float *h_Y_true = read_tensor_txt(path, 
                                 &size_y,
                                 &B_y, &C_y, &H_y, &W_y);

    float *h_Y = (float*)calloc(B_y * C_y * H_y * W_y * 32 * 32, sizeof(float));

    // CONV 0
    BUILD_PATH(path, testcase, "conv_offset0_weight.npy");
    cnpy::NpyArray conv_offset_0_weight_obj = cnpy::npy_load(path);
    size_t conv_offset_0_weight_size = 16 * 1 * 3 * 3 * sizeof(float);
    float* conv_offset_0_weight = conv_offset_0_weight_obj.data<float>();

    BUILD_PATH(path, testcase, "conv_offset0_bias.npy");
    cnpy::NpyArray conv_offset_0_bias_obj = cnpy::npy_load(path);
    size_t conv_offset_0_bias_size = 16 * sizeof(float);
    float* conv_offset_0_bias = conv_offset_0_bias_obj.data<float>();

    // CONV 1
    BUILD_PATH(path, testcase, "conv_offset1_weight.npy");
    cnpy::NpyArray conv_offset_1_weight_obj = cnpy::npy_load(path);
    size_t conv_offset_1_weight_size = 16;
    float* conv_offset_1_weight = conv_offset_1_weight_obj.data<float>();

    BUILD_PATH(path, testcase, "conv_offset1_bias.npy");
    cnpy::NpyArray conv_offset_1_bias_obj = cnpy::npy_load(path);
    size_t conv_offset_1_bias_size = 16;
    float* conv_offset_1_bias = conv_offset_1_bias_obj.data<float>();
  

    // CONV 3
    BUILD_PATH(path, testcase, "conv_offset3_weight.npy");
    cnpy::NpyArray conv_offset_3_weight_obj = cnpy::npy_load(path);
    size_t conv_offset_3_weight_size = 2 * 16 * 1 * 1 * sizeof(float);
    float* conv_offset_3_weight = conv_offset_3_weight_obj.data<float>();

    // PROJ Q
    BUILD_PATH(path, testcase, "proj_q_weight.npy");
    cnpy::NpyArray proj_q_weight_obj = cnpy::npy_load(path);
    size_t proj_q_weight_size = 64 * 64 * 1 * 1 * sizeof(float);
    float* proj_q_weight = proj_q_weight_obj.data<float>();

    BUILD_PATH(path, testcase, "proj_q_bias.npy");
    cnpy::NpyArray proj_q_bias_obj = cnpy::npy_load(path);
    size_t proj_q_bias_size = 64;
    float* proj_q_bias = proj_q_bias_obj.data<float>();

    // PROJ K
    BUILD_PATH(path, testcase, "proj_k_weight.npy");
    cnpy::NpyArray proj_k_weight_obj = cnpy::npy_load(path);
    size_t proj_k_weight_size = 64 * 64 * 1 * 1;
    float* proj_k_weight = proj_k_weight_obj.data<float>();

    BUILD_PATH(path, testcase, "proj_k_bias.npy");
    cnpy::NpyArray proj_k_bias_obj = cnpy::npy_load(path);
    size_t proj_k_bias_size = 64;
    float* proj_k_bias = proj_k_bias_obj.data<float>();

    //PROJ V
    BUILD_PATH(path, testcase, "proj_v_weight.npy");
    cnpy::NpyArray proj_v_weight_obj = cnpy::npy_load(path);
    size_t proj_v_weight_size = 64 * 64 * 1 * 1;
    float* proj_v_weight = proj_v_weight_obj.data<float>();

    BUILD_PATH(path, testcase, "proj_v_bias.npy");
    cnpy::NpyArray proj_v_bias_obj = cnpy::npy_load(path);
    size_t proj_v_bias_size = 64;
    float* proj_v_bias = proj_v_bias_obj.data<float>();

    // PROJ OUT
    BUILD_PATH(path, testcase, "proj_out_weight.npy");
    cnpy::NpyArray proj_out_weight_obj = cnpy::npy_load(path);
    size_t proj_out_weight_size = 64 * 64 * 1 * 1;
    float* proj_out_weight = proj_out_weight_obj.data<float>();

    BUILD_PATH(path, testcase, "proj_out_bias.npy");
    cnpy::NpyArray proj_out_bias_obj = cnpy::npy_load(path);
    size_t proj_out_bias_size = 64;
    float* proj_out_bias = proj_out_bias_obj.data<float>();

    //printf("Input tensor: %d x %d x %d x %d  |  B x C x W x H\n\n", B_x, C_x, H_x, W_x);
    //printf("Output tensor: %d x %d x %d x %d  |  B x C x W x H\n\n", B_y, C_y, H_y, W_y);

    // X
    float *d_X;
    CUDA_CHECK(cudaMalloc(&d_X, size_x * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, size_x * sizeof(float), cudaMemcpyHostToDevice));

    // Y
    float *d_Y;
    int maxsize = 64 * 32 * 32 * 2;
    CUDA_CHECK(cudaMalloc(&d_Y, maxsize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, size_y * sizeof(float), cudaMemcpyHostToDevice));

    // CONV 0
    float *d_conv_offset_0_weight;
    CUDA_CHECK(cudaMalloc(&d_conv_offset_0_weight, conv_offset_0_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv_offset_0_weight, conv_offset_0_weight, conv_offset_0_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    float *d_conv_offset_0_bias;
    CUDA_CHECK(cudaMalloc(&d_conv_offset_0_bias, conv_offset_0_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv_offset_0_bias, conv_offset_0_bias, conv_offset_0_bias_size * sizeof(float), cudaMemcpyHostToDevice));

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
    int n_group_channels = (n_head_channels * n_heads) / n_groups;
    int stride = 1;
    int ksize=3;
    int pad_size = 0;

    if (ksize != stride) {
      pad_size = ksize / 2;
    } else {
      pad_size = 0;
    }

    // setup cudnn
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    //setup cublas
    cublasHandle_t cublas;
    cublasCreate(&cublas);

//    before conv offset: torch.Size([1, 64, 32, 32]) torch.Size([4, 16, 32, 32])
//    after conv offset: torch.Size([4, 2, 32, 32])
//    offset size: torch.Size([4, 2, 32, 32])
//    testcases/test_1/conv_offset0_weight.npy (16, 1, 3, 3)
//    testcases/test_1/conv_offset0_bias.npy (16,)
//    testcases/test_1/conv_offset1_weight.npy (16,)
//    testcases/test_1/conv_offset1_bias.npy (16,)
//    testcases/test_1/conv_offset3_weight.npy (2, 16, 1, 1)
//    testcases/test_1/proj_q_weight.npy (64, 64, 1, 1)
//    testcases/test_1/proj_q_bias.npy (64,)
//    testcases/test_1/proj_k_weight.npy (64, 64, 1, 1)
//    testcases/test_1/proj_k_bias.npy (64,)
//    testcases/test_1/proj_v_weight.npy (64, 64, 1, 1)
//    testcases/test_1/proj_v_bias.npy (64,)
//    testcases/test_1/proj_out_weight.npy (64, 64, 1, 1)
//    testcases/test_1/proj_out_bias.npy (64,)
//    testcases/test_1/rpe_table.npy (4, 63, 63)


    // conv 0: proj q
    conv2d_wbias_prev(d_X, B_x, C_x, H_x, W_x,
                 d_Y, B_y, C_y, H_y, W_y,
                 64, 64, 1, 1,
                 1, 64, 1, 1,
                 1, 1, 0, 1, 1,         // kernel_size, stride, padding, dilation, groups
                 d_proj_q_weight,
                 d_proj_q_bias,
                 handle);
        
    // copy this Q to a new buffer -> to be used later
    float *d_q;
    CUDA_CHECK(cudaMalloc(&d_q, size_y * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Y, d_q, size_y * sizeof(float), cudaMemcpyDeviceToDevice));

    // conv offset neural network part
    float* d_offset = d_Y;
    float* d_out_conv2 = d_X;

    // start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);


    //conv2d

    conv2d_wbias(d_offset, 4, 16, 32, 32,     // input shape
                 n_group_channels*H_x*W_x, H_x*W_x, W_x, 1,   // x_strides
                 d_out_conv2, 4, 16, 32, 32,   // output shape
                 16, 1, 3, 3,                 // filter shape
                 1, 16, 1, 1,                 // bias shape
                 ksize, 
                 stride, 
                 pad_size,
                 1,                           // dilation
                 n_group_channels,            // groups
                 d_conv_offset_0_weight,
                 d_conv_offset_0_bias,
                 handle);

    //layernorm
    float* layernorm_out = d_offset;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    launch_layernorm(
      d_out_conv2,
      nullptr,
      layernorm_out,
      4, 16, 32, 32);

    //gelu

    float* gelu_out = d_out_conv2;

    launch_gelu_tanh_f32(layernorm_out,
                          gelu_out,
                          4 * 16 * 32 * 32,
                          stream);

    //conv2d

    float *offset = d_out_conv2;

    conv2d_nobias(d_offset, 4, 16, 32, 32,     // input shape
                 n_group_channels*H_x*W_x, H_x*W_x, W_x, 1,   // x_strides
                 offset, 4, 2, 32, 32,   // output shape
                 2, 16, 1, 1,                 // filter shape
                 1, 
                 1, 
                 0,
                 1,                           // dilation
                 1,            // groups
                 d_conv_offset_3_weight,
                 handle);

    // offsets?
    //c++ / cuda code for this

    int Hk = 32;
    int Wk = 32;

    //int refsize = Hk*Wk*B_x;
    int refsize = 4 * 32 * 32 * 2;
    //float *ref = (float*)calloc(refsize, sizeof(float));
    //float *d_ref;
    //CUDA_CHECK(cudaMalloc(&d_ref, refsize * sizeof(float)));
    //CUDA_CHECK(cudaMemcpy(d_ref, ref, refsize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridSize(Hk, Wk); 

    get_ref_points_kernel<<<gridSize>>>(offset, Hk, Wk);
    CUDA_CHECK(cudaGetLastError());

    // get ref points
    //cuda kernel? potentially fused with prev

    // grid sample
    // custom kernel

    float *d_X_new;
    CUDA_CHECK(cudaMalloc(&d_X_new, size_x * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X_new, h_X, size_x * sizeof(float), cudaMemcpyHostToDevice));

    float *x_sampled = layernorm_out;

    grid_sample_kernel<<<gridSize>>>(d_X_new, offset, x_sampled, C_x, H_x, W_x, Hk, Wk);
    CUDA_CHECK(cudaGetLastError());

    //norm2<<<1,1>>>(x_sampled, 4 * 2 * 32 * 32);

    float *d_k;
    CUDA_CHECK(cudaMalloc(&d_k, 4 * 16 * 1024 * sizeof(float)));
    float *d_v;
    CUDA_CHECK(cudaMalloc(&d_v, 4 * 16 * 1024 * sizeof(float)));

    // proj k
    conv2d_wbias_prev(
                 x_sampled, 1, 64, 1, 1024,
                 d_k, 1, 64, 1, 1024,
                 64, 64, 1, 1,
                 1, 64, 1, 1,
                 1, 1, 0, 1, 1,         // kernel_size, stride, padding, dilation, groups
                 d_proj_k_weight,
                 d_proj_k_bias,
                 handle);
    // proj v
    conv2d_wbias_prev(
                 x_sampled, 1, 64, 1, 1024,
                 d_v, 1, 64, 1, 1024,
                 64, 64, 1, 1,
                 1, 64, 1, 1,
                 1, 1, 0, 1, 1,         // kernel_size, stride, padding, dilation, groups
                 d_proj_v_weight,
                 d_proj_v_bias,
                 handle);

    sdotprodattn_forward(
        cublas,
        handle,
        d_q, d_k, d_v,
        d_Y,
        B_x * n_heads, n_head_channels, H_x * W_x, 1, 1,
        1.0f
    );

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // TESTCASE
    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    if (size_y == B_x * n_heads * n_head_channels * H_x * W_x) {
      printf("Ran without errors (✅)\n");
      printf("%s passed (✅)\n", testcase);
    } else {
      printf("Shape error on output(❌)\n");
      printf("%s failed (❌)\n", testcase);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // obliterate cudnn
    cudnnDestroy(handle);
    cublasDestroy(cublas);

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

/*


x flat: tensor([-1.9514,  1.3849,  1.3209,  1.0170, -1.9112,  0.0918, -0.3917, -0.2434,
         0.9526,  0.0108])
norm2: tensor(148.5918, grad_fn=<NormBackward1>)
q flat: tensor([-0.4491, -0.2731, -0.3780, -0.0720,  0.4150, -0.1658, -0.7990, -0.0049,
        -0.2562,  1.3731], grad_fn=<SliceBackward0>)
weight flat: tensor([ 0.0799,  0.1074, -0.0124, -0.0280,  0.0018, -0.0075,  0.0301,  0.0350,
        -0.1135, -0.0461], grad_fn=<SliceBackward0>)
bias flat: tensor([ 0.0297,  0.0579, -0.0402,  0.0817,  0.0180, -0.0251,  0.0241, -0.0984,
         0.0046,  0.0573], grad_fn=<SliceBackward0>)

*/