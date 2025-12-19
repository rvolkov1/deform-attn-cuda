#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include "cnpy.h"

#define CUDA_CHECK(call) do {                                        \
    cudaError_t err__ = (call);                                      \
    if (err__ != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err__));      \
        std::exit(EXIT_FAILURE);                                     \
    }                                                                \
} while (0)

__global__ void layernorm_gelu_fused(
    const float* __restrict__ x,     // [B,Cg,H,W]
    const float* __restrict__ gamma, // [Cg] or nullptr
    float* __restrict__ y,           // [B,Cg,H,W]
    int B, int Cg, int H, int W,
    float eps)
{
    int idx = blockIdx.x;
    int HW  = H * W;
    if (idx >= B * HW) return;

    int b = idx / HW;
    int s = idx - b * HW;

    const float* xb = x + (b * Cg * HW) + s;
    float* yb = y + (b * Cg * HW) + s;

    // Initialize shared memory (Must be done by all threads or thread 0 alone)
    __shared__ float sh_sum;
    __shared__ float sh_sq;
    if (threadIdx.x == 0) {
        sh_sum = 0.0f;
        sh_sq = 0.0f;
    }
    __syncthreads();

    //mean
    float sum = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        sum += xb[c * HW];
    }

    // Register-level Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only one thread per warp contributes to the block sum
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&sh_sum, sum);
    }
    __syncthreads();//Wait for all warps to finish atomicAdd
    float mean = sh_sum / (float)Cg;

    //variance
    float sq = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = xb[c * HW] - mean;
        sq += v * v;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sq += __shfl_down_sync(0xffffffff, sq, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&sh_sq, sq);
    }
    __syncthreads(); // Wait for all warps to finish atomicAdd
    float var = sh_sq / (float)Cg;

    float inv_std = rsqrtf(var + eps);

    // output and gelu calculation
    const float kAlpha = 0.79788456f;
    const float kBeta  = 0.044715f;

    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = (xb[c * HW] - mean) * inv_std;
        float g = (gamma ? gamma[c] : 1.0f);
        float z = v * g; 

        float z3 = z * z * z;
        float t  = kAlpha * (z + kBeta * z3);
        yb[c * HW] = 0.5f * z * (1.0f + tanhf(t));
    }
}

void launch_layernorm_gelu_fused(const float* d_x, const float* d_gamma, float* d_y,
                                 int B, int Cg, int H, int W, cudaStream_t stream)
{
    int blocks = B * H * W;
    int threads = 32;     // same as before
    float eps = 1e-5f;

    layernorm_gelu_fused<<<blocks, threads, 0, stream>>>(
        d_x, d_gamma, d_y, B, Cg, H, W, eps
    );
}


__global__ void layernorm_nchw_over_c_nobias_f32(
    const float* __restrict__ x,    // [B,Cg,H,W]
    const float* __restrict__ gamma,// [Cg] or nullptr
    float* __restrict__ y,          // [B,Cg,H,W]
    int B, int Cg, int H, int W,
    float eps)
{
    int idx = blockIdx.x;  // idx in [0, B*H*W)
    int HW  = H * W;
    if (idx >= B * HW) return;

    int b = idx / HW;
    int s = idx - b * HW;  // s = h*W + w

    const float* xb = x + (b * Cg * HW) + s;
    float*       yb = y + (b * Cg * HW) + s;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        sum += xb[c * HW];
    }

    __shared__ float sh_sum;
    if (threadIdx.x == 0) sh_sum = 0.0f;
    __syncthreads();
    atomicAdd(&sh_sum, sum);
    __syncthreads();
    float mean = sh_sum / (float)Cg;

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

    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = (xb[c * HW] - mean) * inv_std;
        float g = (gamma ? gamma[c] : 1.0f);
        yb[c * HW] = v * g; // no beta
    }
}

__global__ void gelu_tanh_f32_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = x[i];
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;

    float v3 = v * v * v;
    float t  = kAlpha * (v + kBeta * v3);
    y[i] = 0.5f * v * (1.0f + tanhf(t));
}

void launch_layernorm(const float* d_x, const float* d_gamma, float* d_y,
                      int B, int Cg, int H, int W, cudaStream_t stream)
{
    int blocks = B * H * W;
    int threads = 32;     // fine for Cg=16
    float eps = 1e-5f;

    layernorm_nchw_over_c_nobias_f32<<<blocks, threads, 0, stream>>>(
        d_x, d_gamma, d_y, B, Cg, H, W, eps
    );
}

void launch_gelu_tanh_f32(const float* d_x, float* d_y, int n, cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    gelu_tanh_f32_kernel<<<grid, block, 0, stream>>>(d_x, d_y, n);
}

static void compare_arrays(const char* label,
                           const float* out,
                           const float* ref,
                           size_t N)
{
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float diff = std::fabs(out[i] - ref[i]);
        sum_abs += diff;
        if (diff > max_abs) max_abs = diff;
    }
    printf("%s: max_abs=%.6g mean_abs=%.6g\n",
           label, max_abs, (float)(sum_abs / (double)N));
}

int main() {
    // Load inputs and references
    cnpy::NpyArray xA    = cnpy::npy_load("testx.npy");
    cnpy::NpyArray gA    = cnpy::npy_load("testgamma.npy");
    cnpy::NpyArray yLnA  = cnpy::npy_load("testy_ln_ref.npy"); // intermediate LN ref
    cnpy::NpyArray yFinA = cnpy::npy_load("testy_ref.npy");    // final GELU(LN(x)) ref

    float* x_h     = xA.data<float>();
    float* g_h     = gA.data<float>();
    float* y_ln_h  = yLnA.data<float>();
    float* y_ref_h = yFinA.data<float>();

    // Extract dims from x (assumes [B,Cg,H,W])
    auto xs = xA.shape;
    int B  = (int)xs[0];
    int Cg = (int)xs[1];
    int H  = (int)xs[2];
    int W  = (int)xs[3];
    size_t N = (size_t)B * (size_t)Cg * (size_t)H * (size_t)W;

    // Basic shape sanity checks
    if (gA.shape.size() != 1 || (int)gA.shape[0] != Cg) {
        printf("gamma shape mismatch: expected [%d]\n", Cg);
        return 1;
    }
    if (yLnA.shape != xA.shape || yFinA.shape != xA.shape) {
        printf("reference output shape mismatch: expected same shape as x\n");
        return 1;
    }

    // Device alloc
    float *d_x = nullptr, *d_g = nullptr, *d_y = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, (size_t)Cg * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_x, x_h, N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_g, g_h, (size_t)Cg * sizeof(float), cudaMemcpyHostToDevice, stream));

    // N = (size_t)B * Cg * H * W;
    int n = (int)N;

    // Device outputs
    float* d_y_unfused = nullptr;
    float* d_y_fused   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_y_unfused, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_fused,   N * sizeof(float)));

    // 1) Unfused GPU reference
    launch_layernorm(d_x, d_g, d_y_unfused, B, Cg, H, W, stream);
    CUDA_CHECK(cudaGetLastError());

    launch_gelu_tanh_f32(d_y_unfused, d_y_unfused, n, stream);
    CUDA_CHECK(cudaGetLastError());

    // 2) Fused
    launch_layernorm_gelu_fused(d_x, d_g, d_y_fused, B, Cg, H, W, stream);
    CUDA_CHECK(cudaGetLastError());

    // Copy back once, compare twice
    std::vector<float> y_unfused_h(N), y_fused_h(N);
    CUDA_CHECK(cudaMemcpyAsync(y_unfused_h.data(), d_y_unfused, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(y_fused_h.data(),   d_y_fused,   N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // A) Fused vs Python baseline (final output)
    compare_arrays("FUSED vs PY (GELU(LN))", y_fused_h.data(), y_ref_h, N);

    // B) Fused vs unfused GPU (final output)
    compare_arrays("FUSED vs UNFUSED (GPU)", y_fused_h.data(), y_unfused_h.data(), N);

/*
Benchmarking
*/
    const int iterations = 1000;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    for (int i = 0; i < 10; ++i) {
        launch_layernorm(d_x, d_g, d_y_unfused, B, Cg, H, W, stream);
        launch_gelu_tanh_f32(d_y_unfused, d_y_unfused, n, stream);
        launch_layernorm_gelu_fused(d_x, d_g, d_y_fused, B, Cg, H, W, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //  Measure Unfused Kernel (LN + GELU)
    float unfused_time = 0;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        launch_layernorm(d_x, d_g, d_y_unfused, B, Cg, H, W, stream);
        launch_gelu_tanh_f32(d_y_unfused, d_y_unfused, n, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&unfused_time, start, stop));

    //Measure Fused Kernel 
    float fused_time = 0;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        launch_layernorm_gelu_fused(d_x, d_g, d_y_fused, B, Cg, H, W, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&fused_time, start, stop));

    // Output Results
    float avg_unfused = unfused_time / iterations;
    float avg_fused   = fused_time / iterations;

    printf("\n--- Performance Results (%d iterations) ---\n", iterations);
    printf("Unfused (LN + GELU): %8.4f ms\n", avg_unfused);
    printf("Fused (LN_GELU):     %8.4f ms\n", avg_fused);
    printf("Speedup:             %8.2fx\n", avg_unfused / avg_fused);

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    CUDA_CHECK(cudaFree(d_y_unfused));
    CUDA_CHECK(cudaFree(d_y_fused));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
