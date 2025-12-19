#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include "cnpy.h"
#include <string>

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


__global__ void layernorm_unfused_serial(
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

__global__ void layernorm_unfused_shuffle(
    const float* __restrict__ x,     // [B,Cg,H,W]
    const float* __restrict__ gamma, // [Cg] or nullptr
    float* __restrict__ y,           // [B,Cg,H,W]
    int B, int Cg, int H, int W,
    float eps)
{
    int idx = blockIdx.x;  // Each block handles one [h, w] pixel across all Cg
    int HW  = H * W;
    if (idx >= B * HW) return;

    int b = idx / HW;
    int s = idx - b * HW;

    const float* xb = x + (b * Cg * HW) + s;
    float* yb = y + (b * Cg * HW) + s;

    // - Compute Mean
    float sum = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        sum += xb[c * HW];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float sh_sum;
    if (threadIdx.x == 0) sh_sum = sum;
    __syncthreads(); // Ensure sh_sum is written before any thread reads it
    
    float mean = sh_sum / (float)Cg;

    // Compute Variance 
    float sq = 0.0f;
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = xb[c * HW] - mean;
        sq += v * v;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sq += __shfl_down_sync(0xffffffff, sq, offset);
    }

    __shared__ float sh_sq;
    if (threadIdx.x == 0) sh_sq = sq;
    __syncthreads(); // Ensure sh_sq is written before any thread reads it

    float inv_std = rsqrtf((sh_sq / (float)Cg) + eps);

    // Apply LayerNorm + Gamma (No GELU)
    for (int c = threadIdx.x; c < Cg; c += blockDim.x) {
        float v = (xb[c * HW] - mean) * inv_std;
        float g = (gamma ? gamma[c] : 1.0f);
        yb[c * HW] = v * g; 
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

void launch_layernorm_serial(const float* d_x, const float* d_gamma, float* d_y,
                      int B, int Cg, int H, int W, cudaStream_t stream)
{
    int blocks = B * H * W;
    int threads = 32;     // fine for Cg=16
    float eps = 1e-5f;

    layernorm_unfused_serial<<<blocks, threads, 0, stream>>>(
        d_x, d_gamma, d_y, B, Cg, H, W, eps
    );
}

void launch_layernorm_shuffle(const float* d_x, const float* d_gamma, float* d_y,
                      int B, int Cg, int H, int W, cudaStream_t stream)
{
    int blocks = B * H * W;
    int threads = 32;     // fine for Cg=16
    float eps = 1e-5f;

    layernorm_unfused_shuffle<<<blocks, threads, 0, stream>>>(
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

int main(int argc, char** argv) {
    // 0. Parse command line argument for test folder
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <test_dir_name> (e.g., test1)\n", argv[0]);
        return 1;
    }

    std::string test_dir = argv[1];
    std::string base_path = "tests/" + test_dir + "/";

    // 1. Load inputs and references from dynamic path
    // We use .c_str() because cnpy::npy_load expects a const char*
    cnpy::NpyArray xA    = cnpy::npy_load((base_path + "testx.npy").c_str());
    cnpy::NpyArray gA    = cnpy::npy_load((base_path + "testgamma.npy").c_str());
    cnpy::NpyArray yFinA = cnpy::npy_load((base_path + "testy_ref.npy").c_str());

    float* x_h     = xA.data<float>();
    float* g_h     = gA.data<float>();
    float* y_ref_h = yFinA.data<float>();

    auto xs = xA.shape;
    int B = (int)xs[0], Cg = (int)xs[1], H = (int)xs[2], W = (int)xs[3];
    size_t N = (size_t)B * Cg * H * W;
    int n = (int)N;

    // 2. Device Allocation
    float *d_x, *d_g, *d_y_serial, *d_y_shfl, *d_y_fused;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, Cg * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_serial, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_shfl,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_fused,  N * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_x, x_h, N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_g, g_h, Cg * sizeof(float), cudaMemcpyHostToDevice, stream));

    // --- CORRECTNESS SECTION ---
    launch_layernorm_serial(d_x, d_g, d_y_serial, B, Cg, H, W, stream);
    launch_gelu_tanh_f32(d_y_serial, d_y_serial, n, stream);

    launch_layernorm_shuffle(d_x, d_g, d_y_shfl, B, Cg, H, W, stream);
    launch_gelu_tanh_f32(d_y_shfl, d_y_shfl, n, stream);

    launch_layernorm_gelu_fused(d_x, d_g, d_y_fused, B, Cg, H, W, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_y_serial(N), h_y_shfl(N), h_y_fused(N);
    CUDA_CHECK(cudaMemcpy(h_y_serial.data(), d_y_serial, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y_shfl.data(),   d_y_shfl,   N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y_fused.data(),  d_y_fused,  N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("--- Correctness for %s vs Python Reference ---\n", test_dir.c_str());
    compare_arrays("Unfused Serial ", h_y_serial.data(), y_ref_h, N);
    compare_arrays("Unfused Shuffle", h_y_shfl.data(),   y_ref_h, N);
    compare_arrays("Fused Kernel   ", h_y_fused.data(),  y_ref_h, N);

    // --- BENCHMARKING SECTION ---
    const int iterations = 1000;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 10; ++i) {
        launch_layernorm_gelu_fused(d_x, d_g, d_y_fused, B, Cg, H, W, stream);
    }

    float time_serial = 0;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        launch_layernorm_serial(d_x, d_g, d_y_serial, B, Cg, H, W, stream);
        launch_gelu_tanh_f32(d_y_serial, d_y_serial, n, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_serial, start, stop));

    float time_shfl = 0;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        launch_layernorm_shuffle(d_x, d_g, d_y_shfl, B, Cg, H, W, stream);
        launch_gelu_tanh_f32(d_y_shfl, d_y_shfl, n, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_shfl, start, stop));

    float time_fused = 0;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        launch_layernorm_gelu_fused(d_x, d_g, d_y_fused, B, Cg, H, W, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_fused, start, stop));

    printf("\n--- Performance Results (%d iterations) ---\n", iterations);
    printf("Unfused Serial:  %8.4f ms\n", time_serial / iterations);
    printf("Unfused Shuffle: %8.4f ms (Speedup vs Serial: %.2fx)\n", 
            time_shfl / iterations, time_serial / time_shfl);
    printf("Fused Optimized: %8.4f ms (Speedup vs Serial: %.2fx)\n", 
            time_fused / iterations, time_serial / time_fused);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_y_serial); cudaFree(d_y_shfl); cudaFree(d_y_fused);
    cudaStreamDestroy(stream);
    return 0;
}