// kyletest/test_proj_q.cu
// Validates proj_q against reference using:
//   x.txt (custom format)
//   proj_q_weight.npy
//   proj_q_bias.npy
//   q_ref.npy

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>

#define CUDA_CHECK(call)                                   \
  do {                                                     \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n",            \
              __FILE__, __LINE__, cudaGetErrorString(err));\
      exit(1);                                             \
    }                                                      \
  } while (0)

/* ===================== TXT LOADER (x.txt) ===================== */

std::vector<float> load_x_txt(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Failed to open " << path << "\n";
    exit(1);
  }

  std::string line;
  bool in_mat = false;
  std::vector<float> data;

  while (std::getline(in, line)) {
    if (line == "  mat") {
      in_mat = true;
      continue;
    }
    if (!in_mat) continue;

    // Strip brackets
    for (char& c : line) {
      if (c == '[' || c == ']') c = ' ';
    }

    // Parse floats
    std::stringstream ss(line);
    float v;
    while (ss >> v) {
      data.push_back(v);
    }
  }

  return data;
}

/* ===================== NPY LOADER (float32, C-order) ===================== */
/* Minimal loader: supports .npy v1.0 / v2.0, float32, little-endian */

std::vector<float> load_npy_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "Failed to open " << path << "\n";
    exit(1);
  }

  char magic[6];
  f.read(magic, 6);
  if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
    std::cerr << "Invalid npy file: " << path << "\n";
    exit(1);
  }

  uint8_t ver[2];
  f.read((char*)ver, 2);

  uint16_t header_len;
  f.read((char*)&header_len, 2);

  std::string header(header_len, ' ');
  f.read(&header[0], header_len);

  if (header.find("'descr': '<f4'") == std::string::npos) {
    std::cerr << "Only float32 npy supported\n";
    exit(1);
  }
  if (header.find("'fortran_order': False") == std::string::npos) {
    std::cerr << "Fortran-order npy not supported\n";
    exit(1);
  }

  // Extract shape
  auto l = header.find('(');
  auto r = header.find(')');
  std::string shape_str = header.substr(l + 1, r - l - 1);

  size_t count = 1;
  size_t pos = 0;
  while (pos < shape_str.size()) {
    size_t comma = shape_str.find(',', pos);
    size_t end = (comma == std::string::npos) ? shape_str.size() : comma;
    std::string dim = shape_str.substr(pos, end - pos);
    int d = std::atoi(dim.c_str());
    if (d > 0) count *= d;
    pos = end + 1;
  }

  std::vector<float> data(count);
  f.read((char*)data.data(), count * sizeof(float));
  return data;
}

/* ===================== CUDA KERNEL ===================== */

__global__ void proj_q_kernel(const float* x,
                              const float* Wq,
                              const float* bq,
                              float* q,
                              int B, int C, int H, int W) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;

  if (w >= W || h >= H || b >= B) return;

  int base = ((b * C) * H + h) * W + w;

  for (int co = 0; co < C; co++) {
    float acc = bq[co];
    for (int ci = 0; ci < C; ci++) {
      acc += Wq[co * C + ci] * x[base + ci * H * W];
    }
    q[base + co * H * W] = acc;
  }
}

/* ===================== MAIN ===================== */

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "Usage:\n"
              << argv[0]
              << " x.txt Wq.npy bq.npy q_ref.npy B C H W\n";
    return 1;
  }

  std::string x_path   = argv[1];
  std::string Wq_path  = argv[2];
  std::string bq_path  = argv[3];
  std::string qref_path= argv[4];
  int B = atoi(argv[5]);
  int C = atoi(argv[6]);
  int H = atoi(argv[7]);
  int W = atoi(argv[8]);

  /* Load data */
  auto x  = load_x_txt(x_path);
  auto Wq = load_npy_f32(Wq_path);
  auto bq = load_npy_f32(bq_path);
  auto q_ref = load_npy_f32(qref_path);

  size_t expected = (size_t)B * C * H * W;
  size_t expected_x = (size_t)B * C * H * W;
size_t expected_Wq = (size_t)C * C;
size_t expected_bq = (size_t)C;

if (x.size() != expected_x) {
  std::cerr << "x size mismatch: " << x.size()
            << " vs " << expected_x << "\n";
  return 1;
}

if (q_ref.size() != expected_x) {
  std::cerr << "q_ref size mismatch: " << q_ref.size()
            << " vs " << expected_x << "\n";
  return 1;
}

if (Wq.size() != expected_Wq) {
  std::cerr << "Wq element count mismatch: " << Wq.size()
            << " vs " << expected_Wq
            << " (expected C*C, extra dims allowed)\n";
  return 1;
}

if (bq.size() != expected_bq) {
  std::cerr << "bq element count mismatch: " << bq.size()
            << " vs " << expected_bq << "\n";
  return 1;
}


  /* Device buffers */
  float *d_x, *d_Wq, *d_bq, *d_q;
  CUDA_CHECK(cudaMalloc(&d_x,  x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Wq, Wq.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bq, bq.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q,  q_ref.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x,  x.data(),  x.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wq, Wq.data(), Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bq, bq.data(), bq.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16, B);

  proj_q_kernel<<<grid, block>>>(d_x, d_Wq, d_bq, d_q, B, C, H, W);
  CUDA_CHECK(cudaDeviceSynchronize());

  /* Copy back */
  std::vector<float> q_out(q_ref.size());
  CUDA_CHECK(cudaMemcpy(q_out.data(), d_q,
                        q_out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  /* Compare */
  float max_err = 0.f;
  for (size_t i = 0; i < q_out.size(); i++) {
    float e = fabs(q_out[i] - q_ref[i]);
    max_err = fmax(max_err, e);
  }

  printf("Max abs error: %.6e\n", max_err);
  printf("%s\n", (max_err < 1e-5f) ? "PASS" : "FAIL");

  return 0;
}
