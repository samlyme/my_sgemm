#include "sgemm.cuh"

#include <cstring>
#include <stdexcept>

namespace {

__global__ void sgemm_naive(int m, int n, int k, float alpha, const float *a,
                            const float *b, float beta, float *c) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    sum += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = alpha * sum + beta * c[row * n + col];
}

} // namespace

const char *kernel_name(SgemmKernel kernel) {
  switch (kernel) {
  case SgemmKernel::Naive:
    return "naive";
  }
  return "unknown";
}

SgemmKernel parse_kernel(const char *name) {
  if (std::strcmp(name, "naive") == 0) {
    return SgemmKernel::Naive;
  }
  throw std::invalid_argument("unknown kernel");
}

void launch_sgemm(SgemmKernel kernel, int m, int n, int k, float alpha,
                  const float *a, const float *b, float beta, float *c,
                  cudaStream_t stream) {
  constexpr int tile = 16;
  const dim3 threads(tile, tile);
  const dim3 blocks((n + tile - 1) / tile, (m + tile - 1) / tile);

  switch (kernel) {
  case SgemmKernel::Naive:
    sgemm_naive<<<blocks, threads, 0, stream>>>(m, n, k, alpha, a, b, beta, c);
    break;
  }
}
