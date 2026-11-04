#include "sgemm.cuh"

#include <cstring>
#include <stdexcept>

namespace {

__global__ void sgemm_naive(int m,
                            int n,
                            int k,
                            float alpha,
                            const float* a,
                            const float* b,
                            float beta,
                            float* c) {
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

template <int Tile>
__global__ void sgemm_tiled(int m,
                            int n,
                            int k,
                            float alpha,
                            const float* a,
                            const float* b,
                            float beta,
                            float* c) {
  __shared__ float a_tile[Tile][Tile];
  __shared__ float b_tile[Tile][Tile];

  const int row = blockIdx.y * Tile + threadIdx.y;
  const int col = blockIdx.x * Tile + threadIdx.x;
  float sum = 0.0f;

  for (int tile_start = 0; tile_start < k; tile_start += Tile) {
    const int a_col = tile_start + threadIdx.x;
    const int b_row = tile_start + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < Tile; ++inner) {
      sum += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = alpha * sum + beta * c[row * n + col];
  }
}

}  // namespace

const char* kernel_name(SgemmKernel kernel) {
  switch (kernel) {
    case SgemmKernel::Naive:
      return "naive";
    case SgemmKernel::Tiled:
      return "tiled";
  }
  return "unknown";
}

SgemmKernel parse_kernel(const char* name) {
  if (std::strcmp(name, "naive") == 0) {
    return SgemmKernel::Naive;
  }
  if (std::strcmp(name, "tiled") == 0) {
    return SgemmKernel::Tiled;
  }
  throw std::invalid_argument("unknown kernel");
}

void launch_sgemm(SgemmKernel kernel,
                  int m,
                  int n,
                  int k,
                  float alpha,
                  const float* a,
                  const float* b,
                  float beta,
                  float* c,
                  cudaStream_t stream) {
  constexpr int tile = 16;
  const dim3 threads(tile, tile);
  const dim3 blocks((n + tile - 1) / tile, (m + tile - 1) / tile);

  switch (kernel) {
    case SgemmKernel::Naive:
      sgemm_naive<<<blocks, threads, 0, stream>>>(m, n, k, alpha, a, b, beta, c);
      break;
    case SgemmKernel::Tiled:
      sgemm_tiled<tile><<<blocks, threads, 0, stream>>>(
          m, n, k, alpha, a, b, beta, c);
      break;
  }
}
