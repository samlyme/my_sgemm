#include "sgemm.cuh"

#include <cstring>
#include <stdexcept>

namespace {

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y < M && x < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[y * K + i] * B[i * N + x];
    }
    C[y * N + x] = alpha * sum + beta * C[y * N + x];
  }
}

template <int BK>
__global__ void sgemm_tiled_basic(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {

  const int y_block = blockIdx.y * blockDim.y;
  const int x_block = blockIdx.x * blockDim.x;

  const int y = y_block + threadIdx.y;
  const int x = x_block + threadIdx.x;

  __shared__ float As[BK][BK];
  __shared__ float Bs[BK][BK];

  float sum = 0.0;

  // load to SMEM
  for (int k = 0; k < K; k += BK) { // Loop over tiles
    // TODO: bounds check for these
    const int A_tile_y = y;
    const int A_tile_x = k + threadIdx.x;
    if (A_tile_y < M && A_tile_x < K) {
      As[threadIdx.y][threadIdx.x] = A[A_tile_y * K + A_tile_x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0;
    }

    const int B_tile_y = k + threadIdx.y;
    const int B_tile_x = x;
    if (B_tile_y < K && B_tile_x < N) {
      Bs[threadIdx.y][threadIdx.x] = B[B_tile_y * N + B_tile_x];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int i = 0; i < BK; i++) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (y < M && x < N) {
    C[y * N + x] = alpha * sum + beta * C[y * N + x];
  }
}

} // namespace

const char *kernel_name(SgemmKernel kernel) {
  switch (kernel) {
  case SgemmKernel::Naive:
    return "naive";
  case SgemmKernel::TiledBasic:
    return "tiled-basic";
  }
  return "unknown";
}

SgemmKernel parse_kernel(const char *name) {
  if (std::strcmp(name, "naive") == 0) {
    return SgemmKernel::Naive;
  }
  if (std::strcmp(name, "tiled-basic") == 0) {
    return SgemmKernel::TiledBasic;
  }
  throw std::invalid_argument("unknown kernel");
}

void launch_sgemm(SgemmKernel kernel, int m, int n, int k, float alpha,
                  const float *a, const float *b, float beta, float *c,
                  cudaStream_t stream) {
  constexpr int tile = 32;
  const dim3 threads(tile, tile);
  const dim3 blocks((n + tile - 1) / tile, (m + tile - 1) / tile);

  switch (kernel) {
  case SgemmKernel::Naive:
    sgemm_naive<<<blocks, threads, 0, stream>>>(m, n, k, alpha, a, b, beta, c);
    break;
  case SgemmKernel::TiledBasic:
    sgemm_tiled_basic<tile><<<blocks, threads, 0, stream>>>(
        m, n, k, alpha, a, b, beta, c);
    break;
  }
}
