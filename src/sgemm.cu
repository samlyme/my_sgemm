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
__global__ void sgemm_tiled_basic_kernel(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {
  const int y_block = blockIdx.y * BK;
  const int x_block = blockIdx.x * BK;

  const int y_thread = threadIdx.y;
  const int x_thread = threadIdx.x;

  const int y = y_block + y_thread;
  const int x = x_block + x_thread;

  __shared__ float As[BK][BK];
  __shared__ float Bs[BK][BK];

  float sum = 0.0;

  // load to SMEM
  for (int k = 0; k < K; k += BK) { // Loop over tiles
    // TODO: bounds check for these
    const int A_tile_y = y;
    const int A_tile_x = k + x_thread;
    if (A_tile_y < M && A_tile_x < K) {
      As[y_thread][x_thread] = A[A_tile_y * K + A_tile_x];
    } else {
      As[y_thread][x_thread] = 0.0;
    }

    const int B_tile_y = k + y_thread;
    const int B_tile_x = x;
    if (B_tile_y < K && B_tile_x < N) {
      Bs[y_thread][x_thread] = B[B_tile_y * N + B_tile_x];
    } else {
      Bs[y_thread][x_thread] = 0.0;
    }

    __syncthreads();

    for (int i = 0; i < BK; i++) {
      sum += As[y_thread][i] * Bs[i][x_thread];
    }
    __syncthreads();
  }

  if (y < M && x < N) {
    C[y * N + x] = alpha * sum + beta * C[y * N + x];
  }
}

template <int BK>
void launch_sgemm_tiled_basic(int m, int n, int k, float alpha, const float *a,
                              const float *b, float beta, float *c,
                              cudaStream_t stream) {
  static_assert(BK > 0 && BK * BK <= 1024,
                "BK must produce a valid CUDA thread block");

  const dim3 threads(BK, BK);
  const dim3 blocks((n + BK - 1) / BK, (m + BK - 1) / BK);
  sgemm_tiled_basic_kernel<BK>
      <<<blocks, threads, 0, stream>>>(m, n, k, alpha, a, b, beta, c);
}

template <int BM, int BN, int BK>
__global__ void sgemm_tiled_general_kernel(int M, int N, int K, float alpha,
                                           const float *A, const float *B,
                                           float beta, float *C) {
  const int y_block = blockIdx.y * BM;
  const int x_block = blockIdx.x * BN;

  const int y_thread = threadIdx.y;
  const int x_thread = threadIdx.x;
  const int id_thread = y_thread * BN + x_thread;
  constexpr int num_threads = BM * BN;

  const int y = y_block + y_thread;
  const int x = x_block + x_thread;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float sum = 0.0;

  for (int k = 0; k < K; k += BK) {
    for (int id = id_thread; id < BM * BK; id += num_threads) {
      const int As_y = id / BK;
      const int As_x = id % BK;
      const int A_tile_y = y_block + As_y;
      const int A_tile_x = k + As_x;
      As[id] = A_tile_y < M && A_tile_x < K
                   ? A[A_tile_y * K + A_tile_x]
                   : 0.0f;
    }

    for (int id = id_thread; id < BK * BN; id += num_threads) {
      const int Bs_y = id / BN;
      const int Bs_x = id % BN;
      const int B_tile_y = k + Bs_y;
      const int B_tile_x = x_block + Bs_x;
      Bs[id] = B_tile_y < K && B_tile_x < N
                   ? B[B_tile_y * N + B_tile_x]
                   : 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < BK; i++) {
      sum += As[y_thread * BK + i] * Bs[i * BN + x_thread];
    }
    __syncthreads();
  }

  if (y < M && x < N) {
    C[y * N + x] = alpha * sum + beta * C[y * N + x];
  }
}

template <int BM, int BN, int BK>
void launch_sgemm_tiled_general(int m, int n, int k, float alpha,
                                const float *a, const float *b, float beta,
                                float *c, cudaStream_t stream) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && BM * BN <= 1024,
                "tile dimensions must produce a valid CUDA thread block");

  const dim3 threads(BN, BM);
  const dim3 blocks((n + BN - 1) / BN, (m + BM - 1) / BM);
  sgemm_tiled_general_kernel<BM, BN, BK>
      <<<blocks, threads, 0, stream>>>(m, n, k, alpha, a, b, beta, c);
}
} // namespace

const char *kernel_name(SgemmKernel kernel) {
  switch (kernel) {
  case SgemmKernel::Naive:
    return "naive";
  case SgemmKernel::TiledBasic:
    return "tiled-basic";
  case SgemmKernel::TiledGeneral:
    return "tiled-general";
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
  if (std::strcmp(name, "tiled-general") == 0) {
    return SgemmKernel::TiledGeneral;
  }
  throw std::invalid_argument("unknown kernel");
}

void launch_sgemm(SgemmKernel kernel, int m, int n, int k, float alpha,
                  const float *a, const float *b, float beta, float *c,
                  cudaStream_t stream) {
  switch (kernel) {
  case SgemmKernel::Naive: {
    constexpr int block_size = 32;
    const dim3 blockDim(block_size, block_size);
    const dim3 gridDim((n + block_size - 1) / block_size,
                       (m + block_size - 1) / block_size);
    sgemm_naive<<<gridDim, blockDim, 0, stream>>>(m, n, k, alpha, a, b, beta,
                                                  c);
    break;
  }
  case SgemmKernel::TiledBasic:
    launch_sgemm_tiled_basic<32>(m, n, k, alpha, a, b, beta, c, stream);
    break;
  case SgemmKernel::TiledGeneral:
    launch_sgemm_tiled_general<16, 16, 16>(m, n, k, alpha, a, b, beta, c,
                                           stream);
    break;
  }
}
