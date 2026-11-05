#include "sgemm.cuh"

#include <cstring>
#include <stdexcept>

namespace {

constexpr int strip_size = 256;

// One-dimensional blocks cover horizontal strips of C. Consecutive threads
// access consecutive B and C elements, while each A load is shared by a warp.
__global__ void sgemm_row_strip(const int M, const int N, const int K,
                                const float alpha, const float *A,
                                const float *B, const float beta, float *C) {
  const int y = blockIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y < M && x < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[y * K + i] * B[i * N + x];
    }
    C[y * N + x] = alpha * sum + beta * C[y * N + x];
  }
}

void launch_sgemm_row_strip(const int M, const int N, const int K,
                            const float alpha, const float *A, const float *B,
                            const float beta, float *C) {
  const dim3 threads(strip_size);
  const dim3 blocks((N + strip_size - 1) / strip_size, M);
  sgemm_row_strip<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}

// Rotating the strip makes consecutive threads walk down a column. For
// row-major matrices, the A and C accesses are consequently strided.
__global__ void sgemm_col_strip(const int M, const int N, const int K,
                                const float alpha, const float *A,
                                const float *B, const float beta, float *C) {
  const int y = blockIdx.y * blockDim.x + threadIdx.x;
  const int x = blockIdx.x;

  if (y < M && x < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[y * K + i] * B[i * N + x];
    }
    C[y * N + x] = alpha * sum + beta * C[y * N + x];
  }
}

void launch_sgemm_col_strip(const int M, const int N, const int K,
                            const float alpha, const float *A, const float *B,
                            const float beta, float *C) {
  const dim3 threads(strip_size);
  const dim3 blocks(N, (M + strip_size - 1) / strip_size);
  sgemm_col_strip<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}

__global__ void sgemm_naive(const int M, const int N, const int K,
                            const float alpha, const float *A, const float *B,
                            const float beta, float *C) {
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

void launch_sgemm_naive(const int M, const int N, const int K,
                        const float alpha, const float *A, const float *B,
                        const float beta, float *C) {
  constexpr int block_size = 32;
  const dim3 threads(block_size, block_size);
  const dim3 blocks((N + block_size - 1) / block_size,
                    (M + block_size - 1) / block_size);
  sgemm_naive<<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}

template <int BK>
__global__ void sgemm_tiled_basic_kernel(const int M, const int N, const int K,
                                         const float alpha, const float *A,
                                         const float *B, const float beta,
                                         float *C) {
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
void launch_sgemm_tiled_basic(const int M, const int N, const int K,
                              const float alpha, const float *A, const float *B,
                              const float beta, float *C) {
  static_assert(BK > 0 && BK * BK <= 1024,
                "BK must produce a valid CUDA thread block");

  const dim3 threads(BK, BK);
  const dim3 blocks((N + BK - 1) / BK, (M + BK - 1) / BK);
  sgemm_tiled_basic_kernel<BK>
      <<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}

// Trying to increase AI, doesn't really help!
template <int BM, int BN, int BK>
__global__ void sgemm_tiled_general_kernel(const int M, const int N,
                                           const int K, const float alpha,
                                           const float *A, const float *B,
                                           const float beta, float *C) {
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

      if (A_tile_y < M && A_tile_x < K) {
        As[As_y * BK + As_x] = A[A_tile_y * K + A_tile_x];
      } else {
        As[As_y * BK + As_x] = 0.0f;
      }
    }

    for (int id = id_thread; id < BK * BN; id += num_threads) {
      const int Bs_y = id / BN;
      const int Bs_x = id % BN;
      const int B_tile_y = k + Bs_y;
      const int B_tile_x = x_block + Bs_x;

      if (B_tile_y < K && B_tile_x < N) {
        Bs[Bs_y * BN + Bs_x] = B[B_tile_y * N + B_tile_x];
      } else {
        Bs[Bs_y * BN + Bs_x] = 0.0f;
      }
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
void launch_sgemm_tiled_general(const int M, const int N, const int K,
                                const float alpha, const float *A,
                                const float *B, const float beta, float *C) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && BM * BN <= 1024,
                "tile dimensions must produce a valid CUDA thread block");

  const dim3 threads(BN, BM);
  const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_tiled_general_kernel<BM, BN, BK>
      <<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}

// AI successfully increased, but now facing L1 Cache pressure.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tiled_general_coarsened_kernel(
    const int M, const int N, const int K, const float alpha, const float *A,
    const float *B, const float beta, float *C) {
  const int y_block = blockIdx.y * BM;
  const int x_block = blockIdx.x * BN;

  const int y_thread = threadIdx.y * TM;
  const int x_thread = threadIdx.x * TN;
  const int id_thread = threadIdx.y * blockDim.x + threadIdx.x;
  constexpr int num_threads = (BM / TM) * (BN / TN);

  const int y = y_block + y_thread;
  const int x = x_block + x_thread;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float sums[TM][TN] = {};

  for (int k = 0; k < K; k += BK) {
    for (int id = id_thread; id < BM * BK; id += num_threads) {
      const int As_y = id / BK;
      const int As_x = id % BK;
      const int A_tile_y = y_block + As_y;
      const int A_tile_x = k + As_x;

      if (A_tile_y < M && A_tile_x < K) {
        As[As_y * BK + As_x] = A[A_tile_y * K + A_tile_x];
      } else {
        As[As_y * BK + As_x] = 0.0f;
      }
    }

    for (int id = id_thread; id < BK * BN; id += num_threads) {
      const int Bs_y = id / BN;
      const int Bs_x = id % BN;
      const int B_tile_y = k + Bs_y;
      const int B_tile_x = x_block + Bs_x;

      if (B_tile_y < K && B_tile_x < N) {
        Bs[Bs_y * BN + Bs_x] = B[B_tile_y * N + B_tile_x];
      } else {
        Bs[Bs_y * BN + Bs_x] = 0.0f;
      }
    }

    __syncthreads();

    for (int sum_id = 0; sum_id < TM * TN; sum_id++) {
      const int sum_y = sum_id / TN, sum_x = sum_id % TN;
      for (int i = 0; i < BK; i++) {
        sums[sum_y][sum_x] +=
            As[(y_thread + sum_y) * BK + i] *
            Bs[i * BN + x_thread + sum_x];
      }
    }
    __syncthreads();
  }

  for (int sum_id = 0; sum_id < TM * TN; sum_id++) {
    const int sum_y = sum_id / TN, sum_x = sum_id % TN;
    const int output_y = y + sum_y;
    const int output_x = x + sum_x;
    if (output_y < M && output_x < N) {
      C[output_y * N + output_x] =
          alpha * sums[sum_y][sum_x] + beta * C[output_y * N + output_x];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
void launch_sgemm_tiled_general_coarsened(
    const int M, const int N, const int K, const float alpha, const float *A,
    const float *B, const float beta, float *C) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0,
                "tile dimensions must be positive");
  static_assert(BM % TM == 0 && BN % TN == 0,
                "thread tiles must evenly divide the block tile");
  static_assert((BM / TM) * (BN / TN) <= 1024,
                "tile dimensions must produce a valid CUDA thread block");

  const dim3 threads(BN / TN, BM / TM);
  const dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_tiled_general_coarsened_kernel<BM, BN, BK, TM, TN>
      <<<blocks, threads>>>(M, N, K, alpha, A, B, beta, C);
}
} // namespace

const char *kernel_name(SgemmKernel kernel) {
  switch (kernel) {
  case SgemmKernel::RowStrip:
    return "row-strip";
  case SgemmKernel::ColStrip:
    return "col-strip";
  case SgemmKernel::Naive:
    return "naive";
  case SgemmKernel::TiledBasic:
    return "tiled-basic";
  case SgemmKernel::TiledGeneral:
    return "tiled-general";
  case SgemmKernel::TiledGeneral8x8x8:
    return "tiled-general-8x8x8";
  case SgemmKernel::TiledGeneral16x16x8:
    return "tiled-general-16x16x8";
  case SgemmKernel::TiledGeneral16x16x16:
    return "tiled-general-16x16x16";
  case SgemmKernel::TiledGeneral32x8x16:
    return "tiled-general-32x8x16";
  case SgemmKernel::TiledGeneral8x32x16:
    return "tiled-general-8x32x16";
  case SgemmKernel::TiledGeneralCoarsened:
    return "tiled-general-coarsened";
  case SgemmKernel::TiledGeneralCoarsened32x32x8x2x2:
    return "tiled-general-coarsened-32x32x8-2x2";
  case SgemmKernel::TiledGeneralCoarsened64x64x8x4x4:
    return "tiled-general-coarsened-64x64x8-4x4";
  case SgemmKernel::TiledGeneralCoarsened64x64x16x4x4:
    return "tiled-general-coarsened-64x64x16-4x4";
  case SgemmKernel::TiledGeneralCoarsened128x64x8x8x4:
    return "tiled-general-coarsened-128x64x8-8x4";
  }
  return "unknown";
}

SgemmKernel parse_kernel(const char *name) {
  if (std::strcmp(name, "row-strip") == 0) {
    return SgemmKernel::RowStrip;
  }
  if (std::strcmp(name, "col-strip") == 0) {
    return SgemmKernel::ColStrip;
  }
  if (std::strcmp(name, "naive") == 0) {
    return SgemmKernel::Naive;
  }
  if (std::strcmp(name, "tiled-basic") == 0) {
    return SgemmKernel::TiledBasic;
  }
  if (std::strcmp(name, "tiled-general") == 0) {
    return SgemmKernel::TiledGeneral;
  }
  if (std::strcmp(name, "tiled-general-8x8x8") == 0) {
    return SgemmKernel::TiledGeneral8x8x8;
  }
  if (std::strcmp(name, "tiled-general-16x16x8") == 0) {
    return SgemmKernel::TiledGeneral16x16x8;
  }
  if (std::strcmp(name, "tiled-general-16x16x16") == 0) {
    return SgemmKernel::TiledGeneral16x16x16;
  }
  if (std::strcmp(name, "tiled-general-32x8x16") == 0) {
    return SgemmKernel::TiledGeneral32x8x16;
  }
  if (std::strcmp(name, "tiled-general-8x32x16") == 0) {
    return SgemmKernel::TiledGeneral8x32x16;
  }
  if (std::strcmp(name, "tiled-general-coarsened") == 0) {
    return SgemmKernel::TiledGeneralCoarsened;
  }
  if (std::strcmp(name, "tiled-general-coarsened-32x32x8-2x2") == 0) {
    return SgemmKernel::TiledGeneralCoarsened32x32x8x2x2;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x8-4x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x8x4x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x16-4x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x16x4x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-128x64x8-8x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened128x64x8x8x4;
  }
  throw std::invalid_argument("unknown kernel");
}

void launch_sgemm(const SgemmKernel kernel, const int M, const int N,
                  const int K, const float alpha, const float *A,
                  const float *B, const float beta, float *C) {
  switch (kernel) {
  case SgemmKernel::RowStrip:
    launch_sgemm_row_strip(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::ColStrip:
    launch_sgemm_col_strip(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::Naive:
    launch_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledBasic:
    launch_sgemm_tiled_basic<32>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral:
  case SgemmKernel::TiledGeneral16x16x16:
    launch_sgemm_tiled_general<16, 16, 16>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral8x8x8:
    launch_sgemm_tiled_general<8, 8, 8>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral16x16x8:
    launch_sgemm_tiled_general<16, 16, 8>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral32x8x16:
    launch_sgemm_tiled_general<32, 8, 16>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral8x32x16:
    launch_sgemm_tiled_general<8, 32, 16>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened:
  case SgemmKernel::TiledGeneralCoarsened64x64x8x4x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 8, 4, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened32x32x8x2x2:
    launch_sgemm_tiled_general_coarsened<32, 32, 8, 2, 2>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x16x4x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 16, 4, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened128x64x8x8x4:
    launch_sgemm_tiled_general_coarsened<128, 64, 8, 8, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  }
}
