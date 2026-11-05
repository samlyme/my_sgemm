#include "sgemm.cuh"

#include <cstring>
#include <stdexcept>

namespace {
__global__ void sgemm_naive(const int M, const int N, const int K,
                            const float alpha, const float *A, const float *B,
                            const float beta, float *C) {
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y < M && x < N) {
    float sum = 0;
    for (int i = 0; i < K; i++) {
      sum += A[y * K + i] * B[i * N + x];
    }

    C[y * N + x] = alpha * sum + beta * C[y * N + x];
  }
}

template <int BM, int BN>
void launch_sgemm_naive(const int M, const int N, const int K,
                        const float alpha, const float *A, const float *B,
                        const float beta, float *C) {
  dim3 blockDim(BN, BM);
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

template <int BK>
__global__ void sgemm_tiled_basic(const int M, const int N, const int K,
                                  const float alpha, const float *A,
                                  const float *B, const float beta, float *C) {
  const int yb = blockIdx.y * blockDim.y;
  const int xb = blockIdx.x * blockDim.x;

  const int yt = threadIdx.y;
  const int xt = threadIdx.x;

  const int y = yb + yt;
  const int x = xb + xt;

  __shared__ float As[BK][BK];
  __shared__ float Bs[BK][BK];

  float sum = 0.0;

  for (int k = 0; k < K; k += BK) {
    const int A_tile_y = y;
    const int A_tile_x = k + xt;
    if (A_tile_y < M && A_tile_x < K) {
      As[yt][xt] = A[A_tile_y * K + A_tile_x];
    } else {
      As[yt][xt] = 0.0;
    }

    const int B_tile_y = k + yt;
    const int B_tile_x = x;
    if (B_tile_y < K && B_tile_x < N) {
      Bs[yt][xt] = B[B_tile_y * N + B_tile_x];
    } else {
      Bs[yt][xt] = 0.0;
    }

    __syncthreads();

    for (int i = 0; i < BK; i++) {
      sum += As[yt][i] * Bs[i][xt];
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

  dim3 blockDim(BK, BK);
  dim3 gridDim((N + BK - 1) / BK, (M + BK - 1) / BK);
  sgemm_tiled_basic<BK><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

template <int BM, int BN, int BK>
__global__ void sgemm_tiled_general(const int M, const int N, const int K,
                                    const float alpha, const float *A,
                                    const float *B, const float beta,
                                    float *C) {
  const int yb = blockIdx.y * BM;
  const int xb = blockIdx.x * BN;

  const int yt = threadIdx.y;
  const int xt = threadIdx.x;

  const int y = yb + yt;
  const int x = xb + xt;

  const int tid = yt * BN + xt;
  const int numThreads = BM * BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float sum = 0.0;

  for (int k = 0; k < K; k += BK) {
    for (int id = tid; id < BM * BK; id += numThreads) {
      const int yid = id / BK;
      const int xid = id % BK;
      const int A_tile_y = yb + yid;
      const int A_tile_x = k + xid;

      if (A_tile_y < M && A_tile_x < K) {
        As[id] = A[A_tile_y * K + A_tile_x];
      } else {
        As[id] = 0.0;
      }
    }

    for (int id = tid; id < BK * BN; id += numThreads) {
      const int yid = id / BN;
      const int xid = id % BN;
      const int B_tile_y = k + yid;
      const int B_tile_x = xb + xid;

      if (B_tile_y < K && B_tile_x < N) {
        Bs[id] = B[B_tile_y * N + B_tile_x];
      } else {
        Bs[id] = 0.0;
      }
    }

    __syncthreads();

    for (int i = 0; i < BK; i++) {
      sum += As[yt * BK + i] * Bs[i * BN + xt];
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
                "BM and BN must produce a valid CUDA thread block");
  static_assert((BM * BK + BK * BN) * sizeof(float) <= 48 * 1024,
                "tile must fit in 48 KiB of shared memory");

  dim3 blockDim(BN, BM);
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_tiled_general<BM, BN, BK>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tiled_general_coarsened(const int M, const int N,
                                              const int K, const float alpha,
                                              const float *A, const float *B,
                                              const float beta, float *C) {
  const int yb = blockIdx.y * BM;
  const int xb = blockIdx.x * BN;

  const int yt = threadIdx.y;
  const int xt = threadIdx.x;

  const int y = yb + yt * TM;
  const int x = xb + xt * TN;

  const int tid = yt * blockDim.x + xt;
  const int numThreads = blockDim.x * blockDim.y;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float sums[TM][TN] = {};

  for (int k = 0; k < K; k += BK) {
    for (int id = tid; id < BM * BK; id += numThreads) {
      const int yid = id / BK;
      const int xid = id % BK;
      const int A_tile_y = yb + yid;
      const int A_tile_x = k + xid;

      if (A_tile_y < M && A_tile_x < K) {
        As[id] = A[A_tile_y * K + A_tile_x];
      } else {
        As[id] = 0.0;
      }
    }

    for (int id = tid; id < BK * BN; id += numThreads) {
      const int yid = id / BN;
      const int xid = id % BN;
      const int B_tile_y = k + yid;
      const int B_tile_x = xb + xid;

      if (B_tile_y < K && B_tile_x < N) {
        Bs[id] = B[B_tile_y * N + B_tile_x];
      } else {
        Bs[id] = 0.0;
      }
    }

    __syncthreads();

    for (int i = 0; i < BK; i++) {
      float a[TM];
      float b[TN];

      for (int ty = 0; ty < TM; ty++) {
        a[ty] = As[(yt * TM + ty) * BK + i];
      }
      for (int tx = 0; tx < TN; tx++) {
        b[tx] = Bs[i * BN + xt * TN + tx];
      }

      for (int ty = 0; ty < TM; ty++) {
        for (int tx = 0; tx < TN; tx++) {
          sums[ty][tx] += a[ty] * b[tx];
        }
      }
    }
    __syncthreads();
  }
  for (int ty = 0; ty < TM; ty++) {
    for (int tx = 0; tx < TN; tx++) {
      if (y + ty < M && x + tx < N) {
        const int C_idx = (y + ty) * N + x + tx;
        C[C_idx] = alpha * sums[ty][tx] + beta * C[C_idx];
      }
    }
  }
}
template <int BM, int BN, int BK, int TM, int TN>
void launch_sgemm_tiled_general_coarsened(const int M, const int N, const int K,
                                          const float alpha, const float *A,
                                          const float *B, const float beta,
                                          float *C) {
  static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0,
                "tile dimensions must be positive");
  static_assert(BN % TN == 0 && BM % TM == 0,
                "thread tiles must divide the block tile evenly");
  static_assert((BM / TM) * (BN / TN) <= 1024,
                "coarsening must produce a valid CUDA thread block");
  static_assert((BM * BK + BK * BN) * sizeof(float) <= 48 * 1024,
                "tile must fit in 48 KiB of shared memory");

  dim3 blockDim(BN / TN, BM / TM);
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_tiled_general_coarsened<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
} // namespace

const char *kernel_name(SgemmKernel kernel) {
  switch (kernel) {
  case SgemmKernel::Naive:
    return "naive";
  case SgemmKernel::Naive8x8:
    return "naive-8x8";
  case SgemmKernel::Naive16x16:
    return "naive-16x16";
  case SgemmKernel::Naive32x32:
    return "naive-32x32";
  case SgemmKernel::Naive64x16:
    return "naive-64x16";
  case SgemmKernel::Naive16x64:
    return "naive-16x64";
  case SgemmKernel::TiledBasic:
    return "tiled-basic";
  case SgemmKernel::TiledBasic8:
    return "tiled-basic-8";
  case SgemmKernel::TiledBasic16:
    return "tiled-basic-16";
  case SgemmKernel::TiledBasic32:
    return "tiled-basic-32";
  case SgemmKernel::TiledGeneral:
    return "tiled-general";
  case SgemmKernel::TiledGeneral16x16x32:
    return "tiled-general-16x16x32";
  case SgemmKernel::TiledGeneral16x16x64:
    return "tiled-general-16x16x64";
  case SgemmKernel::TiledGeneral16x16x128:
    return "tiled-general-16x16x128";
  case SgemmKernel::TiledGeneral16x16x256:
    return "tiled-general-16x16x256";
  case SgemmKernel::TiledGeneral32x32x64:
    return "tiled-general-32x32x64";
  case SgemmKernel::TiledGeneral32x32x96:
    return "tiled-general-32x32x96";
  case SgemmKernel::TiledGeneral32x32x128:
    return "tiled-general-32x32x128";
  case SgemmKernel::TiledGeneralCoarsened:
    return "tiled-general-coarsened";
  case SgemmKernel::TiledGeneralCoarsened64x64x8_2x2:
    return "tiled-general-coarsened-64x64x8-2x2";
  case SgemmKernel::TiledGeneralCoarsened64x64x8_4x4:
    return "tiled-general-coarsened-64x64x8-4x4";
  case SgemmKernel::TiledGeneralCoarsened64x64x8_4x8:
    return "tiled-general-coarsened-64x64x8-4x8";
  case SgemmKernel::TiledGeneralCoarsened64x64x8_8x4:
    return "tiled-general-coarsened-64x64x8-8x4";
  case SgemmKernel::TiledGeneralCoarsened64x64x8_8x8:
    return "tiled-general-coarsened-64x64x8-8x8";
  case SgemmKernel::TiledGeneralCoarsened64x64x16_4x4:
    return "tiled-general-coarsened-64x64x16-4x4";
  case SgemmKernel::TiledGeneralCoarsened64x64x16_4x8:
    return "tiled-general-coarsened-64x64x16-4x8";
  case SgemmKernel::TiledGeneralCoarsened64x64x16_8x4:
    return "tiled-general-coarsened-64x64x16-8x4";
  case SgemmKernel::TiledGeneralCoarsened64x64x16_8x8:
    return "tiled-general-coarsened-64x64x16-8x8";
  case SgemmKernel::TiledGeneralCoarsened64x64x32_4x4:
    return "tiled-general-coarsened-64x64x32-4x4";
  case SgemmKernel::TiledGeneralCoarsened64x64x32_4x8:
    return "tiled-general-coarsened-64x64x32-4x8";
  case SgemmKernel::TiledGeneralCoarsened64x64x32_8x4:
    return "tiled-general-coarsened-64x64x32-8x4";
  case SgemmKernel::TiledGeneralCoarsened64x64x32_8x8:
    return "tiled-general-coarsened-64x64x32-8x8";
  case SgemmKernel::TiledGeneralCoarsened128x64x8_8x4:
    return "tiled-general-coarsened-128x64x8-8x4";
  case SgemmKernel::TiledGeneralCoarsened128x64x8_8x8:
    return "tiled-general-coarsened-128x64x8-8x8";
  case SgemmKernel::TiledGeneralCoarsened128x64x16_8x4:
    return "tiled-general-coarsened-128x64x16-8x4";
  case SgemmKernel::TiledGeneralCoarsened128x64x16_8x8:
    return "tiled-general-coarsened-128x64x16-8x8";
  case SgemmKernel::TiledGeneralCoarsened64x128x8_4x8:
    return "tiled-general-coarsened-64x128x8-4x8";
  case SgemmKernel::TiledGeneralCoarsened64x128x8_8x8:
    return "tiled-general-coarsened-64x128x8-8x8";
  case SgemmKernel::TiledGeneralCoarsened64x128x16_4x8:
    return "tiled-general-coarsened-64x128x16-4x8";
  case SgemmKernel::TiledGeneralCoarsened64x128x16_8x8:
    return "tiled-general-coarsened-64x128x16-8x8";
  }
  return "unknown";
}

SgemmKernel parse_kernel(const char *name) {
  if (std::strcmp(name, "naive") == 0) {
    return SgemmKernel::Naive;
  }
  if (std::strcmp(name, "naive-8x8") == 0) {
    return SgemmKernel::Naive8x8;
  }
  if (std::strcmp(name, "naive-16x16") == 0) {
    return SgemmKernel::Naive16x16;
  }
  if (std::strcmp(name, "naive-32x32") == 0) {
    return SgemmKernel::Naive32x32;
  }
  if (std::strcmp(name, "naive-64x16") == 0) {
    return SgemmKernel::Naive64x16;
  }
  if (std::strcmp(name, "naive-16x64") == 0) {
    return SgemmKernel::Naive16x64;
  }
  if (std::strcmp(name, "tiled-basic") == 0) {
    return SgemmKernel::TiledBasic;
  }
  if (std::strcmp(name, "tiled-basic-8") == 0) {
    return SgemmKernel::TiledBasic8;
  }
  if (std::strcmp(name, "tiled-basic-16") == 0) {
    return SgemmKernel::TiledBasic16;
  }
  if (std::strcmp(name, "tiled-basic-32") == 0) {
    return SgemmKernel::TiledBasic32;
  }
  if (std::strcmp(name, "tiled-general") == 0) {
    return SgemmKernel::TiledGeneral;
  }
  if (std::strcmp(name, "tiled-general-16x16x32") == 0) {
    return SgemmKernel::TiledGeneral16x16x32;
  }
  if (std::strcmp(name, "tiled-general-16x16x64") == 0) {
    return SgemmKernel::TiledGeneral16x16x64;
  }
  if (std::strcmp(name, "tiled-general-16x16x128") == 0) {
    return SgemmKernel::TiledGeneral16x16x128;
  }
  if (std::strcmp(name, "tiled-general-16x16x256") == 0) {
    return SgemmKernel::TiledGeneral16x16x256;
  }
  if (std::strcmp(name, "tiled-general-32x32x64") == 0) {
    return SgemmKernel::TiledGeneral32x32x64;
  }
  if (std::strcmp(name, "tiled-general-32x32x96") == 0) {
    return SgemmKernel::TiledGeneral32x32x96;
  }
  if (std::strcmp(name, "tiled-general-32x32x128") == 0) {
    return SgemmKernel::TiledGeneral32x32x128;
  }
  if (std::strcmp(name, "tiled-general-coarsened") == 0) {
    return SgemmKernel::TiledGeneralCoarsened;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x8-2x2") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x8_2x2;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x8-4x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x8_4x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x8-4x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x8_4x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x8-8x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x8_8x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x8-8x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x8_8x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x16-4x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x16_4x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x16-4x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x16_4x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x16-8x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x16_8x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x16-8x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x16_8x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x32-4x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x32_4x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x32-4x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x32_4x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x32-8x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x32_8x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x64x32-8x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x64x32_8x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-128x64x8-8x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened128x64x8_8x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-128x64x8-8x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened128x64x8_8x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-128x64x16-8x4") == 0) {
    return SgemmKernel::TiledGeneralCoarsened128x64x16_8x4;
  }
  if (std::strcmp(name, "tiled-general-coarsened-128x64x16-8x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened128x64x16_8x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x128x8-4x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x128x8_4x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x128x8-8x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x128x8_8x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x128x16-4x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x128x16_4x8;
  }
  if (std::strcmp(name, "tiled-general-coarsened-64x128x16-8x8") == 0) {
    return SgemmKernel::TiledGeneralCoarsened64x128x16_8x8;
  }
  throw std::invalid_argument(std::string("unknown kernel: ") + name);
}

void launch_sgemm(const SgemmKernel kernel, const int M, const int N,
                  const int K, const float alpha, const float *A,
                  const float *B, const float beta, float *C) {
  switch (kernel) {
  case SgemmKernel::Naive8x8:
    launch_sgemm_naive<8, 8>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::Naive16x16:
    launch_sgemm_naive<16, 16>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::Naive:
  case SgemmKernel::Naive32x32:
    launch_sgemm_naive<32, 32>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::Naive64x16:
    launch_sgemm_naive<64, 16>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::Naive16x64:
    launch_sgemm_naive<16, 64>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledBasic8:
    launch_sgemm_tiled_basic<8>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledBasic16:
    launch_sgemm_tiled_basic<16>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledBasic:
  case SgemmKernel::TiledBasic32:
    launch_sgemm_tiled_basic<32>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral:
  case SgemmKernel::TiledGeneral16x16x32:
    launch_sgemm_tiled_general<16, 16, 32>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral16x16x64:
    launch_sgemm_tiled_general<16, 16, 64>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral16x16x128:
    launch_sgemm_tiled_general<16, 16, 128>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral16x16x256:
    launch_sgemm_tiled_general<16, 16, 256>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral32x32x64:
    launch_sgemm_tiled_general<32, 32, 64>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral32x32x96:
    launch_sgemm_tiled_general<32, 32, 96>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneral32x32x128:
    launch_sgemm_tiled_general<32, 32, 128>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x8_2x2:
    launch_sgemm_tiled_general_coarsened<64, 64, 8, 2, 2>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened:
  case SgemmKernel::TiledGeneralCoarsened64x64x8_4x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 8, 4, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x8_4x8:
    launch_sgemm_tiled_general_coarsened<64, 64, 8, 4, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x8_8x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 8, 8, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x8_8x8:
    launch_sgemm_tiled_general_coarsened<64, 64, 8, 8, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x16_4x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 16, 4, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x16_4x8:
    launch_sgemm_tiled_general_coarsened<64, 64, 16, 4, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x16_8x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 16, 8, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x16_8x8:
    launch_sgemm_tiled_general_coarsened<64, 64, 16, 8, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x32_4x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 32, 4, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x32_4x8:
    launch_sgemm_tiled_general_coarsened<64, 64, 32, 4, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x32_8x4:
    launch_sgemm_tiled_general_coarsened<64, 64, 32, 8, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x64x32_8x8:
    launch_sgemm_tiled_general_coarsened<64, 64, 32, 8, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened128x64x8_8x4:
    launch_sgemm_tiled_general_coarsened<128, 64, 8, 8, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened128x64x8_8x8:
    launch_sgemm_tiled_general_coarsened<128, 64, 8, 8, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened128x64x16_8x4:
    launch_sgemm_tiled_general_coarsened<128, 64, 16, 8, 4>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened128x64x16_8x8:
    launch_sgemm_tiled_general_coarsened<128, 64, 16, 8, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x128x8_4x8:
    launch_sgemm_tiled_general_coarsened<64, 128, 8, 4, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x128x8_8x8:
    launch_sgemm_tiled_general_coarsened<64, 128, 8, 8, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x128x16_4x8:
    launch_sgemm_tiled_general_coarsened<64, 128, 16, 4, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::TiledGeneralCoarsened64x128x16_8x8:
    launch_sgemm_tiled_general_coarsened<64, 128, 16, 8, 8>(
        M, N, K, alpha, A, B, beta, C);
    break;
  }
}
