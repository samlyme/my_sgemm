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

    C[y * K + x] = alpha * sum + beta * C[y * K + x];
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
  }
}
