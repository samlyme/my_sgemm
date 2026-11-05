#include "sgemm.cuh"

#include <cstring>
#include <stdexcept>
#include <string>

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
} // namespace

const char *kernel_name(SgemmKernel kernel) {
  switch (kernel) {
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
  }
  return "unknown";
}

SgemmKernel parse_kernel(const char *name) {
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
  case SgemmKernel::Naive32x32:
    launch_sgemm_naive<32, 32>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::Naive64x16:
    launch_sgemm_naive<64, 16>(M, N, K, alpha, A, B, beta, C);
    break;
  case SgemmKernel::Naive16x64:
    launch_sgemm_naive<16, 64>(M, N, K, alpha, A, B, beta, C);
    break;
  }
}
