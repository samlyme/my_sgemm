#pragma once

#include <cuda_runtime.h>

// All matrices are row-major:
//   A is M x K, B is K x N, and C is M x N.
// Computes C = alpha * A * B + beta * C.
enum class SgemmKernel {
  Naive,
  TiledBasic,
  TiledGeneral,
  TiledGeneral8x8x8,
  TiledGeneral16x16x8,
  TiledGeneral16x16x16,
  TiledGeneral32x8x16,
  TiledGeneral8x32x16,
};

const char *kernel_name(SgemmKernel kernel);
SgemmKernel parse_kernel(const char *name);

void launch_sgemm(const SgemmKernel kernel, const int M, const int N,
                  const int K, const float alpha, const float *A,
                  const float *B, const float beta, float *C,
                  const cudaStream_t stream = nullptr);
