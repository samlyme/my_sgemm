#pragma once

#include <cuda_runtime.h>

// All matrices are row-major:
//   A is M x K, B is K x N, and C is M x N.
// Computes C = alpha * A * B + beta * C.
enum class SgemmKernel {
  Naive8x8,
  Naive16x16,
  Naive32x32,
  Naive64x16,
  Naive16x64,
  TiledBasic,
  TiledBasic8x8,
  TiledBasic16x16,
  TiledBasic32x32
  // TiledGeneral,
  // TiledGeneralCoarsened,
};

const char *kernel_name(SgemmKernel kernel);
SgemmKernel parse_kernel(const char *name);

void launch_sgemm(const SgemmKernel kernel, const int M, const int N,
                  const int K, const float alpha, const float *A,
                  const float *B, const float beta, float *C);
