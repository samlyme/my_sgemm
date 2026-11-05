#pragma once

#include <cuda_runtime.h>

// All matrices are row-major:
//   A is M x K, B is K x N, and C is M x N.
// Computes C = alpha * A * B + beta * C.
enum class SgemmKernel {
  RowStrip,
  ColStrip,
  Naive,
  Naive32x1,
  Naive32x2,
  Naive32x4,
  Naive32x8,
  Naive32x16,
  Naive32x32,
  NaiveCoarsened,
  NaiveCoarsened64x64x2x2,
  NaiveCoarsened64x64x2x4,
  NaiveCoarsened64x64x4x2,
  NaiveCoarsened64x64x4x4,
  NaiveCoarsened64x64x4x8,
  NaiveCoarsened64x64x8x4,
  NaiveCoarsened64x64x8x8,
  TiledBasic,
  TiledGeneral,
  TiledGeneral8x8x8,
  TiledGeneral16x16x8,
  TiledGeneral16x16x16,
  TiledGeneral32x8x16,
  TiledGeneral8x32x16,
  TiledGeneralCoarsened,
  TiledGeneralCoarsened32x32x8x2x2,
  TiledGeneralCoarsened64x64x8x4x4,
  TiledGeneralCoarsened64x64x16x4x4,
  TiledGeneralCoarsened128x64x8x8x4,
};

const char *kernel_name(SgemmKernel kernel);
SgemmKernel parse_kernel(const char *name);

void launch_sgemm(const SgemmKernel kernel, const int M, const int N,
                  const int K, const float alpha, const float *A,
                  const float *B, const float beta, float *C);
