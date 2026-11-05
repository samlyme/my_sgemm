#pragma once

#include <cuda_runtime.h>

// All matrices are row-major:
//   A is M x K, B is K x N, and C is M x N.
// Computes C = alpha * A * B + beta * C.
enum class SgemmKernel {
  CuBLAS,
  Naive,
  Naive8x8,
  Naive16x16,
  Naive32x32,
  Naive64x16,
  Naive16x64,
  TiledBasic,
  TiledBasic8,
  TiledBasic16,
  TiledBasic32,
  TiledGeneral,
  TiledGeneral16x16x32,
  TiledGeneral16x16x64,
  TiledGeneral16x16x128,
  TiledGeneral16x16x256,
  TiledGeneral32x32x64,
  TiledGeneral32x32x96,
  TiledGeneral32x32x128,
  TiledGeneralCoarsened,
  TiledGeneralCoarsened64x64x8_2x2,
  TiledGeneralCoarsened64x64x8_4x4,
  TiledGeneralCoarsened64x64x8_4x8,
  TiledGeneralCoarsened64x64x8_8x4,
  TiledGeneralCoarsened64x64x8_8x8,
  TiledGeneralCoarsened64x64x16_4x4,
  TiledGeneralCoarsened64x64x16_4x8,
  TiledGeneralCoarsened64x64x16_8x4,
  TiledGeneralCoarsened64x64x16_8x8,
  TiledGeneralCoarsened64x64x32_4x4,
  TiledGeneralCoarsened64x64x32_4x8,
  TiledGeneralCoarsened64x64x32_8x4,
  TiledGeneralCoarsened64x64x32_8x8,
  TiledGeneralCoarsened128x64x8_8x4,
  TiledGeneralCoarsened128x64x8_8x8,
  TiledGeneralCoarsened128x64x16_8x4,
  TiledGeneralCoarsened128x64x16_8x8,
  TiledGeneralCoarsened64x128x8_4x8,
  TiledGeneralCoarsened64x128x8_8x8,
  TiledGeneralCoarsened64x128x16_4x8,
  TiledGeneralCoarsened64x128x16_8x8,
};

const char *kernel_name(SgemmKernel kernel);
SgemmKernel parse_kernel(const char *name);

void launch_sgemm(const SgemmKernel kernel, const int M, const int N,
                  const int K, const float alpha, const float *A,
                  const float *B, const float beta, float *C);
