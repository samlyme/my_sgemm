#pragma once

#include <cuda_runtime.h>

// All matrices are row-major:
//   A is M x K, B is K x N, and C is M x N.
// Computes C = alpha * A * B + beta * C.
enum class SgemmKernel {
  Naive,
};

const char *kernel_name(SgemmKernel kernel);
SgemmKernel parse_kernel(const char *name);

void launch_sgemm(SgemmKernel kernel, int m, int n, int k, float alpha,
                  const float *a, const float *b, float beta, float *c,
                  cudaStream_t stream = nullptr);
