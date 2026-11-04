# my_sgemm

A small playground for writing and benchmarking single-precision CUDA matrix multiplication kernels.

Matrices use row-major storage and the benchmark computes:

```text
C[M, N] = A[M, K] * B[K, N]
```

The repository starts with a simple per-output-element kernel and a shared-memory tiled kernel. They are intentionally straightforward baselines to improve upon.

## Requirements

- NVIDIA GPU
- CUDA Toolkit (`nvcc`)
- CMake 3.20+

The default CUDA architecture is `86` (RTX 30-series). Override it at configure time for another GPU.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For a different architecture:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

## Run

```bash
# Defaults: naive kernel, M=N=K=8192, 100 timed iterations
./build/sgemm_bench

./build/sgemm_bench --kernel naive --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --help
```

The program checks the selected kernel against a CPU reference, then reports average latency and GFLOP/s.

## Add a kernel

1. Add an entry to `SgemmKernel` in `include/sgemm.cuh`.
2. Implement the kernel in `src/sgemm.cu` (or add another `.cu` file to `CMakeLists.txt`).
3. Extend `kernel_name`, `parse_kernel`, and `launch_sgemm`.
4. Compare it using the same matrix dimensions and iteration count.

Useful profiling commands:

```bash
ncu ./build/sgemm_bench --kernel tiled --m 2048 --n 2048 --k 2048
nsys profile ./build/sgemm_bench --kernel tiled
```
