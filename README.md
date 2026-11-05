# my_sgemm

A small playground for writing and benchmarking single-precision CUDA matrix multiplication kernels.

Matrices use row-major storage and the benchmark computes:

```text
C[M, N] = A[M, K] * B[K, N]
```

The repository starts with horizontal row strips, vertical column strips, conventional two-dimensional thread blocks, and a register-coarsened naive kernel. These expose coalescing, cache behavior, and intra-thread reuse before the shared-memory tiled kernels introduce explicit inter-thread reuse.

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
# Build and run every kernel/configuration with M=N=K=1024
make

# Sweep every kernel over square problem sizes from 256 through 8192
# Results are cached in benchmarks/sweep.csv
make sweep

./build/sgemm_bench --kernel row-strip --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel col-strip --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel naive --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel naive-32x8 --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel naive-coarsened --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel naive-coarsened-64x64-4x8 --iters 50
./build/sgemm_bench --kernel tiled-general-16x16x64 --iters 50
./build/sgemm_bench --kernel tiled-general-coarsened-64x64x8-4x4 --iters 50
./build/sgemm_bench --help
```

The program checks the selected kernel against a CPU reference, then reports average latency and GFLOP/s. The naive kernel is available with `32x1`, `32x2`, `32x4`, `32x8`, `32x16`, and `32x32` thread blocks; plain `naive` aliases `naive-32x32`. Keeping 32 threads in the x dimension preserves one horizontal output row per warp while varying the number of warps per block. The `naive-coarsened-64x64-TMxTN` variants keep a 64x64 output tile while varying each thread's register tile; plain `naive-coarsened` aliases the `4x4` variant. The default `make` benchmark also includes several `tiled-general-BMxBNxBK` and `tiled-general-coarsened-BMxBNxBK-TMxTN` configurations.

The benchmark tables report performance as a percentage of peak FP32 throughput. `PEAK_GFLOPS` defaults to `35580` for an RTX 3090; override it for a different GPU.

The `sweep` target tests every kernel at `M=N=K` sizes `256 512 1024 2048 4096 8192`. It reduces the iteration count for larger problems and caches each completed kernel/size pair, so interrupted sweeps can be resumed by running `make sweep` again.

Override the sizes, fixed iteration count, output file, or peak throughput when needed:

```bash
make sweep SIZES="128 256 512" SWEEP_ITERS=10 \
  SWEEP_OUTPUT=benchmarks/test-sweep.csv PEAK_GFLOPS=29770
```

Existing kernel/size pairs in the output CSV are skipped.

## Add a kernel

1. Add an entry to `SgemmKernel` in `include/sgemm.cuh`.
2. Implement the kernel in `src/sgemm.cu` (or add another `.cu` file to `CMakeLists.txt`).
3. Extend `kernel_name`, `parse_kernel`, and `launch_sgemm`.
4. Add its command-line name to `KERNELS` in `Makefile`.
5. Compare it using the same matrix dimensions and iteration count.

Useful profiling commands:

```bash
ncu ./build/sgemm_bench --kernel tiled-basic --m 2048 --n 2048 --k 2048
nsys profile ./build/sgemm_bench --kernel tiled-basic
```
