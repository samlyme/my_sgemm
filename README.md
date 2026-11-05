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

# Sweep one kernel over square problem sizes from 256 through 8192
make sweep KERNEL=tiled-general-coarsened-64x64x8-4x4

# Build and run the large M=N=K=8192 benchmark
make large

./build/sgemm_bench --kernel row-strip --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel col-strip --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel naive --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel naive-32x8 --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel naive-coarsened --m 512 --n 768 --k 256 --iters 50
./build/sgemm_bench --kernel tiled-general-32x8x16 --iters 50
./build/sgemm_bench --kernel tiled-general-coarsened-64x64x8-4x4 --iters 50
./build/sgemm_bench --help
```

The program checks the selected kernel against a CPU reference, then reports average latency and GFLOP/s. The naive kernel is available with `32x1`, `32x2`, `32x4`, `32x8`, `32x16`, and `32x32` thread blocks; plain `naive` aliases `naive-32x32`. Keeping 32 threads in the x dimension preserves one horizontal output row per warp while varying the number of warps per block. The default `make` benchmark also includes several `tiled-general-BMxBNxBK` and `tiled-general-coarsened-BMxBNxBK-TMxTN` configurations.

The benchmark tables report performance as a percentage of peak FP32 throughput. `PEAK_GFLOPS` defaults to `35580` for an RTX 3090; override it for a different GPU.

The `sweep` target tests `M=N=K` at `256 512 1024 2048 4096 8192`, reducing the iteration count for larger problems. Override the peak throughput, sizes, or iteration count when needed:

```bash
make sweep KERNEL=naive PEAK_GFLOPS=29770 \
  SIZES="128 256 512" ARGS="--iters 10"
```

## Cached sweep

A full, resumable sweep over all distinct kernel configurations can be cached as
tracked CSV data:

```bash
scripts/cache_benchmark_sweep.sh
```

The checked-in RTX 3090 results and environment notes are under `benchmarks/`.
Existing kernel/size pairs are skipped when the script is run again.

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
