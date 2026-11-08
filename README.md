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

# Run the full family-aware sweep and generate latency/throughput plots
make suite

# Run only the benchmarks or regenerate plots from cached results
make sweep
make plots

./build/sgemm_bench --kernel cublas --m 512 --n 768 --k 256 --iters 50
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

The program checks the selected kernel against a CPU reference, then reports average latency and GFLOP/s. The `cublas` kernel uses `cublasSgemm` in pedantic math mode as an FP32 performance reference. The naive kernel is available with several two-dimensional thread-block shapes; plain `naive` aliases `naive-32x32`. The default `make` benchmark also includes several `tiled-general-BMxBNxBK` and `tiled-general-coarsened-BMxBNxBK-TMxTN` configurations.

The coarsened sweep tests `BK=8`, `16`, and `32` for every represented block-tile/thread-tile pair. It covers 64x64, 96x96, 128x64, 64x128, 128x128, 192x128, and 128x192 block tiles, with additional 2x2, 4x4, 4x8, and 8x4 thread-tile comparisons where the selected block shape supports them.

Ignoring the final C read/write, a coarsened block tile has theoretical FP32 arithmetic intensity `BM * BN / (2 * (BM + BN))` flop/byte. The larger block-tile variants extend the tested range toward the device ridge point:

| Block tile | Arithmetic intensity |
| --- | ---: |
| 64x64 | 16.0 flop/byte |
| 96x96 | 24.0 flop/byte |
| 128x128 | 32.0 flop/byte |
| 192x128 / 128x192 | 38.4 flop/byte |

The benchmark tables report performance as a percentage of peak FP32 throughput. `PEAK_GFLOPS` defaults to `35580` for an RTX 3090; override it for a different GPU.

The `sweep` target tests every kernel at `M=N=K` sizes `256 512 1024 2048 4096 8192`. Naive kernels use fewer iterations because they become expensive quickly; cuBLAS and all tiled families use substantially more iterations for stable timings. Cached rows are reused only when the kernel, size, and requested iteration count all match, so changing the iteration policy automatically refreshes the affected measurements.

The default iteration counts by size are:

| Family | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 100 | 50 | 20 | 10 | 3 | 1 |
| Tiled and cuBLAS | 1000 | 500 | 250 | 100 | 30 | 10 |

`make plots` uses `uv` and Matplotlib to select the fastest configuration in each family at every size. It writes throughput and latency plots in PNG and SVG formats, plus the selected rows in `benchmarks/plots/best_by_family.csv`.

Override sizes, iteration counts, output files, or peak throughput when needed:

```bash
make sweep SIZES="128 256 512" TILED_SWEEP_ITERS=200 \
  NAIVE_SWEEP_ITERS=20 CUBLAS_SWEEP_ITERS=500 \
  SWEEP_OUTPUT=benchmarks/test-sweep.csv PEAK_GFLOPS=29770

# SWEEP_ITERS overrides every family with one fixed count.
make sweep SWEEP_ITERS=10
```

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
