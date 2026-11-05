# Block-tiled and register-coarsened profile: RTX 3090

Profiled on an NVIDIA GeForce RTX 3090 (SM 8.6, 24 GiB) with driver 595.99.02 and Nsight Compute 2025.2.1. The workload is `M=N=K=2048` with row-major matrices.

The `tiled-general-coarsened-BMxBNxBK-TMxTN` kernel combines two forms of reuse:

1. A block cooperatively loads `BM x BK` and `BK x BN` input tiles into shared memory.
2. Each thread accumulates a `TM x TN` output tile in registers.

This decouples output-tile size from thread-block size. All block-tiled/coarsened configurations profiled below use 256 threads, but they compute output tiles ranging from 32x32 to 128x64.

## Standalone performance

These timings were collected with 10 iterations, without a profiler.

| Kernel | Output tile | K tile | Thread tile | Threads | Time | Performance |
|---|---:|---:|---:|---:|---:|---:|
| `tiled-basic` | 32x32 | 32 | 1x1 | 1,024 | 5.867 ms | 2,928.0 GFLOP/s |
| `tiled-general-16x16x16` | 16x16 | 16 | 1x1 | 256 | 5.861 ms | 2,931.4 GFLOP/s |
| `naive-coarsened-64x64-8x8` | 64x64 | global | 8x8 | 64 | 2.132 ms | 8,059.1 GFLOP/s |
| `tiled-general-coarsened-32x32x8-2x2` | 32x32 | 8 | 2x2 | 256 | 2.402 ms | 7,152.0 GFLOP/s |
| `tiled-general-coarsened-64x64x8-4x4` | 64x64 | 8 | 4x4 | 256 | 1.956 ms | 8,785.4 GFLOP/s |
| `tiled-general-coarsened-64x64x16-4x4` | 64x64 | 16 | 4x4 | 256 | 1.590 ms | 10,804.5 GFLOP/s |
| `tiled-general-coarsened-128x64x8-8x4` | 128x64 | 8 | 8x4 | 256 | 1.573 ms | **10,919.8 GFLOP/s** |

The fastest block-tiled/coarsened configuration is about 3.7 times faster than the basic tiled kernel and 1.35 times faster than the best profiled naive-coarsened configuration.

## Profiler comparison

Nsight Compute replays and instruments kernels, so profiled durations are slower than standalone timings. Utilization values below come from the detailed profile of one selected launch.

| Configuration | Ideal input AI | Registers/thread | Shared memory/block | Theoretical occupancy | Achieved occupancy | Profiled duration | Compute throughput | DRAM throughput | L1/TEX throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32x32x8, 2x2 | 8.0 FLOP/B | 40 | 2 KiB | 100% | 96.68% | 3.28 ms | 73.13% | 23.55% | 74.01% |
| 64x64x8, 4x4 | 16.0 FLOP/B | 56 | 4 KiB | 66.67% | 61.32% | 2.56 ms | 48.83% | 5.28% | 55.89% |
| 64x64x16, 4x4 | 16.0 FLOP/B | 120 | 8 KiB | 33.33% | 32.57% | 2.17 ms | 50.67% | 8.72% | 56.19% |
| 128x64x8, 8x4 | 21.33 FLOP/B | 80 | 6 KiB | 50% | 44.97% | 2.07 ms | 48.32% | 6.42% | 48.34% |

For a `BM x BN` output tile, ideal global input arithmetic intensity remains:

```text
AI = BM*BN / (2*(BM + BN)) FLOP/byte
```

Register tiling allows `BM*BN` to exceed the number of threads. The 64x64 and 128x64 blocks therefore obtain substantially more global-memory reuse than the one-output-per-thread kernels without requiring invalid 4,096- or 8,192-thread blocks.

## Interpretation

### Occupancy is no longer the primary target

The 32x32x8 configuration reaches almost 97% achieved occupancy, yet it is the slowest coarsened tiled variant. The 64x64x16 configuration reaches only about 33% occupancy and is over 50% faster. Larger block and thread tiles increase reuse and instruction-level parallelism enough to compensate for fewer resident warps.

The cost is register pressure. Moving from `BK=8` to `BK=16` for the 64x64, 4x4 design raises register allocation from 56 to 120 registers per thread and cuts theoretical occupancy from 66.67% to 33.33%. Despite that, performance rises by about 23% because the wider K tile halves the number of tile iterations and synchronization pairs.

### Larger output tiles improve global reuse

The 32x32 tile has the same ideal 8 FLOP/byte input intensity as the basic tiled kernel. Increasing it to 64x64 doubles ideal intensity to 16 FLOP/byte, while 128x64 raises it to about 21.3 FLOP/byte. DRAM throughput remains below 9% for the three larger configurations, indicating that off-chip bandwidth is not their immediate limit.

The 128x64x8 configuration narrowly wins this comparison. Its larger output tile provides more reuse than 64x64x8, while its 80-register allocation retains more occupancy than the 120-register 64x64x16 kernel. Their similar performance illustrates the tuning tradeoff between output-tile reuse, K-tile synchronization frequency, register pressure, and active warps.

### Shared memory plus register tiling beats either step alone

The one-output-per-thread tiled kernels sustain about 2.9 TFLOP/s because every output has only one accumulator and the output tile cannot exceed 1,024 elements. Naive thread coarsening reaches about 8.1 TFLOP/s through register accumulation but continues loading operands directly from global memory. Combining cooperative shared-memory loading with per-thread output tiles reaches about 10.9 TFLOP/s in this implementation.

This remains far below the RTX 3090's peak FP32 throughput, but it establishes the central block-tiled SGEMM design: shared memory provides inter-thread input reuse, while register tiles provide larger output tiles and more work per thread.

## Reproduction

Standalone timings:

```bash
for kernel in \
  tiled-general-coarsened-32x32x8-2x2 \
  tiled-general-coarsened-64x64x8-4x4 \
  tiled-general-coarsened-64x64x16-4x4 \
  tiled-general-coarsened-128x64x8-8x4; do
  ./build/sgemm_bench --kernel "$kernel" \
    --m 2048 --n 2048 --k 2048 --iters 10
done
```

Detailed profile:

```bash
for kernel in \
  tiled-general-coarsened-32x32x8-2x2 \
  tiled-general-coarsened-64x64x8-4x4 \
  tiled-general-coarsened-64x64x16-4x4 \
  tiled-general-coarsened-128x64x8-8x4; do
  ../SGEMM_CUDA/.ncu/ncu \
    --set detailed \
    --launch-skip 6 \
    --launch-count 1 \
    ./build/sgemm_bench --kernel "$kernel" \
      --m 2048 --n 2048 --k 2048 --iters 1
done
```
