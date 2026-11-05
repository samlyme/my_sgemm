# Naive thread coarsening: RTX 3090

Profiled on an NVIDIA GeForce RTX 3090 (SM 8.6, 24 GiB) with driver 595.99.02 and Nsight Compute 2025.2.1. The workload is `M=N=K=2048` with row-major matrices.

Every configuration computes the same 64x64 output tile per block directly from global memory. A thread computes a `TM x TN` register tile, so increasing the coarsening factors reduces the number of threads and redundant input loads but increases per-thread work and register pressure.

## Results

Standalone timings were collected with 10 iterations, without a profiler. Register use and occupancy were collected from one profiled launch using Nsight Compute's `LaunchStats` and `Occupancy` sections.

| Thread tile | Thread block | Threads/block | Registers/thread | Resident blocks/SM | Theoretical occupancy | Achieved occupancy | Time | Performance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2x2` | 32x32 | 1,024 | 40 | 1 | 66.67% | 66.60% | 3.865 ms | 4,444.8 GFLOP/s |
| `2x4` | 16x32 | 512 | 40 | 3 | 100% | 92.92% | 3.296 ms | 5,211.8 GFLOP/s |
| `4x2` | 32x16 | 512 | 40 | 3 | 100% | 92.91% | 3.157 ms | 5,442.6 GFLOP/s |
| `4x4` | 16x16 | 256 | 48 | 5 | 83.33% | 71.35% | 3.515 ms | 4,887.5 GFLOP/s |
| `4x8` | 8x16 | 128 | 64 | 8 | 66.67% | 52.53% | 7.614 ms | 2,256.4 GFLOP/s |
| `8x4` | 16x8 | 128 | 72 | 7 | 58.33% | 48.47% | 2.428 ms | 7,076.0 GFLOP/s |
| `8x8` | 8x8 | 64 | 127 | 8 | 33.33% | 27.13% | 2.130 ms | **8,067.5 GFLOP/s** |

Plain `naive-coarsened` remains an alias for `naive-coarsened-64x64-4x4`.

## Interpretation

Coarsening trades occupancy for register reuse. Each thread loads `TM` values from `A` and `TN` values from `B` per inner-loop iteration, then performs `TM*TN` FMAs. Larger register tiles therefore perform more arithmetic per input loaded by that thread. The `8x8` configuration reaches the lowest occupancy but the highest performance because its additional reuse outweighs the loss of active warps for this kernel.

The `2x4` and `4x2` configurations have identical thread counts, register counts, and occupancy, but `4x2` is slightly faster. Similarly, `4x8` and `8x4` both use 128 threads, yet their performance differs by more than three times. `threadIdx.x` varies fastest within a warp, so changing `TN` changes the block's x dimension and how many output rows a warp spans. With `4x8`, the block is only eight threads wide and each warp spans four rows; with `8x4`, it is sixteen threads wide and each warp spans two rows. Coarsening direction therefore changes memory access grouping and warp-level reuse even when the number of outputs per thread is the same.

This experiment shows why occupancy is a constraint rather than an optimization target. The 100%-occupancy variants are not the fastest, and the 33%-occupancy `8x8` variant wins by doing much more useful work per loaded operand. Coarsening can continue only until register pressure, instruction-level parallelism, memory access geometry, or insufficient warps outweigh that reuse.

## Reproduction

Standalone timings:

```bash
for tile in 2x2 2x4 4x2 4x4 4x8 8x4 8x8; do
  ./build/sgemm_bench --kernel "naive-coarsened-64x64-$tile" \
    --m 2048 --n 2048 --k 2048 --iters 10
done
```

Register and occupancy profile:

```bash
for tile in 2x2 2x4 4x2 4x4 4x8 8x4 8x8; do
  ../SGEMM_CUDA/.ncu/ncu \
    --section LaunchStats \
    --section Occupancy \
    --launch-skip 6 \
    --launch-count 1 \
    ./build/sgemm_bench --kernel "naive-coarsened-64x64-$tile" \
      --m 2048 --n 2048 --k 2048 --iters 1
done
```

Nsight Compute instruments the selected launch, so the timing printed by the benchmark during profiling should not be used as a standalone performance result.
