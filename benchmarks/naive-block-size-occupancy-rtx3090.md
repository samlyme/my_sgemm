# Naive block-size occupancy: RTX 3090

Profiled on an NVIDIA GeForce RTX 3090 (SM 8.6, 24 GiB) with driver 595.99.02 and Nsight Compute 2025.2.1. The workload is `M=N=K=2048` with row-major matrices.

All configurations use 32 threads in the x dimension, so each warp computes one horizontal output row with coalesced `B` loads and `C` stores. Only the number of warps in each block changes.

## Results

Standalone timings were collected with 10 iterations, without a profiler. Occupancy was collected from one profiled launch using Nsight Compute's `Occupancy` section.

| Kernel | Threads/block | Warps/block | Resident blocks/SM | Theoretical occupancy | Achieved occupancy | Time | Performance |
|---|---:|---:|---:|---:|---:|---:|---:|
| `naive-32x1` | 32 | 1 | 16 | 33.33% | 33.21% | 13.336 ms | 1,288.3 GFLOP/s |
| `naive-32x2` | 64 | 2 | 16 | 66.67% | 66.24% | 8.705 ms | 1,973.5 GFLOP/s |
| `naive-32x4` | 128 | 4 | 12 | 100% | 98.76% | 7.663 ms | 2,241.9 GFLOP/s |
| `naive-32x8` | 256 | 8 | 6 | 100% | 98.80% | 7.135 ms | 2,407.9 GFLOP/s |
| `naive-32x16` | 512 | 16 | 3 | 100% | 98.96% | 6.853 ms | 2,507.0 GFLOP/s |
| `naive-32x32` | 1,024 | 32 | 1 | 66.67% | 66.47% | 7.308 ms | 2,350.7 GFLOP/s |

Plain `naive` remains an alias for `naive-32x32`.

## Interpretation

The RTX 3090 can hold at most 48 warps and 16 blocks per SM. Occupancy therefore is not monotonic with block size:

- A 32-thread block has only one warp. The 16-block limit permits 16 resident warps, or 33.33% occupancy.
- A 64-thread block reaches 32 resident warps before hitting the same block limit, giving 66.67% occupancy.
- Blocks with 128 through 512 threads can fill all 48 warp slots: 12x4, 6x8, or 3x16 resident block-warps.
- A 1,024-thread block has 32 warps, but only one such block fits. Occupancy falls back to 66.67%.

Raising occupancy from 33% to 100% improves this latency-sensitive naive kernel substantially, but occupancy alone does not determine performance. The three 100%-occupancy configurations have different timings. Larger blocks group more adjacent output rows together and can improve temporal locality for the shared `B` rows; scheduling and cache effects therefore still matter. Here, `32x16` is the fastest configuration, while `32x32` gives up occupancy and regresses.

The main lesson is that both very small and very large blocks can reduce occupancy. The best block size is large enough to avoid the resident-block limit but small enough to allow enough blocks and warps to coexist on each SM.

## Reproduction

Standalone timings:

```bash
for kernel in \
  naive-32x1 naive-32x2 naive-32x4 \
  naive-32x8 naive-32x16 naive-32x32; do
  ./build/sgemm_bench --kernel "$kernel" \
    --m 2048 --n 2048 --k 2048 --iters 10
done
```

Occupancy profile:

```bash
for kernel in \
  naive-32x1 naive-32x2 naive-32x4 \
  naive-32x8 naive-32x16 naive-32x32; do
  ../SGEMM_CUDA/.ncu/ncu \
    --section Occupancy \
    --launch-skip 6 \
    --launch-count 1 \
    ./build/sgemm_bench --kernel "$kernel" \
      --m 2048 --n 2048 --k 2048 --iters 1
done
```

Nsight Compute replays or instruments the selected launch, so the timing printed by the benchmark during profiling should not be used as a standalone performance result.
