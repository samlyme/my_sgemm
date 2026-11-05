# Basic and generalized tiled kernel profile: RTX 3090

Profiled on an NVIDIA GeForce RTX 3090 (SM 8.6, 24 GiB) with driver 595.99.02 and Nsight Compute 2025.2.1. The workload is `M=N=K=2048` with row-major matrices.

## Why tile explicitly?

The basic kernel computes a 32x32 output tile with a 32x32 thread block. For each 32-wide slice of `K`, the block loads 32x32 elements from each input into shared memory and reuses them across the block.

For a square tile of width `T`, one `K` slice performs `2*T^3` FLOPs after loading `2*T^2` floats. Ignoring `C`, its global-memory arithmetic intensity is therefore:

```text
AI = 2*T^3 / (2*T^2*4) = T/4 FLOP/byte
```

The 32-wide basic tile has an ideal input arithmetic intensity of 8 FLOP/byte. At `2048^3`, its measured L1 global-load traffic is only 2.16 GB, compared with 42.97 GB for the untiled naive kernel. Explicit tiling has reduced demand at the global-memory interface by about 20 times.

## Basic tiled profile

| Metric | `tiled-basic` (32x32x32) |
|---|---:|
| Standalone time | 5.850 ms |
| Standalone performance | 2,936.6 GFLOP/s |
| Profiled duration | 7.92 ms |
| DRAM reads | 885.02 MB |
| DRAM writes | 18.49 MB |
| DRAM bandwidth | 114.03 GB/s |
| DRAM utilization | 12.51% |
| L1 global-load traffic | 2.16 GB |
| L2 traffic | 2.18 GB |
| L2 hit rate | 59.43% |
| L1/TEX throughput utilization | 82.50% |
| Shared-load wavefronts | 402.65 million |
| Shared-store wavefronts | 20.98 million |
| Shared-memory bank conflicts | 4.20 million |
| Achieved occupancy | 66.63% |

Explicit reuse has moved the kernel away from the DRAM roof: DRAM is only 12.5% utilized. Data movement has not disappeared, however. Every FMA still requires an `A` and `B` operand to be read from shared memory. The L1/TEX/shared-memory path reaches about 82.5% utilization.

The fixed 32x32 design also requires 1,024 threads and 8 KiB of static shared memory per block. Only one block, containing 32 warps, can reside on an SM. The RTX 3090 supports 48 resident warps per SM, so theoretical occupancy is limited to 66.67%.

This creates the motivation for generalized tiling: the output dimensions and `K` slice should be independent tuning parameters rather than one fixed tile width.

## Generalized tile model

For an output tile `BM x BN` and a `K` slice of width `BK`:

```text
FLOPs per slice = 2*BM*BN*BK
input bytes     = 4*(BM*BK + BK*BN)
AI              = BM*BN / (2*(BM + BN)) FLOP/byte
```

`BK` controls shared-memory footprint, synchronization frequency, and loop granularity, but cancels out of the ideal global-memory arithmetic-intensity expression. `BM` and `BN` determine reuse and also determine the number of threads in this one-output-per-thread implementation.

## Generalized configuration comparison

| Configuration | Threads | Ideal input AI | Time | Performance | L1 global loads | DRAM reads | DRAM util. | L2 hit rate | Occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Basic 32x32x32 | 1,024 | 8.0 | 5.850 ms | 2,936.6 GFLOP/s | 2.16 GB | 885 MB | 12.51% | 59.43% | 66.66% |
| General 8x8x8 | 64 | 2.0 | 10.052 ms | 1,709.1 GFLOP/s | 8.60 GB | 877 MB | 7.74% | 84.40% | 66.48% |
| General 16x16x8 | 256 | 4.0 | 7.357 ms | 2,335.1 GFLOP/s | 4.31 GB | 1.31 GB | 15.20% | 62.79% | 98.98% |
| General 16x16x16 | 256 | 4.0 | 5.720 ms | 3,003.3 GFLOP/s | 4.24 GB | 2.15 GB | 30.64% | 49.28% | 98.83% |
| General 32x8x16 | 256 | 3.2 | 6.392 ms | 2,687.7 GFLOP/s | 5.22 GB | 1.10 GB | 14.12% | 79.07% | 98.86% |
| General 8x32x16 | 256 | 3.2 | 6.632 ms | 2,590.6 GFLOP/s | 5.36 GB | 3.91 GB | 49.08% | 27.14% | 98.89% |

Standalone times and performance were measured without a profiler. Traffic, cache, and occupancy metrics come from Nsight Compute replayed launches, so the two kinds of timing should not be compared directly.

## What the configurations demonstrate

### 32x32 versus 16x16: reuse/occupancy tradeoff

The 32x32 basic tile has twice the ideal input arithmetic intensity of a 16x16 tile and generates about half as much global-load traffic. The 16x16x16 configuration, however, uses only 256 threads per block and reaches approximately 99% occupancy. The two effects nearly balance: both kernels sustain about 3 TFLOP/s.

This is the clearest motivation for exposing `BM`, `BN`, and `BK`: maximizing reuse in isolation can reduce the number of resident warps available to hide latency.

### `BK` controls synchronization and loop overhead

The 16x16x8 and 16x16x16 configurations have the same output tile and essentially the same total global traffic. The 8-wide `K` slice requires twice as many tile iterations and therefore twice as many pairs of block-wide synchronizations. Increasing `BK` from 8 to 16 improves performance from 2.34 to 3.00 TFLOP/s without changing ideal arithmetic intensity.

### Very small blocks can also reduce occupancy

The 8x8x8 kernel has only two warps per block. The architectural limit of 16 resident blocks per SM caps it at 32 resident warps, or about 66.7% occupancy. It also has only 2 FLOP/byte of ideal input arithmetic intensity and incurs many more synchronization points. Smaller blocks are not automatically better for occupancy or performance.

### Tile orientation affects cache behavior

The 32x8x16 and 8x32x16 configurations have the same thread count and theoretical arithmetic intensity, but their cache behavior differs. The 32x8 tile obtains a 79% L2 hit rate and reads 1.10 GB from DRAM; the 8x32 tile obtains a 27% L2 hit rate and reads 3.91 GB. Grid traversal, warp layout, and which operand is reused by neighboring blocks make nominally equivalent tile areas behave differently.

The performance difference is modest because neither configuration is purely DRAM-bound, but it demonstrates why tile shape is an empirical tuning parameter.

## Resulting design lesson

The basic tiled kernel establishes explicit global-memory reuse. The generalized kernel is then motivated by three coupled constraints:

1. Larger output tiles increase global-memory arithmetic intensity.
2. More threads and shared memory per block can reduce occupancy.
3. `BK` and tile orientation change synchronization and cache behavior even when arithmetic intensity is unchanged.

The generalized kernel exposes these tradeoffs, but it retains one important limitation: one thread computes one output, so `BM*BN` cannot exceed 1,024. It also performs roughly two shared-memory operand reads for every FMA. These limitations motivate the next optimization—thread coarsening/register tiling—where each thread computes multiple outputs and reuses operands from registers.

## Reproduction

Standalone timings:

```bash
for kernel in \
  tiled-basic \
  tiled-general-8x8x8 \
  tiled-general-16x16x8 \
  tiled-general-16x16x16 \
  tiled-general-32x8x16 \
  tiled-general-8x32x16
do
  ./build/sgemm_bench --kernel "$kernel" \
    --m 2048 --n 2048 --k 2048 --iters 10
done
```

Detailed profile of one configuration:

```bash
../SGEMM_CUDA/.ncu/ncu \
  --set detailed \
  --launch-skip 6 \
  --launch-count 1 \
  ./build/sgemm_bench --kernel tiled-basic \
    --m 2048 --n 2048 --k 2048 --iters 1
```
