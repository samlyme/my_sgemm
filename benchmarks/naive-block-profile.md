# Naive block-shape profile: 8x8, 16x16, and 32x32

> The repository does not contain a `naive-16x6` configuration. This report assumes
> `16x6` was a typo for `16x16`.

## Setup

- GPU: NVIDIA GeForce RTX 3090 (compute capability 8.6)
- Problem: `M=N=K=4096`
- Kernel: `sgemm_naive` in `src/sgemm.cu`
- Build resource usage: 40 registers/thread, no static shared memory
- Timing command:

  ```bash
  ./build/sgemm_bench --kernel <name> --m 4096 --n 4096 --k 4096 --iters 5
  ```

- Nsight Compute sections: `SpeedOfLight`, `MemoryWorkloadAnalysis`,
  `MemoryWorkloadAnalysis_Tables`, `SchedulerStats`, `WarpStateStats`, and
  `Occupancy`. One warm kernel launch was profiled after the correctness launch.

Profiler replay changes absolute kernel duration, so the normal benchmark supplies
latency and throughput below. The hardware-counter columns are used to explain the
relative behavior.

## Results

| Block | Threads/block | Warps/block | Time (ms) | GFLOP/s | Theoretical occupancy | L1 load hit rate | L2 load hit rate | DRAM traffic rate | Avg. useful bytes / 32-byte load sector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8x8 | 64 | 2 | 122.916 | 1118.149 | 66.67% | 83.51% | 39.20% | 346.34 GB/s | 9.60 |
| 16x16 | 256 | 8 | 55.755 | 2465.036 | 100.00% | 87.13% | 49.77% | 231.31 GB/s | 18.00 |
| 32x32 | 1024 | 32 | 59.660 | 2303.700 | 66.67% | 94.98% | 49.21% | 110.96 GB/s | 26.40 |

The 16x16 block is **2.20x faster than 8x8** and about **7% faster than
32x32** for this shape.

Additional profiler evidence:

| Block | Active warps / scheduler | Eligible warps / scheduler | Compute throughput | L1 load-miss sectors | L2 device read-miss sectors | Dominant reported stall |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8x8 | 7.96 | 0.36 | 75.83% | 1.771B | 1.076B | Long scoreboard, 27.1 cycles/instruction |
| 16x16 | 11.96 | 1.46 | 98.60% | 1.106B | 0.552B | L1 instruction queue, 23.5 cycles/instruction |
| 32x32 | 8.00 | 1.18 | 94.82% | 0.539B | 0.273B | L1 instruction queue, 18.3 cycles/instruction |

## Why the block shapes differ

### Warp layout and transaction efficiency

CUDA linearizes this 2D block with `threadIdx.x` changing fastest. Therefore:

- **8x8:** one warp covers four output rows, eight columns per row.
- **16x16:** one warp covers two output rows, sixteen columns per row.
- **32x32:** one warp covers one output row and thirty-two columns.

For each inner-loop iteration, every thread loads one `A[y, i]` and one
`B[i, x]`. Threads sharing an output row request the same A value, while B is
contiguous along x. The resulting sectors per warp iteration are approximately:

| Block | A sectors | B sectors | Total sectors | Useful requested bytes | Useful bytes/sector |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8x8 | 4 | 1 | 5 | `4*4 + 8*4 = 48` | 9.6 |
| 16x16 | 2 | 2 | 4 | `2*4 + 16*4 = 72` | 18.0 |
| 32x32 | 1 | 4 | 5 | `1*4 + 32*4 = 132` | 26.4 |

These values exactly match Nsight Compute's average useful bytes per 32-byte
sector. The 8x8 warp wastes the most transaction capacity because it touches four
rows while using only eight adjacent B values. The 32x32 warp has the cleanest
mapping: one broadcast A load plus a fully contiguous 32-float B load.

### Occupancy makes 16x16 the winner

On the RTX 3090, an SM can hold 48 warps and 16 blocks:

- **8x8** has only two warps/block. The 16-block limit caps it at 32 resident
  warps, or 66.7% occupancy.
- **16x16** has eight warps/block. Six blocks supply all 48 resident warps, or
  100% occupancy.
- **32x32** has 32 warps/block. Only one 1024-thread block fits, also giving 32
  resident warps and 66.7% occupancy.

The kernel has a dependent load/FMA loop and needs many warps to hide cache and
memory latency. The 16x16 launch has 11.96 active and 1.46 eligible warps per
scheduler, compared with 7.96/0.36 for 8x8 and 8.00/1.18 for 32x32. That extra
latency hiding outweighs 32x32's better memory coalescing and cache behavior.

### Why 8x8 falls far behind

A smaller x dimension creates more x-direction blocks. Because each block
re-reads the same A rows while computing a different set of output columns, the
number of A traversals scales approximately as `N / blockDim.x`. At 4096:

- 8x8 uses 512 x blocks,
- 16x16 uses 256,
- 32x32 uses 128.

Once the matrices no longer remain cache-resident, this appears directly in the
L2/DRAM traffic: the measured L2 device read misses are approximately
1.076B, 0.552B, and 0.273B sectors, nearly a 4:2:1 ratio. The implied bytes are
about 34.4 GB, 17.7 GB, and 8.7 GB, which closely match repeatedly streaming A
for 512, 256, and 128 column tiles.

The 8x8 case combines the worst conditions:

1. lowest useful bytes per memory sector,
2. four times the cache/DRAM traffic of 32x32,
3. only 66.7% occupancy to hide that latency.

Nsight consequently reports long-scoreboard stalls consuming 74.3% of its issue
interval, only 0.36 eligible warps per scheduler, and 75.83% compute throughput.

## Conclusion

- **16x16 is fastest** because it is the only shape that reaches full occupancy;
  it has enough coalescing and cache reuse while maximizing latency hiding.
- **32x32 is close behind** because its one-row-per-warp mapping has the best
  load efficiency and least DRAM traffic, but a single 1024-thread block per SM
  limits occupancy to 66.7%.
- **8x8 is slowest** because its multi-row warps use memory sectors poorly, its
  small x tile causes many more A reloads, and the 16-block limit prevents it
  from recovering with higher occupancy.

This also shows why block size alone is not a monotonic tuning parameter: 16x16
wins by balancing warp mapping, data reuse, and occupancy rather than maximizing
any one of them.
