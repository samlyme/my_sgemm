# Cached benchmark results

`rtx3090-sm86.csv` contains 66 measurements: 11 distinct kernel
configurations at square sizes 256, 512, 1024, 2048, 4096, and 8192. Every
correctness check passed.

Detailed presentation-oriented profiles:

- [`pre-tiled-profile-rtx3090.md`](pre-tiled-profile-rtx3090.md): row strips,
  column strips, coalescing, and implicit cache reuse
- [`basic-tiled-profile-rtx3090.md`](basic-tiled-profile-rtx3090.md): explicit
  shared-memory reuse and the generalized tile design space

Measurement environment:

- GPU: NVIDIA GeForce RTX 3090 (SM 8.6, 24 GiB)
- Driver: 595.99.02
- Peak FP32 reference: 35,580 GFLOP/s
- Source revision: `6990b49`

Fastest result at each size:

| M=N=K | Kernel | GFLOP/s |
| ---: | --- | ---: |
| 256 | `tiled-basic` | 2,171.504 |
| 512 | `tiled-general-coarsened-64x64x16-4x4` | 6,606.452 |
| 1024 | `tiled-general-coarsened-128x64x8-8x4` | 9,365.213 |
| 2048 | `tiled-general-coarsened-128x64x8-8x4` | 10,953.760 |
| 4096 | `tiled-general-coarsened-128x64x8-8x4` | 12,722.783 |
| 8192 | `tiled-general-coarsened-128x64x8-8x4` | 12,966.487 |

Regenerate or resume the cache with:

```bash
scripts/cache_benchmark_sweep.sh
```

Existing `(kernel, size)` rows are skipped, so interrupted sweeps can be resumed.
Delete rows (or choose a different output path as the first argument) to rerun
specific measurements. `SIZES`, `PEAK_GFLOPS`, and `TARGET` can be overridden in
the environment.
