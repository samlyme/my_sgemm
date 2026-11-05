#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"

target=${TARGET:-build/sgemm_bench}
output=${1:-benchmarks/rtx3090-sm86.csv}
peak_gflops=${PEAK_GFLOPS:-35580}

kernels=(
  naive
  tiled-basic
  tiled-general-8x8x8
  tiled-general-16x16x8
  tiled-general-16x16x16
  tiled-general-32x8x16
  tiled-general-8x32x16
  tiled-general-coarsened-32x32x8-2x2
  tiled-general-coarsened-64x64x8-4x4
  tiled-general-coarsened-64x64x16-4x4
  tiled-general-coarsened-128x64x8-8x4
)
sizes=(${SIZES:-256 512 1024 2048 4096 8192})

if [[ ! -x "$target" ]]; then
  make build
fi

mkdir -p "$(dirname "$output")"
if [[ ! -f "$output" ]]; then
  printf '%s\n' 'kernel,size,iterations,check,time_ms,gflops,peak_percent,measured_at_utc,source_commit' > "$output"
fi

for kernel in "${kernels[@]}"; do
  for size in "${sizes[@]}"; do
    if awk -F, -v kernel="$kernel" -v size="$size" \
      'NR > 1 && $1 == kernel && $2 == size { found = 1 } END { exit !found }' "$output"; then
      echo "cached: $kernel at ${size}x${size}x${size}"
      continue
    fi

    if (( size <= 1024 )); then
      iterations=100
    elif (( size <= 2048 )); then
      iterations=25
    elif (( size <= 4096 )); then
      iterations=5
    else
      iterations=1
    fi

    echo "running: $kernel at ${size}x${size}x${size} ($iterations iterations)"
    result=$("$target" --kernel "$kernel" --m "$size" --n "$size" --k "$size" --iters "$iterations")
    printf '%s\n' "$result"

    check=$(awk '/^check:/ { print $2 }' <<< "$result")
    time_ms=$(awk '/^time:/ { print $2 }' <<< "$result")
    gflops=$(awk '/^perf:/ { print $2 }' <<< "$result")
    peak_percent=$(awk -v perf="$gflops" -v peak="$peak_gflops" 'BEGIN { printf "%.2f", 100 * perf / peak }')
    measured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    source_commit=$(git rev-parse --short HEAD 2>/dev/null || printf unknown)

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$kernel" "$size" "$iterations" "$check" "$time_ms" "$gflops" \
      "$peak_percent" "$measured_at" "$source_commit" >> "$output"
  done
done

echo "cached benchmark results in $output"
