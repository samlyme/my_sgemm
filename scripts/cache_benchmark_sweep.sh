#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"

target=${TARGET:-build/sgemm_bench}
output=${OUTPUT:-benchmarks/sweep.csv}
peak_gflops=${PEAK_GFLOPS:-35580}
fixed_iterations=${SWEEP_ITERS:-}

read -r -a kernels <<< "${KERNELS:-naive-8x8 naive-16x16 naive-32x32 naive-64x16 naive-16x64 tiled-basic-8 tiled-basic-16 tiled-basic-32}"
read -r -a sizes <<< "${SIZES:-256 512 1024 2048 4096 8192}"

if [[ ! -x "$target" ]]; then
  echo "error: benchmark executable not found: $target" >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"
if [[ ! -f "$output" ]]; then
  printf '%s\n' \
    'kernel,size,iterations,check,time_ms,gflops,peak_percent,measured_at_utc,source_commit' \
    > "$output"
fi

for kernel in "${kernels[@]}"; do
  for size in "${sizes[@]}"; do
    if awk -F, -v kernel="$kernel" -v size="$size" \
      'NR > 1 && $1 == kernel && $2 == size { found = 1 } END { exit !found }' \
      "$output"; then
      echo "cached: $kernel at ${size}x${size}x${size}"
      continue
    fi

    if [[ -n "$fixed_iterations" ]]; then
      iterations=$fixed_iterations
    elif (( size <= 1024 )); then
      iterations=100
    elif (( size <= 2048 )); then
      iterations=50
    elif (( size <= 4096 )); then
      iterations=10
    else
      iterations=3
    fi

    echo "running: $kernel at ${size}x${size}x${size} ($iterations iterations)"
    result=$("$target" --kernel "$kernel" --m "$size" --n "$size" --k "$size" \
      --iters "$iterations")
    printf '%s\n' "$result"

    check=$(awk '/^check:/ { print $2 }' <<< "$result")
    time_ms=$(awk '/^time:/ { print $2 }' <<< "$result")
    gflops=$(awk '/^perf:/ { print $2 }' <<< "$result")
    peak_percent=$(awk -v perf="$gflops" -v peak="$peak_gflops" \
      'BEGIN { printf "%.2f", 100 * perf / peak }')
    measured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    source_commit=$(git rev-parse --short HEAD 2>/dev/null || printf unknown)

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$kernel" "$size" "$iterations" "$check" "$time_ms" "$gflops" \
      "$peak_percent" "$measured_at" "$source_commit" >> "$output"
  done
done

echo "cached benchmark results in $output"
