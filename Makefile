NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
NVCC := $(if $(wildcard ../SGEMM_CUDA/.cuda/bin/nvcc),../SGEMM_CUDA/.cuda/bin/nvcc,nvcc)
endif

ARCH ?= 86
BUILD_DIR ?= build
NVCCFLAGS ?= -O3 -DNDEBUG -std=c++17
CPPFLAGS ?= -Iinclude
CUDA_LIB_DIR := $(abspath $(dir $(shell readlink -f $(NVCC)))/../lib64)
LDFLAGS ?= -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)
LDLIBS ?= -lcublas
ARGS ?=
KERNELS ?= cublas naive-8x8 naive-16x16 naive-32x32 naive-64x16 naive-16x64 \
	tiled-basic-8 tiled-basic-16 tiled-basic-32 \
	tiled-general-16x16x32 tiled-general-16x16x64 \
	tiled-general-16x16x128 tiled-general-16x16x256 \
	tiled-general-32x32x64 \
	tiled-general-32x32x96 tiled-general-32x32x128 \
	tiled-general-coarsened-64x64x8-2x2 \
	tiled-general-coarsened-64x64x8-4x4 \
	tiled-general-coarsened-64x64x8-4x8 \
	tiled-general-coarsened-64x64x8-8x4 \
	tiled-general-coarsened-64x64x8-8x8 \
	tiled-general-coarsened-64x64x16-4x4 \
	tiled-general-coarsened-64x64x16-4x8 \
	tiled-general-coarsened-64x64x16-8x4 \
	tiled-general-coarsened-64x64x16-8x8 \
	tiled-general-coarsened-64x64x32-4x4 \
	tiled-general-coarsened-64x64x32-4x8 \
	tiled-general-coarsened-64x64x32-8x4 \
	tiled-general-coarsened-64x64x32-8x8 \
	tiled-general-coarsened-128x64x8-8x4 \
	tiled-general-coarsened-128x64x8-8x8 \
	tiled-general-coarsened-128x64x16-8x4 \
	tiled-general-coarsened-128x64x16-8x8 \
	tiled-general-coarsened-64x128x8-4x8 \
	tiled-general-coarsened-64x128x8-8x8 \
	tiled-general-coarsened-64x128x16-4x8 \
	tiled-general-coarsened-64x128x16-8x8 \
	tiled-general-coarsened-96x96x32-8x8 \
	tiled-general-coarsened-128x128x32-8x8 \
	tiled-general-coarsened-192x128x32-8x8 \
	tiled-general-coarsened-128x192x32-8x8
SIZES ?= 256 512 1024 2048 4096 8192
SWEEP_OUTPUT ?= benchmarks/sweep.csv
SWEEP_ITERS ?=
NAIVE_SWEEP_ITERS ?=
TILED_SWEEP_ITERS ?=
CUBLAS_SWEEP_ITERS ?=
PLOT_OUTPUT_DIR ?= benchmarks/plots
PEAK_GFLOPS ?= 35580

TARGET := $(BUILD_DIR)/sgemm_bench
SOURCES := src/main.cu src/sgemm.cu
HEADERS := $(wildcard include/*.cuh)
GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

.DEFAULT_GOAL := benchmark

.PHONY: all build benchmark sweep plots suite run clean rebuild

all: benchmark

build: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(GENCODE) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

benchmark: build
	@set -e; \
	summary=$$(mktemp); \
	output=$$(mktemp); \
	trap 'rm -f "$$summary" "$$output"' EXIT; \
	for kernel in $(KERNELS); do \
		echo "=== $$kernel ==="; \
		if ! ./$(TARGET) --kernel "$$kernel" $(ARGS) > "$$output" 2>&1; then \
			cat "$$output"; \
			exit 1; \
		fi; \
		cat "$$output"; \
		status=$$(awk '/^check:/ { print $$2 }' "$$output"); \
		time_ms=$$(awk '/^time:/ { print $$2 }' "$$output"); \
		gflops=$$(awk '/^perf:/ { print $$2 }' "$$output"); \
		printf '%s\t%s\t%s\t%s\n' "$$kernel" "$$status" "$$time_ms" "$$gflops" >> "$$summary"; \
	done; \
	printf '\n%-28s  %-6s  %12s  %14s\n' "Kernel" "Check" "Time (ms)" "GFLOP/s"; \
	printf '%-28s  %-6s  %12s  %14s\n' "----------------------------" "------" "------------" "--------------"; \
	awk -F '\t' '{ printf "%-28s  %-6s  %12.3f  %14.3f\n", $$1, $$2, $$3, $$4 }' "$$summary"

sweep: build
	@KERNELS='$(KERNELS)' \
	SIZES='$(SIZES)' \
	TARGET='$(TARGET)' \
	OUTPUT='$(SWEEP_OUTPUT)' \
	SWEEP_ITERS='$(SWEEP_ITERS)' \
	NAIVE_SWEEP_ITERS='$(NAIVE_SWEEP_ITERS)' \
	TILED_SWEEP_ITERS='$(TILED_SWEEP_ITERS)' \
	CUBLAS_SWEEP_ITERS='$(CUBLAS_SWEEP_ITERS)' \
	PEAK_GFLOPS='$(PEAK_GFLOPS)' \
	./scripts/cache_benchmark_sweep.sh

plots:
	uv run scripts/plot_benchmark_sweep.py \
		--input '$(abspath $(SWEEP_OUTPUT))' \
		--output-dir '$(abspath $(PLOT_OUTPUT_DIR))'

suite: sweep plots

run: benchmark

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
