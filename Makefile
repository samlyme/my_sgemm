CUDA_NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(CUDA_NVCC),)
CUDA_NVCC := $(wildcard ../SGEMM_CUDA/.cuda/bin/nvcc)
endif
NVCC ?= $(if $(CUDA_NVCC),$(CUDA_NVCC),nvcc)

ARCH ?= 86
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/sgemm_bench
KERNELS := row-strip col-strip \
	naive-32x1 naive-32x2 naive-32x4 naive-32x8 naive-32x16 naive-32x32 \
	naive-coarsened tiled-basic \
	tiled-general-8x8x8 \
	tiled-general-16x16x8 \
	tiled-general-16x16x16 \
	tiled-general-32x8x16 \
	tiled-general-8x32x16 \
	tiled-general-coarsened-32x32x8-2x2 \
	tiled-general-coarsened-64x64x8-4x4 \
	tiled-general-coarsened-64x64x16-4x4 \
	tiled-general-coarsened-128x64x8-8x4
SOURCES := src/main.cu src/sgemm.cu
HEADERS := $(wildcard include/*.cuh)
CPPFLAGS ?= -Iinclude
NVCCFLAGS ?= -O3 -DNDEBUG
GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
ARGS ?=
KERNEL ?=
SIZES ?= 256 512 1024 2048 4096 8192
PEAK_GFLOPS ?= 35580

.DEFAULT_GOAL := run

.PHONY: all build run sweep large clean

all: run

build: $(TARGET)

run: $(TARGET)
	@set -e; \
	summary=$$(mktemp); \
	trap 'rm -f "$$summary"' EXIT; \
	for kernel in $(KERNELS); do \
		echo "=== $$kernel ==="; \
		output=$$(./$(TARGET) --kernel $$kernel $(ARGS)); \
		printf '%s\n' "$$output"; \
		check=$$(printf '%s\n' "$$output" | awk '/^check:/ { print $$2 }'); \
		time=$$(printf '%s\n' "$$output" | awk '/^time:/ { print $$2 " " $$3 }'); \
		perf=$$(printf '%s\n' "$$output" | awk '/^perf:/ { print $$2 " " $$3 }'); \
		peak=$$(awk -v perf="$$perf" -v peak="$(PEAK_GFLOPS)" \
			'BEGIN { printf "%.1f%%", 100 * perf / peak }'); \
		printf '%s|%s|%s|%s|%s\n' "$$kernel" "$$check" "$$time" "$$perf" "$$peak" >> "$$summary"; \
	done; \
	printf '\nPeak FP32 throughput: %s GFLOP/s\n' "$(PEAK_GFLOPS)"; \
	printf '%-44s  %-7s  %12s  %16s  %9s\n' \
		"KERNEL" "CHECK" "TIME" "PERFORMANCE" "% OF PEAK"; \
	printf '%-44s  %-7s  %12s  %16s  %9s\n' \
		'--------------------------------------------' '-------' '------------' '----------------' '---------'; \
	awk -F '|' '{ printf "%-44s  %-7s  %12s  %16s  %9s\n", $$1, $$2, $$3, $$4, $$5 }' "$$summary"

sweep: $(TARGET)
	@if [ -z "$(KERNEL)" ]; then \
		echo "error: specify a kernel, for example:"; \
		echo "  make sweep KERNEL=tiled-general-coarsened-64x64x8-4x4"; \
		exit 2; \
	fi
	@set -e; \
	summary=$$(mktemp); \
	trap 'rm -f "$$summary"' EXIT; \
	for size in $(SIZES); do \
		if [ "$$size" -le 1024 ]; then iters=100; \
		elif [ "$$size" -le 2048 ]; then iters=25; \
		elif [ "$$size" -le 4096 ]; then iters=5; \
		else iters=1; fi; \
		echo "=== $(KERNEL): $$size x $$size x $$size ==="; \
		output=$$(./$(TARGET) --kernel "$(KERNEL)" \
			--m "$$size" --n "$$size" --k "$$size" --iters "$$iters" $(ARGS)); \
		printf '%s\n' "$$output"; \
		check=$$(printf '%s\n' "$$output" | awk '/^check:/ { print $$2 }'); \
		time=$$(printf '%s\n' "$$output" | awk '/^time:/ { print $$2 " " $$3 }'); \
		perf=$$(printf '%s\n' "$$output" | awk '/^perf:/ { print $$2 " " $$3 }'); \
		peak=$$(awk -v perf="$$perf" -v peak="$(PEAK_GFLOPS)" \
			'BEGIN { printf "%.1f%%", 100 * perf / peak }'); \
		printf '%s|%s|%s|%s|%s\n' "$$size" "$$check" "$$time" "$$perf" "$$peak" >> "$$summary"; \
	done; \
	printf '\nKernel: %s\n' "$(KERNEL)"; \
	printf 'Peak FP32 throughput: %s GFLOP/s\n' "$(PEAK_GFLOPS)"; \
	printf '%10s  %-7s  %12s  %16s  %9s\n' \
		"M=N=K" "CHECK" "TIME" "PERFORMANCE" "% OF PEAK"; \
	printf '%10s  %-7s  %12s  %16s  %9s\n' \
		'----------' '-------' '------------' '----------------' '---------'; \
	awk -F '|' '{ printf "%10s  %-7s  %12s  %16s  %9s\n", $$1, $$2, $$3, $$4, $$5 }' "$$summary"

large: $(TARGET)
	./$(TARGET) --m 8192 --n 8192 --k 8192 $(ARGS)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(GENCODE) $(SOURCES) -o $@

clean:
	rm -f $(TARGET)
