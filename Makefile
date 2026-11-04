CUDA_NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(CUDA_NVCC),)
CUDA_NVCC := $(wildcard ../SGEMM_CUDA/.cuda/bin/nvcc)
endif
NVCC ?= $(if $(CUDA_NVCC),$(CUDA_NVCC),nvcc)

ARCH ?= 86
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/sgemm_bench
KERNELS := naive tiled-basic \
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
NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr
GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
ARGS ?=

.DEFAULT_GOAL := run

.PHONY: all build run large clean

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
		printf '%s|%s|%s|%s\n' "$$kernel" "$$check" "$$time" "$$perf" >> "$$summary"; \
	done; \
	printf '\n%-44s  %-7s  %12s  %16s\n' "KERNEL" "CHECK" "TIME" "PERFORMANCE"; \
	printf '%-44s  %-7s  %12s  %16s\n' \
		'--------------------------------------------' '-------' '------------' '----------------'; \
	awk -F '|' '{ printf "%-44s  %-7s  %12s  %16s\n", $$1, $$2, $$3, $$4 }' "$$summary"

large: $(TARGET)
	./$(TARGET) --m 8192 --n 8192 --k 8192 $(ARGS)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(GENCODE) $(SOURCES) -o $@

clean:
	rm -f $(TARGET)
