NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
NVCC := $(if $(wildcard ../SGEMM_CUDA/.cuda/bin/nvcc),../SGEMM_CUDA/.cuda/bin/nvcc,nvcc)
endif

ARCH ?= 86
BUILD_DIR ?= build
NVCCFLAGS ?= -O3 -DNDEBUG -std=c++17
CPPFLAGS ?= -Iinclude
ARGS ?=
KERNELS ?= naive-8x8 naive-16x16 naive-32x32 naive-64x16 naive-16x64 \
	tiled-basic-8x8 tiled-basic-16x16 tiled-basic-32x32

TARGET := $(BUILD_DIR)/sgemm_bench
SOURCES := src/main.cu src/sgemm.cu
HEADERS := $(wildcard include/*.cuh)
GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

.DEFAULT_GOAL := benchmark

.PHONY: all build benchmark run clean rebuild

all: benchmark

build: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(GENCODE) $(SOURCES) -o $@

benchmark: build
	@set -e; \
	for kernel in $(KERNELS); do \
		echo "=== $$kernel ==="; \
		./$(TARGET) --kernel "$$kernel" $(ARGS); \
	done

run: benchmark

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
