CUDA_NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(CUDA_NVCC),)
CUDA_NVCC := $(wildcard ../SGEMM_CUDA/.cuda/bin/nvcc)
endif
NVCC ?= $(if $(CUDA_NVCC),$(CUDA_NVCC),nvcc)

ARCH ?= 86
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/sgemm_bench
SOURCES := src/main.cu src/sgemm.cu
HEADERS := $(wildcard include/*.cuh)
CPPFLAGS ?= -Iinclude
NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr
GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
ARGS ?=

.DEFAULT_GOAL := run

.PHONY: all build run clean

all: run

build: $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(GENCODE) $(SOURCES) -o $@

clean:
	rm -f $(TARGET)
