NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
NVCC := $(if $(wildcard ../SGEMM_CUDA/.cuda/bin/nvcc),../SGEMM_CUDA/.cuda/bin/nvcc,nvcc)
endif

ARCH ?= 86
BUILD_DIR ?= build
NVCCFLAGS ?= -O3 -DNDEBUG -std=c++17
CPPFLAGS ?= -Iinclude
ARGS ?=

TARGET := $(BUILD_DIR)/sgemm_bench
SOURCES := src/main.cu src/sgemm.cu
HEADERS := $(wildcard include/*.cuh)
GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

.DEFAULT_GOAL := build

.PHONY: all build run clean rebuild

all: build

build: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(GENCODE) $(SOURCES) -o $@

run: build
	./$(TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
