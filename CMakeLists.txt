cmake_minimum_required(VERSION 3.20)

# RTX 30-series default. Override for another GPU, for example:
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(my_sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(sgemm_bench
  src/main.cu
  src/sgemm.cu
)

target_include_directories(sgemm_bench PRIVATE include)
