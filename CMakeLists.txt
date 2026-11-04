cmake_minimum_required(VERSION 3.20)

# RTX 30-series default. Override for another GPU, for example:
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(my_sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(sgemm_bench
  src/main.cu
  src/sgemm.cu
)

target_include_directories(sgemm_bench PRIVATE include)
target_compile_options(sgemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
