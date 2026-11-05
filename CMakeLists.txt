cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(my_sgemm LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(sgemm_bench
  src/main.cu
  src/sgemm.cu
)

target_include_directories(sgemm_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(sgemm_bench PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(sgemm_bench PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD_REQUIRED ON
)
