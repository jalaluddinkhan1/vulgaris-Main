cmake_minimum_required(VERSION 3.18)
project(vulgaris_kernels CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(CUDA REQUIRED)

include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

add_library(ssm_scan SHARED kernels/ssm_scan.cu)
target_link_libraries(ssm_scan ${Python3_LIBRARIES})
set_target_properties(ssm_scan PROPERTIES PREFIX "" SUFFIX ".so")

add_library(wavelet_conv SHARED kernels/wavelet.cu)
target_link_libraries(wavelet_conv ${Python3_LIBRARIES})
set_target_properties(wavelet_conv PROPERTIES PREFIX "" SUFFIX ".so")

set_property(TARGET ssm_scan wavelet_conv PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)
