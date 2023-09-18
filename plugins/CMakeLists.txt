
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

add_library(Baseline SHARED GPUWorkerBaseline.cu)
add_library(U0 SHARED GPUWorkerU0.cu)
add_library(U1 SHARED GPUWorkerU1.cu)
add_library(U1B SHARED GPUWorkerU1B.cu)
add_library(U2 SHARED GPUWorkerU2.cu)
add_library(U3 SHARED GPUWorkerU3.cu)
add_library(U3B SHARED GPUWorkerU3B.cu)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)