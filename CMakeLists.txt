cmake_minimum_required(VERSION 3.18)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# use the 11.2 CUDA in the system
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(TMotifGPU
        VERSION 0.1 
        DESCRIPTION "GPU implementation of temporal motif matching" 
        LANGUAGES CXX CUDA)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -export-dynamic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-lineinfo -arch=sm_86")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -arch=sm_86")

# for fast graph parsing
# find_package(OpenMP)

find_package(Boost REQUIRED COMPONENTS system iostreams)
include_directories(${Boost_INCLUDE_DIR} ${pybind11_INCLUDE_DIR})

include(FetchContent)

FetchContent_Declare(
        Range-v3
        GIT_REPOSITORY https://github.com/ericniebler/range-v3
        GIT_TAG 0.12.0
)

FetchContent_MakeAvailable(Range-v3)

# FetchContent_Declare(
#   mimalloc
#   GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
#   GIT_TAG v2.0.9
# )
# FetchContent_MakeAvailable(mimalloc)

# pybind11 =================================================================== #
 FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
#   GIT_TAG        2.6.1
  )
FetchContent_GetProperties(pybind11)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

if(NOT pybind11_POPULATED)
  message(STATUS "Populating pybind11")
  FetchContent_Populate(pybind11)
  set(pybind11_INCLUDE_DIR ${pybind11_SOURCE_DIR}/include)
  add_library(pybind11 IMPORTED INTERFACE)
  set_property(TARGET pybind11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${pybind11_INCLUDE_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

#---------------------------------------------------------------------------


include_directories(include ${pybind11_INCLUDE_DIR})

set(CMAKE_CUDA_ARCHITECTURES "86")


add_subdirectory(plugins)
add_subdirectory(src)