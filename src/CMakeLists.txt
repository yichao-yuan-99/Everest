FILE(GLOB CORE_CPU_SRC core/cpu/*)
FILE(GLOB CORE_GPU_SRC core/gpu/*)
add_library(libCore ${CORE_CPU_SRC} ${CORE_GPU_SRC})
# list(REMOVE_ITEM CORE_CPU_SRC "core/cpu/data.cc")
# add_library(libCore core/cpu/DataPack_cpu.cc core/cpu/GPUWorker.cc core/cpu/Graph.cc core/cpu/MineJob.cc core/cpu/MultiGPU.cc core/gpu/DataPack_gpu.cu)

target_link_libraries(libCore PUBLIC fmt::fmt)
target_link_libraries(libCore PUBLIC Boost::iostreams)
target_link_libraries(libCore PUBLIC range-v3)


add_executable(generateRand tools/generateRand.cc)
target_link_libraries(generateRand PUBLIC libCore)

add_executable(cacheGraph tools/cacheGraph.cc)
target_link_libraries(cacheGraph PUBLIC libCore)

pybind11_add_module(corelib mods/corelib.cc)
target_link_libraries(corelib PUBLIC libCore)
set_target_properties(corelib
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)



