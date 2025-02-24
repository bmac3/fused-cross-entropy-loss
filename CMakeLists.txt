cmake_minimum_required(VERSION 3.25.2)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX CUDA
)

set(PYBIND11_FINDPYTTHON True)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

find_package(pybind11 REQUIRED)
include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/cu_ext/cutlass/include
    ${CMAKE_SOURCE_DIR}/cu_ext/cutlass/tools/util/include
)

pybind11_add_module(
    fused_ce
    ${CMAKE_CURRENT_LIST_DIR}/cu_ext/cu_ext.cu
)

install(TARGETS fused_ce LIBRARY DESTINATION .)
