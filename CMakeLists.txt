cmake_minimum_required(VERSION 3.15...3.27)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX CUDA
)

find_package(pybind11 REQUIRED)
include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/cu_ext/cutlass/include
    ${CMAKE_SOURCE_DIR}/cu_ext/cutlass/tools/util/include
)

pybind11_add_module(
    cu_ext
    ${CMAKE_CURRENT_LIST_DIR}/cu_ext/kernels.cu
    ${CMAKE_CURRENT_LIST_DIR}/cu_ext/cu_ext.cpp
)

install(TARGETS cu_ext LIBRARY DESTINATION .)
