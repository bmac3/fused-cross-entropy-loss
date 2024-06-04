cmake_minimum_required(VERSION 3.15...3.27)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX CUDA
)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    c_ext
    ${CMAKE_CURRENT_LIST_DIR}/csrc/kernels.cu
    ${CMAKE_CURRENT_LIST_DIR}/csrc/gpu_ops.cpp
)

install(TARGETS c_ext LIBRARY DESTINATION .)
