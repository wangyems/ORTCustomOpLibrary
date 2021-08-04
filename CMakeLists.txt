# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.18)

project(CustomOpDLib LANGUAGES CXX CUDA)

set(CUSTOMOPLIB_ROOT ${PROJECT_SOURCE_DIR}/custom_op_library)
set(ORTPACKAGE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/ort_package)
set(ORTPACKAGE_INCLUDE ${ORTPACKAGE_ROOT}/build/native/include)
set(ORT_HEADER_SRC
    "${ORTPACKAGE_INCLUDE}/onnxruntime_c_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_inline.h")

file(GLOB CUSTOMOPLIB_SRC
     "${CUSTOMOPLIB_ROOT}/*.h"
     "${CUSTOMOPLIB_ROOT}/*.cc"
     "${CUSTOMOPLIB_ROOT}/*.cu"
     "${CUSTOMOPLIB_ROOT}/*.lds"
     "${CUSTOMOPLIB_ROOT}/*.def"
)

file(COPY ${ORT_HEADER_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CUSTOMOPLIB_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB CUSTOMOPCUDAIMPL_SRC
     "${CMAKE_CURRENT_BINARY_DIR}/sequence_pooling.h"
     "${CMAKE_CURRENT_BINARY_DIR}/sequence_pooling.cu")

file(GLOB CUSTOMOPCPUIMPL_SRC
     "${CMAKE_CURRENT_BINARY_DIR}/sequence_pooling.h"
     "${CMAKE_CURRENT_BINARY_DIR}/sequence_pooling.cc")

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_library(custom_op_library_fp32 SHARED
  ${CMAKE_CURRENT_BINARY_DIR}/custom_op_library_fp32.cc
  ${CUSTOMOPCUDAIMPL_SRC}
)

add_library(custom_op_library_fp16 SHARED
  ${CMAKE_CURRENT_BINARY_DIR}/custom_op_library_fp16.cc
  ${CUSTOMOPCUDAIMPL_SRC}
)

set_target_properties(
  custom_op_library_fp32 custom_op_library_fp16
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

add_library(custom_op_library_cpu SHARED
  ${CMAKE_CURRENT_BINARY_DIR}/custom_op_library_cpu.cc
  ${CUSTOMOPCPUIMPL_SRC}
)

if(UNIX)
  if (APPLE)
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker -dead_strip")
  else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG
        "-Xlinker --version-script=${CUSTOMOPLIB_ROOT}/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
  endif()
else()
  set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${CUSTOMOPLIB_ROOT}/custom_op_library.def")
endif()

set_property(
  TARGET custom_op_library_fp32 custom_op_library_fp16 custom_op_library_cpu
  APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG}
)
