# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.18)

project(CustomOpDLib)

cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

cmake_policy(SET CMP0092 NEW)
cmake_policy(SET CMP0091 NEW)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
  cmake_policy(SET CMP0117 NEW)
endif()

set(CUSTOMOPLIB_ROOT ${PROJECT_SOURCE_DIR}/custom_op_library)

message(STATUS "CUSTOMOPLIB_ROOT = ${CUSTOMOPLIB_ROOT}")

add_library(custom_op_library SHARED ${CUSTOMOPLIB_ROOT}/custom_op_library.cc)
#add_executable(custom_op_library_o ${CUSTOMOPLIB_ROOT}/custom_op_library.cc)