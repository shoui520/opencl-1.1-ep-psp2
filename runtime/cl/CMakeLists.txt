# The OpenCL runtime as a static library for Vita applications:
#   add_subdirectory(<this directory> opencl) and link OpenCL, or build it on
#   its own with the VitaSDK toolchain.
cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "Please define VITASDK to point to your SDK path!")
  endif()
endif()

project(opencl_psp2 C)

set(CL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_library(OpenCL STATIC
  core.c
  context.c
  queue.c
  mem.c
  devmem.c
  program.c
  ${CL_ROOT}/runtime/gpu/gxc.c
  ${CL_ROOT}/runtime/gpu/gxc_kernel.c
  ${CL_ROOT}/runtime/gpu/null_fragment.c
)
target_include_directories(OpenCL
  PUBLIC ${CL_ROOT}/third_party/OpenCL-Headers
  PRIVATE ${CL_ROOT}/runtime/gpu ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(OpenCL PUBLIC CL_TARGET_OPENCL_VERSION=110)
target_compile_options(OpenCL PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(OpenCL PUBLIC
  SceGxm_stub
  SceSysmem_stub
  SceKernelThreadMgr_stub
  SceProcessmgr_stub
  ScePower_stub
)
