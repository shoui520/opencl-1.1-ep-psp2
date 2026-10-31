cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "Please define VITASDK to point to your SDK path!")
  endif()
endif()

project(ocl_harness C)
include("$ENV{VITASDK}/share/vita.cmake" REQUIRED)

set(VITA_APP_NAME "OpenCL Harness")
set(VITA_TITLEID "OPCL00001")
set(VITA_VERSION "01.00")

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_subdirectory(${ROOT}/runtime/cl opencl)

add_executable(${PROJECT_NAME} main.c ep_api.c)
target_compile_options(${PROJECT_NAME} PRIVATE -O2 -Wall)
target_link_libraries(${PROJECT_NAME} OpenCL)

vita_create_self(${PROJECT_NAME}.self ${PROJECT_NAME})
vita_create_vpk(${PROJECT_NAME}.vpk ${VITA_TITLEID} ${PROJECT_NAME}.self
  VERSION ${VITA_VERSION}
  NAME ${VITA_APP_NAME}
)
