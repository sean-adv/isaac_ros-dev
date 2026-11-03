cmake_minimum_required(VERSION 3.24)
# Set before enabling CUDA so compiler detection uses the Thor architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110 CACHE STRING "CUDA architectures")
endif()
project(dinov3_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(DEEPSTREAM_DIR "/opt/nvidia/deepstream/deepstream-9.1" CACHE PATH "DeepStream SDK root")
set(NATIVE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts" CACHE PATH "Native artifact directory")
option(NATIVE_BUILD_BRIDGES "Build DeepStream NvDCF helper" ON)

find_package(CUDAToolkit REQUIRED)
# These relative paths support both artifacts/{nvdcf,cuda} and bin/../lib.
# Use the same RPATH for build and install so renamed installed programs work too.
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../cuda;$ORIGIN/../lib;${CUDAToolkit_LIBRARY_DIR}")

add_library(gpu_geometry SHARED gpu_geometry.cu)
target_link_libraries(gpu_geometry PRIVATE CUDA::cudart)
set_target_properties(gpu_geometry PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${NATIVE_OUTPUT_DIR}/cuda")
install(TARGETS gpu_geometry LIBRARY DESTINATION lib)

if(NATIVE_BUILD_BRIDGES)
  find_path(DEEPSTREAM_INCLUDE_DIR nvdstracker.h
    PATHS "${DEEPSTREAM_DIR}/sources/includes" NO_DEFAULT_PATH REQUIRED)
  foreach(library nvbufsurface nvbufsurftransform)
    find_library(DEEPSTREAM_${library}_LIBRARY NAMES ${library}
      PATHS "${DEEPSTREAM_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
  endforeach()
  add_executable(nvdcf_bridge nvdcf_bridge.cpp)
  target_include_directories(nvdcf_bridge PRIVATE "${DEEPSTREAM_INCLUDE_DIR}")
  target_compile_options(nvdcf_bridge PRIVATE -Wall -Wextra)
  target_link_libraries(nvdcf_bridge PRIVATE gpu_geometry CUDA::cudart ${CMAKE_DL_LIBS}
    "${DEEPSTREAM_nvbufsurface_LIBRARY}" "${DEEPSTREAM_nvbufsurftransform_LIBRARY}")
  set_property(TARGET nvdcf_bridge APPEND PROPERTY INSTALL_RPATH "${DEEPSTREAM_DIR}/lib")
  set_target_properties(nvdcf_bridge PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${NATIVE_OUTPUT_DIR}/nvdcf")
  install(PROGRAMS "$<TARGET_FILE:nvdcf_bridge>" DESTINATION bin RENAME dinov3-nvdcf-bridge)
endif()
