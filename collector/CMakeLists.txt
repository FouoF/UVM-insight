cmake_minimum_required(VERSION 3.18)
project(uvminsight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "uvm-insight collector requires CUDA 12.0 or newer (found ${CUDAToolkit_VERSION})")
endif()

add_library(uvminsight SHARED
  src/encode.cpp
  src/injection.cpp
)
target_include_directories(uvminsight PRIVATE src)
target_link_libraries(uvminsight PRIVATE CUDA::cupti ${CMAKE_DL_LIBS})
find_package(Threads REQUIRED)
target_link_libraries(uvminsight PRIVATE Threads::Threads)

set_target_properties(uvminsight PROPERTIES
  OUTPUT_NAME uvminsight
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)