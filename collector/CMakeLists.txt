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

# CUDA 12.9 documents CUpti_ActivityUnifiedMemoryCounter3 as the record emitted
# for CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER. That record's flags field is
# the migration cause. Older toolkits document Counter2, which is a shorter
# prefix, so flags are only read on 12.9+ unless the builder overrides this.
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.9)
  target_compile_definitions(uvminsight PRIVATE UVMI_READ_UM_FLAGS=1)
else()
  target_compile_definitions(uvminsight PRIVATE UVMI_READ_UM_FLAGS=0)
endif()
