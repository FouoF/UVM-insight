cmake_minimum_required(VERSION 3.18)
project(uvm_examples LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

foreach(name prefetch alternate oversubscribe thrash)
  add_executable(${name} ${name}.cu common.h)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
