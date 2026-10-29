#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call)                                                         \
  do {                                                                      \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

inline size_t argMB(int argc, char** argv, int index, size_t fallback) {
  if (argc > index && argv[index] != nullptr && argv[index][0] != '\0') {
    return static_cast<size_t>(std::strtoull(argv[index], nullptr, 10));
  }
  return fallback;
}
