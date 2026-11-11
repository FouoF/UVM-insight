#include "common.h"

#include <unistd.h>

// Hold a managed buffer on GPU then on host long enough for the collector
// to sample cuMemGetLocationInfo (or the delta fallback). Prefetch APIs
// differ across CUDA 12/13, so residency is driven by first-touch instead.

__global__ void touchPages(char* p, size_t pages) {
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pages;
       i += stride) {
    p[i * 4096] = static_cast<char>(i);
  }
}

int main(int argc, char** argv) {
  const size_t mb = argMB(argc, argv, 1, 256);
  const int holdSec = static_cast<int>(argMB(argc, argv, 2, 8));
  const size_t bytes = mb << 20;
  const size_t pages = bytes / 4096;

  CHECK(cudaSetDevice(0));
  char* p = nullptr;
  CHECK(cudaMallocManaged(&p, bytes));
  std::printf("residency_hold %zu MB, hold %ds per side\n", mb, holdSec);
  std::fflush(stdout);

  touchPages<<<128, 256>>>(p, pages);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::printf("phase=gpu\n");
  std::fflush(stdout);
  sleep(holdSec);

  for (size_t i = 0; i < pages; ++i) {
    volatile char sink = p[i * 4096];
    (void)sink;
  }
  std::printf("phase=host\n");
  std::fflush(stdout);
  sleep(holdSec);

  CHECK(cudaFree(p));
  return 0;
}
