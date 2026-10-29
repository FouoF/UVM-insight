#include "common.h"

// Touch every page of a managed allocation on the device, then prefetch it
// back to the CPU. This is the explicit-migration path (cause = user).

__global__ void touchPages(char* p, size_t pages) {
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pages; i += stride) {
    p[i * 4096] = static_cast<char>(i);
  }
}

int main(int argc, char** argv) {
  const size_t mb = argMB(argc, argv, 1, 256);
  const int rounds = static_cast<int>(argMB(argc, argv, 2, 8));
  const size_t bytes = mb << 20;

  int device = 0;
  CHECK(cudaGetDevice(&device));
  char* p = nullptr;
  CHECK(cudaMallocManaged(&p, bytes));
  std::printf("prefetch %zu MB, %d rounds\n", mb, rounds);

  for (int r = 0; r < rounds; ++r) {
    CHECK(cudaMemPrefetchAsync(p, bytes, device, 0));
    touchPages<<<128, 256>>>(p, bytes / 4096);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId, 0));
    CHECK(cudaDeviceSynchronize());
    volatile char sink = p[r * 4096];
    (void)sink;
  }
  CHECK(cudaFree(p));
  return 0;
}
