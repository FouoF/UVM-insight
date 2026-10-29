#include "common.h"

// Allocate a managed region and walk every page from the GPU, then from the CPU.
// Pass a size larger than the GPU's memory to force eviction back to host memory:
//   oversubscribe 0
// uses total device memory + 512 MB. Any other number is a size in MB.

__global__ void touchPages(char* p, size_t pages) {
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pages; i += stride) {
    p[i * 4096] = 1;
  }
}

int main(int argc, char** argv) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  size_t bytes = 0;
  if (argc > 1 && std::strcmp(argv[1], "0") == 0) {
    bytes = totalBytes + (512ull << 20);
  } else {
    bytes = argMB(argc, argv, 1, 1024) << 20;
  }
  bytes &= ~size_t{4095};

  char* p = nullptr;
  std::printf("device memory %zu MB, allocating managed %zu MB\n", totalBytes >> 20, bytes >> 20);
  CHECK(cudaMallocManaged(&p, bytes));
  const size_t pages = bytes / 4096;
  touchPages<<<256, 256>>>(p, pages);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  for (size_t i = 0; i < pages; i += 16) {
    p[i * 4096] = 2;
  }
  CHECK(cudaFree(p));
  return 0;
}
