#include "common.h"

// GPU writes the buffer, then the CPU writes it, turn by turn.
// On Pascal and newer this shows up as coherence migrations.

__global__ void fill(int* p, int n, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    p[i] = value;
  }
}

int main(int argc, char** argv) {
  const size_t mb = argMB(argc, argv, 1, 256);
  const int rounds = static_cast<int>(argMB(argc, argv, 2, 20));
  const size_t bytes = mb << 20;
  const int n = static_cast<int>(bytes / sizeof(int));

  int* p = nullptr;
  CHECK(cudaMallocManaged(&p, bytes));
  std::printf("alternate %zu MB, %d rounds\n", mb, rounds);

  for (int r = 0; r < rounds; ++r) {
    fill<<<256, 256>>>(p, n, r);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < n; i += 1024) {
      p[i] = r + 1;
    }
  }
  CHECK(cudaFree(p));
  return 0;
}
