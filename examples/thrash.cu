#include "common.h"

// Ping-pong a small managed region between the CPU and the GPU.
// The driver may report this as thrashing and then throttle the CPU.
// Whether those counters fire depends on the GPU and the driver heuristic;
// the migrations and page faults are the part that always shows up.

__global__ void spin(int* p, int n, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = p[i] + value;
  }
}

int main(int argc, char** argv) {
  const size_t mb = argMB(argc, argv, 1, 64);
  const int rounds = static_cast<int>(argMB(argc, argv, 2, 400));
  const size_t bytes = mb << 20;
  const int n = static_cast<int>(bytes / sizeof(int));

  int* p = nullptr;
  CHECK(cudaMallocManaged(&p, bytes));
  std::printf("thrash %zu MB, %d rounds\n", mb, rounds);
  for (int r = 0; r < rounds; ++r) {
    spin<<<(n + 255) / 256, 256>>>(p, n, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < n; i += 256) {
      p[i] = p[i] + 1;
    }
  }
  CHECK(cudaFree(p));
  return 0;
}
