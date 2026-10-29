#include "encode.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int fail(const char* msg) {
  std::cerr << msg << "\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  uvm::Buf helloEnv;
  uvm::Hello hello;
  hello.pid = 42;
  hello.comm = "vec";
  hello.app_name = "vecAdd";
  hello.hostname = "n1";
  hello.cuda_version = "12.8";
  hello.driver_version = "12.8";
  uvm::GpuInfo gpu;
  gpu.device_id = 0;
  gpu.uuid = "GPU-abc";
  gpu.name = "A100";
  gpu.total_memory_bytes = 80ull << 30;
  hello.gpus.push_back(gpu);
  uvm::encodeHello(helloEnv, hello);
  const std::string helloFrame = uvm::frame(helloEnv);

  // Envelope field 1, length-delimited: 0x0a, then a varint length.
  if (helloFrame.size() < 6 || static_cast<unsigned char>(helloFrame[4]) != 0x0a) {
    return fail("hello envelope tag");
  }
  const uint32_t helloLen = (static_cast<unsigned char>(helloFrame[0]) << 24) |
                            (static_cast<unsigned char>(helloFrame[1]) << 16) |
                            (static_cast<unsigned char>(helloFrame[2]) << 8) |
                            static_cast<unsigned char>(helloFrame[3]);
  if (helloLen + 4 != helloFrame.size()) {
    return fail("hello frame length");
  }

  uvm::Buf batchEnv;
  uvm::EventBatch batch;
  uvm::UvmEvent ev;
  ev.kind = uvm::EventKind::kBytesHtoD;
  ev.bytes = 4096;
  ev.start_ns = 1000;
  ev.end_ns = 2500;
  ev.address = 0x7f00000000ull;
  ev.src_id = 0xffffffffu;
  ev.dst_id = 0;
  ev.cause = uvm::MigrationCause::kCoherence;
  ev.stream_id = 7;
  ev.correlation_id = 9;
  ev.device_id = 0;
  batch.uvm.push_back(ev);

  uvm::AllocEvent alloc;
  alloc.address = 0x7f00000000ull;
  alloc.size = 1 << 20;
  alloc.is_managed = true;
  alloc.timestamp_ns = 900;
  alloc.device_id = 0;
  batch.allocs.push_back(alloc);

  uvm::KernelEvent kernel;
  kernel.name = "touch";
  kernel.start_ns = 1100;
  kernel.end_ns = 2400;
  kernel.device_id = 0;
  kernel.correlation_id = 9;
  kernel.stream_id = 7;
  batch.kernels.push_back(kernel);
  batch.dropped = 3;
  uvm::encodeBatch(batchEnv, batch);
  const std::string batchFrame = uvm::frame(batchEnv);
  if (static_cast<unsigned char>(batchFrame[4]) != 0x12) {
    return fail("batch envelope tag");
  }

  const char* maps =
      "7f0000000000-7f0000200000 rw-s 00000000 00:00 0 /dev/nvidia-uvm\n"
      "7f1000000000-7f1000001000 rw-s 00000000 00:00 0 /dev/nvidia-uvm-tools\n"
      "7f2000000000-7f2000100000 rw-p 00000000 00:00 0 [heap]\n"
      "7f3000000000-7f3000002000 rw-s 00000000 00:00 0 /dev/nvidia-uvm\n";
  if (uvm::sumNvidiaUvmMaps(maps) != (0x200000ull + 0x2000ull)) {
    return fail("uvm map sum");
  }

  const char* path = argc > 1 ? argv[1] : "proto/testdata/sample.frames";
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "open " << path << "\n";
    return 1;
  }
  out.write(helloFrame.data(), static_cast<std::streamsize>(helloFrame.size()));
  out.write(batchFrame.data(), static_cast<std::streamsize>(batchFrame.size()));
  std::cout << "wrote " << (helloFrame.size() + batchFrame.size()) << " bytes to " << path << "\n";
  return 0;
}
