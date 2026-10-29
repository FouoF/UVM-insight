#include "encode.h"

namespace uvm {
namespace {

void writeGpu(Buf& out, const GpuInfo& gpu) {
  out.varint(1, gpu.device_id);
  out.str(2, gpu.uuid);
  out.str(3, gpu.name);
  out.varint(4, gpu.total_memory_bytes);
}

void writeUvm(Buf& out, const UvmEvent& ev) {
  out.varint(1, static_cast<uint64_t>(ev.kind));
  out.varint(2, ev.bytes);
  out.varint(3, ev.start_ns);
  out.varint(4, ev.end_ns);
  out.varint(5, ev.address);
  out.varint(6, ev.src_id);
  out.varint(7, ev.dst_id);
  out.varint(8, static_cast<uint64_t>(ev.cause));
  out.varint(9, ev.stream_id);
  out.varint(10, ev.correlation_id);
  out.varint(11, ev.device_id);
}

void writeAlloc(Buf& out, const AllocEvent& ev) {
  out.varint(1, ev.address);
  out.varint(2, ev.size);
  if (ev.is_managed) {
    out.varint(3, 1);
  }
  if (ev.is_free) {
    out.varint(4, 1);
  }
  out.varint(5, ev.timestamp_ns);
  out.varint(6, ev.device_id);
}

void writeKernel(Buf& out, const KernelEvent& ev) {
  out.str(1, ev.name);
  out.varint(2, ev.start_ns);
  out.varint(3, ev.end_ns);
  out.varint(4, ev.device_id);
  out.varint(5, ev.correlation_id);
  out.varint(6, ev.stream_id);
}

}  // namespace

void Buf::appendVarint(std::string& out, uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<char>((v & 0x7f) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<char>(v));
}

void Buf::key(uint32_t field, uint32_t wire) {
  appendVarint(buf_, (static_cast<uint64_t>(field) << 3) | wire);
}

void Buf::varint(uint32_t field, uint64_t v) {
  if (v == 0) {
    return;
  }
  key(field, 0);
  appendVarint(buf_, v);
}

void Buf::str(uint32_t field, std::string_view s) {
  if (s.empty()) {
    return;
  }
  key(field, 2);
  appendVarint(buf_, s.size());
  buf_.append(s.data(), s.size());
}

void Buf::msg(uint32_t field, const Buf& nested) {
  key(field, 2);
  appendVarint(buf_, nested.buf_.size());
  buf_.append(nested.buf_);
}

void encodeHello(Buf& envelope, const Hello& hello) {
  Buf body;
  body.varint(1, hello.pid);
  body.str(2, hello.comm);
  body.str(3, hello.app_name);
  body.str(4, hello.hostname);
  body.str(5, hello.cuda_version);
  body.str(6, hello.driver_version);
  for (const GpuInfo& gpu : hello.gpus) {
    Buf g;
    writeGpu(g, gpu);
    body.msg(7, g);
  }
  envelope.msg(1, body);
}

void encodeBatch(Buf& envelope, const EventBatch& batch) {
  Buf body;
  for (const UvmEvent& ev : batch.uvm) {
    Buf m;
    writeUvm(m, ev);
    body.msg(1, m);
  }
  for (const AllocEvent& ev : batch.allocs) {
    Buf m;
    writeAlloc(m, ev);
    body.msg(2, m);
  }
  for (const KernelEvent& ev : batch.kernels) {
    Buf m;
    writeKernel(m, ev);
    body.msg(3, m);
  }
  body.varint(4, batch.dropped);
  envelope.msg(2, body);
}

std::string frame(const Buf& envelope) {
  const std::string& payload = envelope.data();
  std::string out;
  out.resize(4 + payload.size());
  const uint32_t n = static_cast<uint32_t>(payload.size());
  out[0] = static_cast<char>((n >> 24) & 0xff);
  out[1] = static_cast<char>((n >> 16) & 0xff);
  out[2] = static_cast<char>((n >> 8) & 0xff);
  out[3] = static_cast<char>(n & 0xff);
  out.replace(4, payload.size(), payload);
  return out;
}

}  // namespace uvm
