#include "encode.h"

#include <cupti.h>

#include <cuda.h>

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>

// CUDA_INJECTION64_PATH entry point. CUPTI calls this while the driver is
// loading. Keep the CUPTI setup here and push every blocking operation
// (socket, device enumeration) onto a background thread.

namespace {

// CUPTI recommends 1–10 MB buffers handed back from a pool. Returning a null
// buffer makes CUPTI drop records instead of stalling the application.
constexpr size_t kBufferBytes = 4 << 20;
constexpr int kPoolCap = 8;
constexpr uint64_t kAddrBucket = 2ull << 20;
constexpr size_t kMaxPendingEvents = 4096;
constexpr size_t kMaxQueuedFrames = 32;
constexpr uint32_t kCpuId = 0xffffffffu;

struct AggKey {
  uint32_t kind = 0;
  uint32_t cause = 0;
  uint32_t src = 0;
  uint32_t dst = 0;
  uint32_t device = 0;
  uint64_t addr = 0;
  bool operator<(const AggKey& o) const {
    return std::tie(kind, cause, src, dst, device, addr) <
           std::tie(o.kind, o.cause, o.src, o.dst, o.device, o.addr);
  }
};

struct UmView {
  CUpti_ActivityKind kind;
  CUpti_ActivityUnifiedMemoryCounterKind counterKind;
  uint64_t value;
  uint64_t start;
  uint64_t end;
  uint64_t address;
  uint32_t srcId;
  uint32_t dstId;
  uint32_t streamId;
  uint32_t processId;
  uint32_t flags;
  uint32_t pad;
} __attribute__((packed));

struct MemcpyPrefix {
  CUpti_ActivityKind kind;
  uint8_t copyKind;
  uint8_t srcKind;
  uint8_t dstKind;
  uint8_t flags;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  uint32_t correlationId;
  uint32_t runtimeCorrelationId;
} __attribute__((packed));

// The record struct a kind emits is chosen by the libcupti loaded at run
// time, which is often the copy bundled with PyTorch rather than the one this
// library was built against. Across CUDA 12.0-12.9 the newer records only
// append fields, so the prefixes below are read from whichever record arrives:
//   UNIFIED_MEMORY_COUNTER  Counter2 (<12.8), Counter3 (12.8+)
//   MEMORY2                 Memory3 (<12.6), Memory4 (12.6+)
//   MEMCPY                  Memcpy5 (<12.8), Memcpy6 (12.8+)
//   CONCURRENT_KERNEL       Kernel9
// In both counter records flags carries the migration cause for transfers.
static_assert(offsetof(CUpti_ActivityUnifiedMemoryCounter2, address) == offsetof(UmView, address),
              "unified memory counter prefix drifted");
static_assert(offsetof(CUpti_ActivityUnifiedMemoryCounter2, flags) == offsetof(UmView, flags),
              "unified memory counter flags drifted");
static_assert(offsetof(CUpti_ActivityMemcpy5, bytes) == offsetof(MemcpyPrefix, bytes), "memcpy prefix");
static_assert(offsetof(CUpti_ActivityMemcpy5, correlationId) == offsetof(MemcpyPrefix, correlationId),
              "memcpy correlation");
#if CUPTI_API_VERSION >= 26
static_assert(offsetof(CUpti_ActivityUnifiedMemoryCounter3, flags) == offsetof(UmView, flags),
              "unified memory counter flags drifted");
static_assert(offsetof(CUpti_ActivityMemcpy6, correlationId) == offsetof(MemcpyPrefix, correlationId),
              "memcpy correlation");
#endif
using ActivityMemory = CUpti_ActivityMemory3;
#if CUPTI_API_VERSION >= 24
static_assert(offsetof(CUpti_ActivityMemory4, deviceId) == offsetof(ActivityMemory, deviceId), "memory record");
static_assert(offsetof(CUpti_ActivityMemory4, timestamp) == offsetof(ActivityMemory, timestamp), "memory record");
#endif
static_assert(offsetof(CUpti_ActivityKernel9, name) > 0, "kernel record");

// CUPTI 12.0 reports API version 17. Older libraries emit different records.
constexpr uint32_t kMinCuptiApi = 17;

struct Pending {
  std::vector<uvm::UvmEvent> uvm;
  std::vector<uvm::AllocEvent> allocs;
  std::vector<uvm::KernelEvent> kernels;
  uint64_t dropped = 0;
};

// Feature 1 — swap events: UVMI_EVENTS. Feature 2 — residency gauges:
// UVMI_RESIDENCY=off|auto|driver|delta. Location API when available; CUPTI
// HtoD/DtoH inference as fallback. The two pipelines are independent.
enum class ResidencyWanted { Off, Auto, Driver, Delta };
enum class ResidencySrc { None, Driver, Delta };

using CuMemGetLocationInfoFn = CUresult (*)(CUdeviceptr, size_t, size_t, size_t, CUmemLocation*);
using CuDeviceGetAttributeFn = CUresult (*)(int*, CUdevice_attribute, CUdevice);
using CuCtxGetCurrentFn = CUresult (*)(CUcontext*);
using CuCtxSetCurrentFn = CUresult (*)(CUcontext);
using CuDevicePrimaryCtxRetainFn = CUresult (*)(CUcontext*, CUdevice);
using CuGetErrorStringFn = CUresult (*)(CUresult, const char**);
using CuDeviceGetFn = CUresult (*)(CUdevice*, int);

struct Collector {
  std::string socketPath = "/tmp/uvm-insight/agent.sock";
  int flushMs = 200;
  bool enableEvents = true;  // UVMI_EVENTS — migration / fault CUPTI path
  bool enableMemcpy = true;
  bool enableKernel = true;
  bool quiet = false;
  ResidencyWanted residencyWanted = ResidencyWanted::Auto;
  ResidencySrc residencySrc = ResidencySrc::None;
  // Majority-vote region size / sample stride for cuMemGetLocationInfo.
  size_t locSummaryBytes = 64ull << 20;
  size_t locSampleBytes = 2ull << 20;
  CuMemGetLocationInfoFn locInfoFn = nullptr;
  CuCtxGetCurrentFn ctxGetCurrentFn = nullptr;
  CuCtxSetCurrentFn ctxSetCurrentFn = nullptr;
  CuDevicePrimaryCtxRetainFn primaryRetainFn = nullptr;
  CuDeviceGetFn deviceGetFn = nullptr;
  CuGetErrorStringFn getErrorStringFn = nullptr;
  bool locationApiOk = false;
  bool residencyDriverActive = false;  // true after a successful location sample
  bool locQueryWarned = false;
  // Last CUDA context observed on a CUPTI activity buffer (app thread).
  CUcontext lastCtx = nullptr;

  std::mutex mu;
  Pending pending;
  // Byte totals for one flush window, keyed like Nsight's per-address summary.
  std::map<AggKey, uint64_t> agg;

  std::mutex bufMu;
  std::vector<uint8_t*> freeBufs;
  struct Filled {
    uint8_t* data = nullptr;
    size_t valid = 0;
    CUcontext ctx = nullptr;
    uint32_t stream = 0;
  };
  std::deque<Filled> filled;
  bool poolReady = false;

  std::mutex sendMu;
  int fd = -1;
  std::deque<std::string> queue;
  uint64_t unsentDrops = 0;
  uvm::Hello hello;
  bool helloReady = false;
  uint64_t managedBytes = 0;
  uint64_t managedSent = 0;
  bool managedKnown = false;
  bool managedSentValid = false;
  // Three-way split of managedBytes: gpu + host + unknown.
  // unknown = managedBytes - gpuBytes - hostBytes is implicit.
  // Delta mode: HtoD/DtoH move mass between the buckets; a DtoH that exceeds
  // gpuBytes imputes missing prior GPU residency (first-touch) then credits host.
  // Driver mode: cuMemGetLocationInfo overwrites these each refresh.
  uint64_t gpuBytes = 0;
  uint64_t hostBytes = 0;
  uint64_t resident = 0;
  uint64_t hostResident = 0;
  uint64_t residentSent = 0;
  uint64_t hostResidentSent = 0;

  std::mutex dropMu;
  std::vector<std::pair<uint64_t, size_t>> dropBase;

  std::atomic<bool> stop{false};
  std::atomic<bool> disabled{false};
  std::atomic<bool> forked{false};
  std::thread worker;
  bool started = false;
};

Collector& C() {
  static Collector c;
  return c;
}

void logf(const char* fmt, ...) {
  if (C().quiet) {
    return;
  }
  std::fprintf(stderr, "[uvm-insight] ");
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
}

int envInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') {
    return fallback;
  }
  return std::atoi(v);
}

bool envEnabled(const char* name, bool fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') {
    return fallback;
  }
  return std::atoi(v) != 0;
}

std::string readTokenFile(const char* path) {
  FILE* f = std::fopen(path, "r");
  if (f == nullptr) {
    return {};
  }
  char buf[1024];
  const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  buf[n] = '\0';
  for (size_t i = 0; i < n; ++i) {
    if (buf[i] == '\0' || buf[i] == '\n') {
      buf[i] = '\0';
      break;
    }
  }
  std::string s(buf);
  const auto slash = s.find_last_of('/');
  if (slash != std::string::npos) {
    s = s.substr(slash + 1);
  }
  return s;
}

const char* cuptiErr(CUptiResult st) {
  const char* s = nullptr;
  cuptiGetResultString(st, &s);
  return s != nullptr ? s : "unknown";
}

void noteDrop(uint64_t n) {
  if (n == 0) {
    return;
  }
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.mu);
  c.pending.dropped += n;
}

template <typename T>
void pushBounded(std::vector<T>& dst, T ev) {
  auto& c = C();
  const size_t total = c.pending.uvm.size() + c.pending.allocs.size() + c.pending.kernels.size();
  if (total >= kMaxPendingEvents) {
    c.pending.dropped++;
    return;
  }
  dst.push_back(std::move(ev));
}

uvm::MigrationCause mapCause(uint32_t flags) {
  switch (flags) {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER:
      return uvm::MigrationCause::kUser;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE:
      return uvm::MigrationCause::kCoherence;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH:
      return uvm::MigrationCause::kPrefetch;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION:
      return uvm::MigrationCause::kEviction;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS:
      return uvm::MigrationCause::kAccessCounters;
    default:
      return uvm::MigrationCause::kUnknown;
  }
}

uint32_t deviceFor(CUpti_ActivityUnifiedMemoryCounterKind kind, uint32_t src, uint32_t dst) {
  switch (kind) {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
      return dst;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
      return src;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
      return src;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
      return kCpuId;
    default:
      return src;
  }
}

void fold(const uvm::UvmEvent& ev);

void onUnified(const CUpti_Activity* rec) {
  UmView view{};
  std::memcpy(&view, rec, offsetof(UmView, pad));

  uvm::UvmEvent ev;
  ev.start_ns = view.start;
  ev.end_ns = view.end;
  ev.address = view.address;
  ev.src_id = view.srcId;
  ev.dst_id = view.dstId;
  ev.stream_id = view.streamId;
  ev.device_id = deviceFor(view.counterKind, view.srcId, view.dstId);

  switch (view.counterKind) {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
      ev.kind = uvm::EventKind::kBytesHtoD;
      ev.bytes = view.value;
      ev.cause = mapCause(view.flags);
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
      ev.kind = uvm::EventKind::kBytesDtoH;
      ev.bytes = view.value;
      ev.cause = mapCause(view.flags);
      break;
#ifdef CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD:
      ev.kind = uvm::EventKind::kBytesDtoD;
      ev.bytes = view.value;
      ev.device_id = view.dstId;
      if (view.flags <= CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS) {
        ev.cause = mapCause(view.flags);
      }
      break;
#endif
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
      ev.kind = uvm::EventKind::kCpuPageFault;
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
      ev.kind = uvm::EventKind::kGpuPageFault;
      ev.bytes = view.value == 0 ? 1 : view.value;
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
      ev.kind = uvm::EventKind::kThrashing;
      ev.bytes = view.value;
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
      ev.kind = uvm::EventKind::kThrottling;
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP:
      ev.kind = uvm::EventKind::kRemoteMap;
      ev.bytes = view.value;
      break;
    default:
      return;
  }

  fold(ev);
}

bool isTransfer(uvm::EventKind kind) {
  switch (kind) {
    case uvm::EventKind::kBytesHtoD:
    case uvm::EventKind::kBytesDtoH:
    case uvm::EventKind::kBytesDtoD:
    case uvm::EventKind::kMemcpyHtoD:
    case uvm::EventKind::kMemcpyDtoH:
    case uvm::EventKind::kMemcpyDtoD:
    case uvm::EventKind::kMemcpyOther:
    case uvm::EventKind::kRemoteMap:
      return true;
    default:
      return false;
  }
}

void applyHtoD(Collector& c, uint64_t n) {
  if (n == 0) {
    return;
  }
  const uint64_t fromHost = std::min(c.hostBytes, n);
  c.hostBytes -= fromHost;
  c.gpuBytes += n;
}

void applyDtoH(Collector& c, uint64_t n) {
  if (n == 0) {
    return;
  }
  if (c.gpuBytes >= n) {
    c.gpuBytes -= n;
    c.hostBytes += n;
    return;
  }
  // gpuBytes underflows: pages were on the GPU without a recorded HtoD
  // (first-touch). Impute that residency, then move the full transfer to host.
  c.hostBytes += n;
  c.gpuBytes = 0;
}

void fold(const uvm::UvmEvent& ev) {
  AggKey key;
  key.kind = static_cast<uint32_t>(ev.kind);
  key.cause = static_cast<uint32_t>(ev.cause);
  key.src = ev.src_id;
  key.dst = ev.dst_id;
  key.device = ev.device_id;
  if (isTransfer(ev.kind)) {
    key.addr = ev.address & ~(kAddrBucket - 1);
  }
  uint64_t n = ev.bytes;
  if (!isTransfer(ev.kind) && n == 0) {
    n = 1;
  }
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.mu);
  c.agg[key] += n;
  // Delta residency only; driver mode overwrites gauges in refreshManaged.
  if (c.residencySrc != ResidencySrc::Delta) {
    return;
  }
  switch (ev.kind) {
    case uvm::EventKind::kBytesHtoD:
    case uvm::EventKind::kMemcpyHtoD:
      applyHtoD(c, ev.bytes);
      break;
    case uvm::EventKind::kBytesDtoH:
    case uvm::EventKind::kMemcpyDtoH:
      applyDtoH(c, ev.bytes);
      break;
    default:
      break;
  }
}

void onMemory(const CUpti_Activity* rec) {
  const auto* m = reinterpret_cast<const ActivityMemory*>(rec);
  const bool managed = m->memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED ||
                       m->memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC;
  uvm::AllocEvent ev;
  ev.address = m->address;
  ev.size = m->bytes;
  ev.is_managed = managed;
  ev.is_free = m->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE;
  ev.timestamp_ns = m->timestamp;
  ev.device_id = m->deviceId;
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.mu);
  pushBounded(c.pending.allocs, ev);
}

void onKernel(const CUpti_Activity* rec) {
  const auto* k = reinterpret_cast<const CUpti_ActivityKernel9*>(rec);
  uvm::KernelEvent ev;
  if (k->name != nullptr) {
    ev.name.assign(k->name, std::min<size_t>(std::strlen(k->name), 256));
  }
  ev.start_ns = k->start;
  ev.end_ns = k->end;
  ev.device_id = k->deviceId;
  ev.correlation_id = k->correlationId;
  ev.stream_id = k->streamId;
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.mu);
  pushBounded(c.pending.kernels, ev);
}

uvm::EventKind memcpyKind(uint8_t copyKind) {
  switch (copyKind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return uvm::EventKind::kMemcpyHtoD;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return uvm::EventKind::kMemcpyDtoH;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return uvm::EventKind::kMemcpyDtoD;
    default:
      return uvm::EventKind::kMemcpyOther;
  }
}

void onMemcpy(const CUpti_Activity* rec) {
  MemcpyPrefix view{};
  std::memcpy(&view, rec, sizeof(view));
  uvm::UvmEvent ev;
  ev.kind = memcpyKind(view.copyKind);
  ev.bytes = view.bytes;
  ev.start_ns = view.start;
  ev.end_ns = view.end;
  ev.device_id = view.deviceId;
  ev.dst_id = view.deviceId;
  ev.src_id = view.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH ? view.deviceId : kCpuId;
  ev.stream_id = view.streamId;
  ev.correlation_id = view.correlationId;
  fold(ev);
}

void handleRecord(const CUpti_Activity* rec) {
  if (C().disabled.load(std::memory_order_relaxed)) {
    return;
  }
  switch (rec->kind) {
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
      onUnified(rec);
      break;
    case CUPTI_ACTIVITY_KIND_MEMORY2:
      onMemory(rec);
      break;
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      onKernel(rec);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      onMemcpy(rec);
      break;
    default:
      break;
  }
}

void initPool() {
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.bufMu);
  if (c.poolReady) {
    return;
  }
  c.freeBufs.reserve(kPoolCap);
  for (int i = 0; i < kPoolCap; ++i) {
    void* p = std::malloc(kBufferBytes);
    if (p != nullptr) {
      c.freeBufs.push_back(static_cast<uint8_t*>(p));
    }
  }
  c.poolReady = true;
  logf("activity buffer pool %zu x %zu", c.freeBufs.size(), kBufferBytes);
}

void recycleBuffer(uint8_t* buffer) {
  if (buffer == nullptr) {
    return;
  }
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.bufMu);
  c.freeBufs.push_back(buffer);
}

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *buffer = nullptr;
  *size = 0;
  *maxNumRecords = 0;
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.bufMu);
  if (c.freeBufs.empty()) {
    return;
  }
  *buffer = c.freeBufs.back();
  c.freeBufs.pop_back();
  *size = kBufferBytes;
}

void accountDropped(CUcontext ctx, uint32_t streamId) {
  size_t dropped = 0;
  const CUptiResult st = cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
  if (st != CUPTI_SUCCESS) {
    return;
  }
  const uint64_t key = (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctx)) << 16) ^ streamId;
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.dropMu);
  size_t prev = 0;
  for (auto& slot : c.dropBase) {
    if (slot.first == key) {
      prev = slot.second;
      slot.second = dropped;
      if (dropped >= prev) {
        noteDrop(dropped - prev);
      }
      return;
    }
  }
  c.dropBase.emplace_back(key, dropped);
  noteDrop(dropped);
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t /*size*/,
                              size_t validSize) {
  if (buffer == nullptr) {
    return;
  }
  auto& c = C();
  if (c.disabled.load(std::memory_order_relaxed)) {
    recycleBuffer(buffer);
    return;
  }
  Collector::Filled slot;
  slot.data = buffer;
  slot.valid = validSize;
  slot.ctx = ctx;
  slot.stream = streamId;
  std::lock_guard<std::mutex> lock(c.bufMu);
  if (ctx != nullptr) {
    c.lastCtx = ctx;
  }
  c.filled.push_back(slot);
}

void drainBuffers() {
  auto& c = C();
  for (;;) {
    Collector::Filled slot;
    {
      std::lock_guard<std::mutex> lock(c.bufMu);
      if (c.filled.empty()) {
        break;
      }
      slot = c.filled.front();
      c.filled.pop_front();
    }
    if (!c.disabled.load(std::memory_order_relaxed) && slot.valid > 0) {
      CUpti_Activity* rec = nullptr;
      for (;;) {
        const CUptiResult st = cuptiActivityGetNextRecord(slot.data, slot.valid, &rec);
        if (st != CUPTI_SUCCESS) {
          break;
        }
        handleRecord(rec);
      }
      accountDropped(slot.ctx, slot.stream);
    }
    recycleBuffer(slot.data);
  }
}

using CuInitFn = CUresult (*)(unsigned int);
using CuGetCountFn = CUresult (*)(int*);
using CuGetFn = CUresult (*)(CUdevice*, int);
using CuNameFn = CUresult (*)(char*, int, CUdevice);

bool probeLocationApi(Collector& c, void* lib, CuGetCountFn cuCount, CuGetFn cuGet) {
  c.locInfoFn = nullptr;
  c.locationApiOk = false;
  auto locFn = reinterpret_cast<CuMemGetLocationInfoFn>(dlsym(lib, "cuMemGetLocationInfo"));
  auto cuAttr = reinterpret_cast<CuDeviceGetAttributeFn>(dlsym(lib, "cuDeviceGetAttribute"));
  if (locFn == nullptr || cuAttr == nullptr || cuCount == nullptr || cuGet == nullptr) {
    return false;
  }
  int count = 0;
  if (cuCount(&count) != CUDA_SUCCESS || count <= 0) {
    return false;
  }
  for (int i = 0; i < count; ++i) {
    CUdevice dev;
    if (cuGet(&dev, i) != CUDA_SUCCESS) {
      return false;
    }
    int cma = 0;
    if (cuAttr(&cma, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev) != CUDA_SUCCESS || cma == 0) {
      return false;
    }
  }
  c.locInfoFn = locFn;
  c.ctxGetCurrentFn = reinterpret_cast<CuCtxGetCurrentFn>(dlsym(lib, "cuCtxGetCurrent"));
  c.ctxSetCurrentFn = reinterpret_cast<CuCtxSetCurrentFn>(dlsym(lib, "cuCtxSetCurrent"));
  c.primaryRetainFn = reinterpret_cast<CuDevicePrimaryCtxRetainFn>(dlsym(lib, "cuDevicePrimaryCtxRetain"));
  c.deviceGetFn = cuGet;
  c.getErrorStringFn = reinterpret_cast<CuGetErrorStringFn>(dlsym(lib, "cuGetErrorString"));
  c.locationApiOk = true;
  return true;
}

const char* residencySrcName(ResidencySrc src) {
  switch (src) {
    case ResidencySrc::Driver:
      return "driver";
    case ResidencySrc::Delta:
      return "delta";
    default:
      return "none";
  }
}

void resolveResidencySrc(Collector& c) {
  c.residencySrc = ResidencySrc::None;
  c.residencyDriverActive = false;
  switch (c.residencyWanted) {
    case ResidencyWanted::Off:
      break;
    case ResidencyWanted::Driver:
      if (c.locationApiOk) {
        c.residencySrc = ResidencySrc::Driver;
      } else {
        logf("residency_source=none (driver requested, cuMemGetLocationInfo unavailable)");
      }
      break;
    case ResidencyWanted::Delta:
      if (c.enableEvents) {
        c.residencySrc = ResidencySrc::Delta;
      } else {
        logf("residency_source=none (delta needs UVMI_EVENTS=1)");
      }
      break;
    case ResidencyWanted::Auto:
    default:
      if (c.locationApiOk) {
        c.residencySrc = ResidencySrc::Driver;
      } else if (c.enableEvents) {
        c.residencySrc = ResidencySrc::Delta;
      }
      break;
  }
  c.hello.residency_source = residencySrcName(c.residencySrc);
  if (c.residencySrc == ResidencySrc::Driver) {
    logf("events=%d residency_source=driver summary=%zuMiB sample=%zuMiB", c.enableEvents ? 1 : 0,
         c.locSummaryBytes >> 20, c.locSampleBytes >> 20);
  } else {
    logf("events=%d residency_source=%s", c.enableEvents ? 1 : 0, residencySrcName(c.residencySrc));
  }
}

bool ensureCudaContext(Collector& c) {
  if (c.ctxGetCurrentFn == nullptr || c.ctxSetCurrentFn == nullptr) {
    return true;
  }
  CUcontext cur = nullptr;
  if (c.ctxGetCurrentFn(&cur) == CUDA_SUCCESS && cur != nullptr) {
    return true;
  }
  // Prefer the context CUPTI attributed activity to; it is the app's context.
  if (c.lastCtx != nullptr && c.ctxSetCurrentFn(c.lastCtx) == CUDA_SUCCESS) {
    return true;
  }
  if (c.primaryRetainFn == nullptr || c.deviceGetFn == nullptr) {
    return false;
  }
  CUdevice dev;
  if (c.deviceGetFn(&dev, 0) != CUDA_SUCCESS) {
    return false;
  }
  CUcontext primary = nullptr;
  if (c.primaryRetainFn(&primary, dev) != CUDA_SUCCESS || primary == nullptr) {
    return false;
  }
  return c.ctxSetCurrentFn(primary) == CUDA_SUCCESS;
}

bool queryLocationResidency(Collector& c, const std::vector<uvm::UvmMapRange>& ranges, uint64_t* gpuOut,
                            uint64_t* hostOut) {
  if (c.locInfoFn == nullptr || ranges.empty()) {
    return false;
  }
  if (!ensureCudaContext(c)) {
    if (!c.locQueryWarned) {
      logf("cuMemGetLocationInfo skipped: no CUDA context on collector thread");
      c.locQueryWarned = true;
    }
    return false;
  }
  uint64_t gpu = 0;
  uint64_t host = 0;
  std::vector<CUmemLocation> locs;
  int okRanges = 0;
  CUresult lastErr = CUDA_SUCCESS;
  for (const uvm::UvmMapRange& r : ranges) {
    if (r.size == 0) {
      continue;
    }
    size_t summary = c.locSummaryBytes;
    if (summary == 0 || summary > r.size) {
      summary = r.size;
    }
    size_t sampling = c.locSampleBytes;
    if (sampling > summary) {
      sampling = summary;
    }
    const size_t n = (r.size + summary - 1) / summary;
    locs.assign(n, CUmemLocation{});
    const CUresult st = c.locInfoFn(static_cast<CUdeviceptr>(r.start), static_cast<size_t>(r.size), summary,
                                    sampling, locs.data());
    if (st != CUDA_SUCCESS) {
      // /proc maps can include nvidia-uvm regions that are not queryable
      // managed allocations (driver bookkeeping). Skip those and keep going.
      lastErr = st;
      continue;
    }
    ++okRanges;
    for (size_t i = 0; i < n; ++i) {
      size_t chunk = summary;
      if ((i + 1) * summary > r.size) {
        chunk = static_cast<size_t>(r.size - i * summary);
      }
      switch (locs[i].type) {
        case CU_MEM_LOCATION_TYPE_DEVICE:
          gpu += chunk;
          break;
        case CU_MEM_LOCATION_TYPE_HOST:
        case CU_MEM_LOCATION_TYPE_HOST_NUMA:
        case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
          host += chunk;
          break;
        default:
          break;
      }
    }
    if (!c.locQueryWarned && n > 0) {
      logf("location sample first_type=%d id=%d regions=%zu range=%zuMiB -> gpu=%zuMiB host=%zuMiB",
           static_cast<int>(locs[0].type), locs[0].id, n, r.size >> 20, gpu >> 20, host >> 20);
      c.locQueryWarned = true;
    }
  }
  if (okRanges == 0) {
    if (!c.locQueryWarned) {
      const char* msg = nullptr;
      if (c.getErrorStringFn != nullptr && lastErr != CUDA_SUCCESS) {
        c.getErrorStringFn(lastErr, &msg);
      }
      logf("cuMemGetLocationInfo failed: %s", msg != nullptr ? msg : "no queryable ranges");
      c.locQueryWarned = true;
    }
    return false;
  }
  *gpuOut = gpu;
  *hostOut = host;
  return true;
}
using CuMemFn = CUresult (*)(size_t*, CUdevice);
using CuUuidFn = CUresult (*)(CUuuid*, CUdevice);
using CuVersionFn = CUresult (*)(int*);

std::string formatUuid(const CUuuid& id) {
  const unsigned char* b = reinterpret_cast<const unsigned char*>(id.bytes);
  char buf[64];
  std::snprintf(buf, sizeof(buf),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1],
                b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return buf;
}

std::string versionString(int packed) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%d.%d", packed / 1000, (packed % 1000) / 10);
  return buf;
}

void fillHello() {
  auto& c = C();
  const uint32_t api = c.hello.cupti_api_version;
  c.hello = uvm::Hello{};
  c.hello.cupti_api_version = api;
  c.hello.pid = static_cast<uint32_t>(::getpid());
  c.hello.comm = readTokenFile("/proc/self/comm");
  const char* app = std::getenv("UVMI_APP_NAME");
  if (app != nullptr && app[0] != '\0') {
    c.hello.app_name = app;
  } else {
    c.hello.app_name = readTokenFile("/proc/self/cmdline");
    if (c.hello.app_name.empty()) {
      c.hello.app_name = c.hello.comm;
    }
  }
  char host[256];
  if (::gethostname(host, sizeof(host)) == 0) {
    host[sizeof(host) - 1] = '\0';
    c.hello.hostname = host;
  }
  {
    std::ifstream in("/proc/self/mountinfo");
    if (in) {
      const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      uvm::parseKubeletMounts(text, &c.hello.pod_uid, &c.hello.container_name);
    }
  }
  // An agent outside the host pid namespace cannot see the real pid, and
  // containerized processes are often pid 1. Mix in the hostname so pods
  // sharing one agent stay separate.
  if (c.hello.pid <= 1 && !c.hello.hostname.empty()) {
    uint32_t h = 2166136261u;
    for (unsigned char ch : c.hello.hostname) {
      h ^= ch;
      h *= 16777619u;
    }
    if (h < 2) {
      h += 2;
    }
    c.hello.pid = h;
  }

  void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (lib == nullptr) {
    lib = dlopen("libcuda.so.1", RTLD_NOW);
  }
  if (lib == nullptr) {
    c.helloReady = true;
    return;
  }
  auto cuInit = reinterpret_cast<CuInitFn>(dlsym(lib, "cuInit"));
  auto cuCount = reinterpret_cast<CuGetCountFn>(dlsym(lib, "cuDeviceGetCount"));
  auto cuGet = reinterpret_cast<CuGetFn>(dlsym(lib, "cuDeviceGet"));
  auto cuName = reinterpret_cast<CuNameFn>(dlsym(lib, "cuDeviceGetName"));
  auto cuMem = reinterpret_cast<CuMemFn>(dlsym(lib, "cuDeviceTotalMem_v2"));
  if (cuMem == nullptr) {
    cuMem = reinterpret_cast<CuMemFn>(dlsym(lib, "cuDeviceTotalMem"));
  }
  auto cuUuid = reinterpret_cast<CuUuidFn>(dlsym(lib, "cuDeviceGetUuid"));
  auto cuVer = reinterpret_cast<CuVersionFn>(dlsym(lib, "cuDriverGetVersion"));
  if (cuInit == nullptr || cuCount == nullptr || cuGet == nullptr) {
    c.helloReady = true;
    return;
  }
  if (cuInit(0) != CUDA_SUCCESS) {
    c.helloReady = true;
    return;
  }
  int driver = 0;
  if (cuVer != nullptr && cuVer(&driver) == CUDA_SUCCESS) {
    c.hello.driver_version = versionString(driver);
    c.hello.cuda_version = c.hello.driver_version;
  }
  int count = 0;
  if (cuCount(&count) == CUDA_SUCCESS) {
    for (int i = 0; i < count; ++i) {
      CUdevice dev;
      if (cuGet(&dev, i) != CUDA_SUCCESS) {
        continue;
      }
      uvm::GpuInfo gpu;
      gpu.device_id = static_cast<uint32_t>(i);
      char name[128];
      name[0] = '\0';
      if (cuName != nullptr && cuName(name, sizeof(name), dev) == CUDA_SUCCESS) {
        gpu.name = name;
      }
      size_t bytes = 0;
      if (cuMem != nullptr && cuMem(&bytes, dev) == CUDA_SUCCESS) {
        gpu.total_memory_bytes = bytes;
      }
      CUuuid uuid{};
      if (cuUuid != nullptr && cuUuid(&uuid, dev) == CUDA_SUCCESS) {
        gpu.uuid = formatUuid(uuid);
      }
      c.hello.gpus.push_back(std::move(gpu));
    }
  }
  probeLocationApi(c, lib, cuCount, cuGet);
  resolveResidencySrc(c);
  c.helloReady = true;
}

bool writeAll(int fd, const char* data, size_t n) {
  while (n > 0) {
    const ssize_t w = ::send(fd, data, n, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    data += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

bool connectSocket(Collector& c) {
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return false;
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (c.socketPath.size() >= sizeof(addr.sun_path)) {
    ::close(fd);
    return false;
  }
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", c.socketPath.c_str());
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return false;
  }
  uvm::Buf env;
  uvm::encodeHello(env, c.hello);
  const std::string framed = uvm::frame(env);
  if (!writeAll(fd, framed.data(), framed.size())) {
    ::close(fd);
    return false;
  }
  c.fd = fd;
  logf("connected pid=%u socket=%s", c.hello.pid, c.socketPath.c_str());
  return true;
}

// eventCount is the number of records in this frame. If the queue is full the
// frame is discarded and that count is carried on the next frame that fits.
bool enqueueFrame(Collector& c, std::string framed, uint64_t eventCount) {
  std::lock_guard<std::mutex> lock(c.sendMu);
  if (c.queue.size() >= kMaxQueuedFrames) {
    c.unsentDrops += eventCount;
    return false;
  }
  c.queue.push_back(std::move(framed));
  return true;
}

void refreshManaged(Collector& c) {
  std::ifstream in("/proc/self/maps");
  if (!in) {
    return;
  }
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::vector<uvm::UvmMapRange> ranges = uvm::listNvidiaUvmMaps(text);
  const uint64_t before = c.managedBytes;
  uint64_t managed = 0;
  for (const uvm::UvmMapRange& r : ranges) {
    managed += r.size;
  }
  c.managedBytes = managed;
  c.managedKnown = true;
  c.hello.managed_bytes = c.managedBytes;
  c.hello.managed_known = true;
  c.hello.residency_source = residencySrcName(c.residencySrc);

  uint64_t driverGpu = 0;
  uint64_t driverHost = 0;
  const bool gotDriver =
      c.residencySrc == ResidencySrc::Driver && queryLocationResidency(c, ranges, &driverGpu, &driverHost);

  std::lock_guard<std::mutex> lock(c.mu);
  if (c.residencySrc == ResidencySrc::None) {
    c.gpuBytes = 0;
    c.hostBytes = 0;
  } else if (gotDriver) {
    c.residencyDriverActive = true;
    if (driverGpu + driverHost > c.managedBytes && c.managedBytes > 0) {
      const double scale = static_cast<double>(c.managedBytes) / static_cast<double>(driverGpu + driverHost);
      driverGpu = static_cast<uint64_t>(static_cast<double>(driverGpu) * scale);
      driverHost = c.managedBytes - driverGpu;
    }
    c.gpuBytes = driverGpu;
    c.hostBytes = driverHost;
  } else if (c.residencySrc == ResidencySrc::Delta) {
    // A free leaves no eviction record. Scale both observed buckets with the
    // footprint so unknown stays non-negative in proportion.
    if (before > 0 && c.managedBytes < before) {
      const double keep = static_cast<double>(c.managedBytes) / static_cast<double>(before);
      c.gpuBytes = static_cast<uint64_t>(static_cast<double>(c.gpuBytes) * keep);
      c.hostBytes = static_cast<uint64_t>(static_cast<double>(c.hostBytes) * keep);
    }
    const uint64_t accounted = c.gpuBytes + c.hostBytes;
    if (accounted > c.managedBytes && accounted > 0) {
      const double scale = static_cast<double>(c.managedBytes) / static_cast<double>(accounted);
      c.gpuBytes = static_cast<uint64_t>(static_cast<double>(c.gpuBytes) * scale);
      c.hostBytes = c.managedBytes - c.gpuBytes;
    }
  }
  // Driver mode before the first successful sample: leave prior zeros.
  c.resident = c.gpuBytes;
  c.hostResident = c.hostBytes;
  c.hello.gpu_resident_bytes = c.resident;
  c.hello.host_resident_bytes = c.hostResident;
}

void drainPending() {
  auto& c = C();
  refreshManaged(c);
  // Re-send when either resident estimate moves by 1 MiB; smaller jitter is not worth a frame.
  const uint64_t movedGpu =
      c.resident > c.residentSent ? c.resident - c.residentSent : c.residentSent - c.resident;
  const uint64_t movedHost = c.hostResident > c.hostResidentSent ? c.hostResident - c.hostResidentSent
                                                                 : c.hostResidentSent - c.hostResident;
  const bool residencyOn = c.residencySrc != ResidencySrc::None;
  const bool gaugeDirty =
      c.managedKnown &&
      (!c.managedSentValid || c.managedSent != c.managedBytes ||
       (residencyOn && (movedGpu >= (1u << 20) || movedHost >= (1u << 20))));
  Pending local;
  std::map<AggKey, uint64_t> agg;
  {
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.agg.empty() && c.pending.uvm.empty() && c.pending.allocs.empty() && c.pending.kernels.empty() &&
        c.pending.dropped == 0) {
      std::lock_guard<std::mutex> sendLock(c.sendMu);
      if (c.unsentDrops == 0 && !gaugeDirty) {
        return;
      }
    }
    agg.swap(c.agg);
    local.uvm.swap(c.pending.uvm);
    local.allocs.swap(c.pending.allocs);
    local.kernels.swap(c.pending.kernels);
    local.dropped = c.pending.dropped;
    c.pending.dropped = 0;
  }
  local.uvm.reserve(local.uvm.size() + agg.size());
  for (const auto& item : agg) {
    uvm::UvmEvent ev;
    ev.kind = static_cast<uvm::EventKind>(item.first.kind);
    ev.cause = static_cast<uvm::MigrationCause>(item.first.cause);
    ev.src_id = item.first.src;
    ev.dst_id = item.first.dst;
    ev.device_id = item.first.device;
    ev.address = item.first.addr;
    ev.bytes = item.second;
    local.uvm.push_back(ev);
  }
  uint64_t extra = 0;
  {
    std::lock_guard<std::mutex> lock(c.sendMu);
    extra = c.unsentDrops;
    c.unsentDrops = 0;
  }
  local.dropped += extra;
  const uint64_t eventCount = local.uvm.size() + local.allocs.size() + local.kernels.size();
  if (eventCount == 0 && local.dropped == 0 && !gaugeDirty) {
    return;
  }
  uvm::EventBatch batch;
  batch.uvm.swap(local.uvm);
  batch.allocs.swap(local.allocs);
  batch.kernels.swap(local.kernels);
  batch.dropped = local.dropped;
  batch.managed_bytes = c.managedBytes;
  batch.managed_known = c.managedKnown;
  if (c.residencySrc != ResidencySrc::None) {
    batch.gpu_resident_bytes = c.resident;
    batch.host_resident_bytes = c.hostResident;
  }
  uvm::Buf env;
  uvm::encodeBatch(env, batch);
  // On failure the whole frame is counted as dropped, including records and
  // any drop total already folded into this frame.
  if (enqueueFrame(c, uvm::frame(env), eventCount + local.dropped) && c.managedKnown) {
    c.managedSent = c.managedBytes;
    c.managedSentValid = true;
    c.residentSent = c.resident;
    c.hostResidentSent = c.hostResident;
  }
}

void flushSendQueue() {
  auto& c = C();
  std::lock_guard<std::mutex> lock(c.sendMu);
  if (!c.helloReady) {
    return;
  }
  if (c.fd < 0 && !c.queue.empty()) {
    if (!connectSocket(c)) {
      return;
    }
  }
  while (!c.queue.empty() && c.fd >= 0) {
    const std::string& frame = c.queue.front();
    if (!writeAll(c.fd, frame.data(), frame.size())) {
      ::close(c.fd);
      c.fd = -1;
      return;
    }
    c.queue.pop_front();
  }
}

bool configureCounters(const char** why);

bool cuptiActive(const Collector& c) {
  return c.enableEvents || c.enableMemcpy || c.enableKernel;
}

void workerMain() {
  auto& c = C();
  // Let the driver finish InitializeInjection before touching libcuda.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  if (!c.disabled.load()) {
    fillHello();
  }
  // Unified-memory counters return NOT_READY until a CUDA context exists.
  // InitializeInjection runs before that, so retry from this thread.
  bool uvmOn = false;
  int uvmTries = 0;
  const char* uvmWhy = "";
  while (!c.stop.load()) {
    if (c.enableEvents && !uvmOn && !c.disabled.load() && uvmTries < 100) {
      const char* why = "";
      uvmOn = configureCounters(&why);
      ++uvmTries;
      if (uvmOn) {
        logf("unified memory counters enabled");
      } else if (uvmTries == 100) {
        logf("unified memory counters unavailable: %s", why != nullptr ? why : uvmWhy);
      } else if (why != nullptr) {
        uvmWhy = why;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(c.flushMs));
    if (c.disabled.load()) {
      break;
    }
    if (cuptiActive(c)) {
      cuptiActivityFlushAll(0);
      drainBuffers();
    }
    drainPending();
    flushSendQueue();
  }
  if (!c.disabled.load()) {
    if (cuptiActive(c)) {
      cuptiActivityFlushAll(0);
      drainBuffers();
    }
    drainPending();
    flushSendQueue();
  }
}

void shutdownCollector() {
  auto& c = C();
  // After fork only the calling thread exists. Joining the copied std::thread
  // would wait for a thread the child does not have.
  if (c.forked.load()) {
    return;
  }
  c.stop.store(true);
  if (c.worker.joinable()) {
    c.worker.join();
  }
  if (c.fd >= 0) {
    ::close(c.fd);
    c.fd = -1;
  }
}

bool configureCounters(const char** why) {
  CUpti_ActivityUnifiedMemoryCounterConfig cfg[8];
  uint32_t n = 0;
  auto add = [&](CUpti_ActivityUnifiedMemoryCounterKind kind) {
    cfg[n].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
    cfg[n].kind = kind;
    cfg[n].deviceId = 0;
    cfg[n].enable = 1;
    ++n;
  };
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD);
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH);
#ifdef CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD);
#endif
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT);
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT);
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING);
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING);
  add(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP);

  const CUptiResult st = cuptiActivityConfigureUnifiedMemoryCounter(cfg, n);
  if (st != CUPTI_SUCCESS) {
    *why = cuptiErr(st);
    return false;
  }
  const CUptiResult en = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
  if (en != CUPTI_SUCCESS) {
    *why = cuptiErr(en);
    return false;
  }
  return true;
}

void enableActivities() {
  auto& c = C();
  if (!cuptiActive(c)) {
    logf("CUPTI activities skipped (events=%d memcpy=%d kernel=%d)", c.enableEvents ? 1 : 0,
         c.enableMemcpy ? 1 : 0, c.enableKernel ? 1 : 0);
    return;
  }
  CUptiResult st = CUPTI_SUCCESS;
  if (c.enableEvents) {
    st = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
    if (st != CUPTI_SUCCESS) {
      logf("enable MEMORY2: %s", cuptiErr(st));
    }
  }
  if (c.enableKernel) {
    // CONCURRENT_KERNEL keeps kernel launches concurrent. KIND_KERNEL serializes them.
    st = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (st != CUPTI_SUCCESS) {
      logf("enable CONCURRENT_KERNEL: %s", cuptiErr(st));
    }
  }
  if (c.enableMemcpy) {
    st = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    if (st != CUPTI_SUCCESS) {
      logf("enable MEMCPY: %s", cuptiErr(st));
    }
  }
  st = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  if (st != CUPTI_SUCCESS) {
    logf("register activity callbacks: %s", cuptiErr(st));
  }
}

void loadConfig() {
  auto& c = C();
  if (const char* path = std::getenv("UVMI_SOCKET")) {
    if (path[0] != '\0') {
      c.socketPath = path;
    }
  }
  c.flushMs = envInt("UVMI_FLUSH_MS", 200);
  if (c.flushMs < 10) {
    c.flushMs = 10;
  }
  c.enableEvents = envEnabled("UVMI_EVENTS", true);
  // Memcpy/kernel only make sense with the event pipeline; force off when events are off.
  c.enableMemcpy = c.enableEvents && envEnabled("UVMI_ENABLE_MEMCPY", true);
  c.enableKernel = c.enableEvents && envEnabled("UVMI_ENABLE_KERNEL", true);
  c.quiet = envEnabled("UVMI_QUIET", false);
  c.residencyWanted = ResidencyWanted::Auto;
  if (const char* mode = std::getenv("UVMI_RESIDENCY")) {
    if (std::strcmp(mode, "off") == 0 || std::strcmp(mode, "0") == 0 || std::strcmp(mode, "none") == 0) {
      c.residencyWanted = ResidencyWanted::Off;
    } else if (std::strcmp(mode, "driver") == 0) {
      c.residencyWanted = ResidencyWanted::Driver;
    } else if (std::strcmp(mode, "delta") == 0) {
      c.residencyWanted = ResidencyWanted::Delta;
    } else if (std::strcmp(mode, "auto") == 0) {
      c.residencyWanted = ResidencyWanted::Auto;
    }
  }
  const int summaryMiB = envInt("UVMI_LOC_SUMMARY_MIB", 64);
  const int sampleMiB = envInt("UVMI_LOC_SAMPLE_MIB", 2);
  if (summaryMiB > 0) {
    c.locSummaryBytes = static_cast<size_t>(summaryMiB) << 20;
  }
  if (sampleMiB >= 0) {
    c.locSampleBytes = static_cast<size_t>(sampleMiB) << 20;
  }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int InitializeInjection() {
  auto& c = C();
  if (c.started) {
    return 1;
  }
  c.started = true;
  loadConfig();
  uint32_t api = 0;
  cuptiGetVersion(&api);
  c.hello.cupti_api_version = api;
  if (api < kMinCuptiApi) {
    logf("libcupti API version %u is older than CUDA 12.0; collector disabled", api);
    c.disabled.store(true);
    return 1;
  }
  Dl_info where{};
  if (dladdr(reinterpret_cast<void*>(&cuptiGetVersion), &where) != 0 && where.dli_fname != nullptr) {
    logf("using %s (CUPTI API %u)", where.dli_fname, api);
  }
  initPool();
  enableActivities();
  c.worker = std::thread(workerMain);
  std::atexit(shutdownCollector);
  pthread_atfork(nullptr, nullptr, [] {
    C().disabled.store(true);
    C().forked.store(true);
  });
  logf("collector attached pid=%d flush=%dms", ::getpid(), c.flushMs);
  return 1;
}
