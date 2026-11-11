#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Hand-written protobuf encoder for proto/uvm_event.proto.
// Field numbers must stay aligned with that file and the Go decoder.

namespace uvm {

enum class EventKind : uint32_t {
  kUnspecified = 0,
  kBytesHtoD = 1,
  kBytesDtoH = 2,
  kBytesDtoD = 3,
  kCpuPageFault = 4,
  kGpuPageFault = 5,
  kThrashing = 6,
  kThrottling = 7,
  kRemoteMap = 8,
  kMemcpyHtoD = 9,
  kMemcpyDtoH = 10,
  kMemcpyDtoD = 11,
  kMemcpyOther = 12,
};

enum class MigrationCause : uint32_t {
  kUnknown = 0,
  kUser = 1,
  kCoherence = 2,
  kPrefetch = 3,
  kEviction = 4,
  kAccessCounters = 5,
};

struct GpuInfo {
  uint32_t device_id = 0;
  std::string uuid;
  std::string name;
  uint64_t total_memory_bytes = 0;
};

struct Hello {
  uint32_t pid = 0;
  std::string comm;
  std::string app_name;
  std::string hostname;
  std::string cuda_version;
  std::string driver_version;
  std::vector<GpuInfo> gpus;
  uint64_t managed_bytes = 0;
  bool managed_known = false;
  uint64_t gpu_resident_bytes = 0;
  std::string pod_uid;
  std::string container_name;
  // cuptiGetVersion() of the libcupti this process actually loaded.
  uint32_t cupti_api_version = 0;
  // Estimated part of managed_bytes resident on the host. With
  // gpu_resident_bytes, unknown = managed - gpu - host.
  uint64_t host_resident_bytes = 0;
  // "none" | "driver" | "delta" — how gpu/host resident gauges are produced.
  std::string residency_source = "none";
};

struct UvmEvent {
  EventKind kind = EventKind::kUnspecified;
  uint64_t bytes = 0;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint64_t address = 0;
  uint32_t src_id = 0;
  uint32_t dst_id = 0;
  MigrationCause cause = MigrationCause::kUnknown;
  uint32_t stream_id = 0;
  uint32_t correlation_id = 0;
  uint32_t device_id = 0;
};

struct AllocEvent {
  uint64_t address = 0;
  uint64_t size = 0;
  bool is_managed = false;
  bool is_free = false;
  uint64_t timestamp_ns = 0;
  uint32_t device_id = 0;
};

struct KernelEvent {
  std::string name;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint32_t device_id = 0;
  uint32_t correlation_id = 0;
  uint32_t stream_id = 0;
};

struct EventBatch {
  std::vector<UvmEvent> uvm;
  std::vector<AllocEvent> allocs;
  std::vector<KernelEvent> kernels;
  uint64_t dropped = 0;
  uint64_t managed_bytes = 0;
  bool managed_known = false;
  uint64_t gpu_resident_bytes = 0;
  uint64_t host_resident_bytes = 0;
};

// One /dev/nvidia-uvm mapping (excluding nvidia-uvm-tools).
struct UvmMapRange {
  uint64_t start = 0;
  uint64_t size = 0;
};

// Lists live nvidia-uvm VA ranges. Their sizes sum to the process unified
// memory footprint, including allocations made before this collector attached.
std::vector<UvmMapRange> listNvidiaUvmMaps(std::string_view text);

// Sums the virtual size of /dev/nvidia-uvm mappings.
uint64_t sumNvidiaUvmMaps(std::string_view text);

// Finds the kubelet pod directory in /proc/self/mountinfo. Pods get
// /etc/hosts from pods/<uid>/etc-hosts and /dev/termination-log from
// pods/<uid>/containers/<name>/<id>, so this works with cgroup namespaces
// that hide the container id. Returns false outside Kubernetes.
bool parseKubeletMounts(std::string_view mountinfo, std::string* podUid, std::string* container);

class Buf {
 public:
  void varint(uint32_t field, uint64_t v);
  void str(uint32_t field, std::string_view s);
  void msg(uint32_t field, const Buf& nested);
  const std::string& data() const { return buf_; }
  void clear() { buf_.clear(); }

 private:
  void key(uint32_t field, uint32_t wire);
  static void appendVarint(std::string& out, uint64_t v);
  std::string buf_;
};

void encodeHello(Buf& envelope, const Hello& hello);
void encodeBatch(Buf& envelope, const EventBatch& batch);

// uint32 big-endian length prefix + envelope bytes.
std::string frame(const Buf& envelope);

}  // namespace uvm
