package protocol

// Event kinds match proto/uvm_event.proto.
const (
	KindUnspecified  EventKind = 0
	KindBytesHtoD    EventKind = 1
	KindBytesDtoH    EventKind = 2
	KindBytesDtoD    EventKind = 3
	KindCPUPageFault EventKind = 4
	KindGPUPageFault EventKind = 5
	KindThrashing    EventKind = 6
	KindThrottling   EventKind = 7
	KindRemoteMap    EventKind = 8
	KindMemcpyHtoD   EventKind = 9
	KindMemcpyDtoH   EventKind = 10
	KindMemcpyDtoD   EventKind = 11
	KindMemcpyOther  EventKind = 12
)

// Migration causes match CUpti_ActivityUnifiedMemoryMigrationCause.
const (
	CauseUnknown        Cause = 0
	CauseUser           Cause = 1
	CauseCoherence      Cause = 2
	CausePrefetch       Cause = 3
	CauseEviction       Cause = 4
	CauseAccessCounters Cause = 5
)

type EventKind int32

type Cause int32

type GpuInfo struct {
	DeviceID         uint32 `json:"deviceId"`
	UUID             string `json:"uuid"`
	Name             string `json:"name"`
	TotalMemoryBytes uint64 `json:"totalMemoryBytes"`
}

type Hello struct {
	PID           uint32
	Comm          string
	AppName       string
	Hostname      string
	CudaVersion   string
	DriverVersion string
	GPUs          []GpuInfo
}

type UvmEvent struct {
	Kind          EventKind
	Bytes         uint64
	StartNs       uint64
	EndNs         uint64
	Address       uint64
	SrcID         uint32
	DstID         uint32
	Cause         Cause
	StreamID      uint32
	CorrelationID uint32
	DeviceID      uint32
}

type AllocEvent struct {
	Address     uint64
	Size        uint64
	IsManaged   bool
	IsFree      bool
	TimestampNs uint64
	DeviceID    uint32
}

type KernelEvent struct {
	Name          string
	StartNs       uint64
	EndNs         uint64
	DeviceID      uint32
	CorrelationID uint32
	StreamID      uint32
}

type EventBatch struct {
	Uvm     []UvmEvent
	Allocs  []AllocEvent
	Kernels []KernelEvent
	Dropped uint64
}

type Envelope struct {
	Hello *Hello
	Batch *EventBatch
}
