package mock

import (
	"context"
	"math"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

const (
	trainPID = 4242
	inferPID = 4243
	stepDur  = 200 * time.Millisecond
)

// Run fills about 90 seconds of history, then keeps emitting a live demo
// workload so the UI can be exercised on a machine without a GPU.
func Run(ctx context.Context, st *aggregate.Store) {
	base := time.Now().Add(-90 * time.Second)
	steps := int(90 * time.Second / stepDur)
	st.Register(trainHello())
	st.IngestAt(trainPID, protocol.EventBatch{Allocs: []protocol.AllocEvent{{
		Address: 0x7f0000000000, Size: 6 << 30, IsManaged: true, TimestampNs: 1,
	}}}, base)
	for step := 0; step < steps; step++ {
		at := base.Add(time.Duration(step) * stepDur)
		st.IngestAt(trainPID, trainBatch(step, at), at)
	}
	st.Register(inferHello())
	st.IngestAt(inferPID, protocol.EventBatch{Allocs: []protocol.AllocEvent{{
		Address: 0x7f1000000000, Size: 512 << 20, IsManaged: true,
	}}}, time.Now())

	go func() {
		ticker := time.NewTicker(stepDur)
		defer ticker.Stop()
		step := steps
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-ticker.C:
				st.IngestAt(trainPID, trainBatch(step, now), now)
				st.IngestAt(inferPID, inferBatch(step, now), now)
				step++
			}
		}
	}()
}

func trainHello() protocol.Hello {
	return protocol.Hello{
		PID: trainPID, Comm: "python", AppName: "train.py", Hostname: "devbox",
		CudaVersion: "12.9", DriverVersion: "12.9",
		GPUs: []protocol.GpuInfo{
			{DeviceID: 0, UUID: "GPU-a100-0", Name: "NVIDIA A100-SXM4-80GB", TotalMemoryBytes: 80 << 30},
			{DeviceID: 1, UUID: "GPU-a100-1", Name: "NVIDIA A100-SXM4-80GB", TotalMemoryBytes: 80 << 30},
		},
	}
}

func inferHello() protocol.Hello {
	return protocol.Hello{
		PID: inferPID, Comm: "python", AppName: "infer.py", Hostname: "devbox",
		CudaVersion: "12.9", DriverVersion: "12.9",
		GPUs: []protocol.GpuInfo{
			{DeviceID: 0, UUID: "GPU-a100-0", Name: "NVIDIA A100-SXM4-80GB", TotalMemoryBytes: 80 << 30},
		},
	}
}

func trainBatch(step int, at time.Time) protocol.EventBatch {
	phase := float64(step) / 8
	wave := math.Sin(phase)
	addr := uint64(0x7f0000000000) + uint64(step%8)*(2<<20)
	hot := uint64(0x7f0000000000)
	var b protocol.EventBatch
	b.Uvm = append(b.Uvm,
		mig(protocol.KindBytesHtoD, mb(8+6*wave), addr, 0xffffffff, 0, protocol.CauseCoherence, at),
		mig(protocol.KindBytesDtoH, mb(3+2*wave), addr+(2<<20), 0, 0xffffffff, protocol.CauseCoherence, at),
		mig(protocol.KindBytesHtoD, mb(1.5), addr+(4<<20), 0xffffffff, 1, protocol.CausePrefetch, at),
	)
	if step%10 == 0 {
		b.Uvm = append(b.Uvm, mig(protocol.KindBytesHtoD, mb(48), hot, 0xffffffff, 0, protocol.CauseUser, at))
	}
	if step%25 == 0 {
		b.Uvm = append(b.Uvm, mig(protocol.KindBytesDtoH, mb(16), hot+(8<<20), 0, 0xffffffff, protocol.CauseEviction, at))
	}
	if step%12 == 0 {
		b.Uvm = append(b.Uvm, mig(protocol.KindBytesDtoD, mb(8), hot, 0, 1, protocol.CauseAccessCounters, at))
	}
	if step%20 == 0 {
		b.Uvm = append(b.Uvm, mig(protocol.KindMemcpyHtoD, mb(32), 0, 0xffffffff, 0, protocol.CauseUnknown, at))
	}
	end := uint64(at.UnixNano())
	b.Uvm = append(b.Uvm,
		protocol.UvmEvent{Kind: protocol.KindCPUPageFault, Address: addr, DeviceID: 0xffffffff, EndNs: end},
		protocol.UvmEvent{Kind: protocol.KindGPUPageFault, Bytes: 1 + uint64(step%3), Address: addr, DeviceID: 0, SrcID: 0, EndNs: end},
	)
	if wave > 0.82 {
		b.Uvm = append(b.Uvm,
			protocol.UvmEvent{Kind: protocol.KindThrashing, Bytes: 2 << 20, Address: hot, DeviceID: 0, SrcID: 0, EndNs: end},
			protocol.UvmEvent{Kind: protocol.KindThrottling, DeviceID: 0, SrcID: 0, EndNs: end},
		)
	}
	if step%18 == 0 {
		b.Uvm = append(b.Uvm, protocol.UvmEvent{Kind: protocol.KindRemoteMap, Bytes: 8192, Address: addr, DeviceID: 1, DstID: 1, EndNs: end})
	}
	b.Kernels = []protocol.KernelEvent{{
		Name: "gemm_nt", DeviceID: 0, StreamID: 7, CorrelationID: uint32(step),
		StartNs: end - 18_000_000, EndNs: end,
	}}
	if step%7 == 0 {
		b.Kernels = append(b.Kernels, protocol.KernelEvent{
			Name: "allreduce", DeviceID: 1, StartNs: end - 6_000_000, EndNs: end,
		})
	}
	return b
}

func inferBatch(step int, at time.Time) protocol.EventBatch {
	addr := uint64(0x7f1000000000) + uint64(step%4)*(2<<20)
	wave := math.Sin(float64(step) / 6)
	return protocol.EventBatch{Uvm: []protocol.UvmEvent{
		mig(protocol.KindBytesHtoD, mb(2+wave), addr, 0xffffffff, 0, protocol.CausePrefetch, at),
		mig(protocol.KindBytesDtoH, mb(1), addr, 0, 0xffffffff, protocol.CauseCoherence, at),
	}, Kernels: []protocol.KernelEvent{{
		Name: "decode_step", DeviceID: 0, StartNs: uint64(at.UnixNano()) - 4_000_000, EndNs: uint64(at.UnixNano()),
	}}}
}

func mig(kind protocol.EventKind, bytes, addr uint64, src, dst uint32, cause protocol.Cause, at time.Time) protocol.UvmEvent {
	end := uint64(at.UnixNano())
	start := end - 250_000
	return protocol.UvmEvent{
		Kind: kind, Bytes: bytes, Address: addr, SrcID: src, DstID: dst, Cause: cause,
		DeviceID: dst, StartNs: start, EndNs: end,
	}
}

func mb(v float64) uint64 {
	if v < 0.2 {
		v = 0.2
	}
	return uint64(v * 1_000_000)
}
