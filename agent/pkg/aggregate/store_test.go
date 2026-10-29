package aggregate

import (
	"testing"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

func TestMigrationAndManagedMemory(t *testing.T) {
	s := New(time.Minute)
	now := time.Unix(1_700_000_000, 500*int64(time.Millisecond))
	s.SetClock(func() time.Time { return now })
	s.Register(protocol.Hello{
		PID: 7, AppName: "app", Comm: "app",
		GPUs: []protocol.GpuInfo{{DeviceID: 0, Name: "A100", TotalMemoryBytes: 80 << 30}},
	})
	s.IngestAt(7, protocol.EventBatch{
		Allocs: []protocol.AllocEvent{{Address: 0x1000, Size: 4096, IsManaged: true}},
		Uvm: []protocol.UvmEvent{{
			Kind: protocol.KindBytesHtoD, Bytes: 2_000_000, StartNs: 1, EndNs: 1_000_000,
			Address: 0x200000, SrcID: 0xffffffff, DstID: 0, Cause: protocol.CauseCoherence, DeviceID: 0,
		}},
	}, now)

	snap := s.Snapshot(7)
	if snap.ManagedBytes != 4096 {
		t.Fatalf("managed = %d", snap.ManagedBytes)
	}
	if len(snap.Flows) != 1 {
		t.Fatalf("flows: %+v", snap.Flows)
	}
	f := snap.Flows[0]
	if f.Src != "host" || f.Dst != "gpu0" || f.Direction != "htod" || f.Cause != "coherence" {
		t.Fatalf("flow: %+v", f)
	}
	if f.MBps < 3.9 || f.MBps > 4.1 {
		t.Fatalf("mbps = %v", f.MBps)
	}
	if len(snap.Nodes) < 2 {
		t.Fatalf("nodes: %+v", snap.Nodes)
	}

	s.IngestAt(7, protocol.EventBatch{
		Allocs: []protocol.AllocEvent{{Address: 0x1000, Size: 4096, IsManaged: true, IsFree: true}},
	}, now)
	if got := s.Snapshot(7).ManagedBytes; got != 0 {
		t.Fatalf("managed after free = %d", got)
	}

	prom := s.Prom()
	if len(prom.Migrated) != 1 || prom.Migrated[0].Value != 2_000_000 {
		t.Fatalf("prom: %+v", prom.Migrated)
	}
	if len(prom.Durations) != 1 || prom.Durations[0].Count != 1 {
		t.Fatalf("hist: %+v", prom.Durations)
	}
	heat := s.Heatmap(7)
	if len(heat.Cells) != 1 || heat.Cells[0].Bytes != 2_000_000 {
		t.Fatalf("heat: %+v", heat.Cells)
	}
}
