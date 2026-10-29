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
	// 2 MB over a 1 s window whose newest slot has just started (0.9 s elapsed).
	if f.MBps < 2.2 || f.MBps > 2.25 {
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

func TestManagedGaugeSurvivesUnknownFree(t *testing.T) {
	s := New(time.Minute)
	now := time.Unix(1_700_000_000, 0)
	s.SetClock(func() time.Time { return now })
	s.Register(protocol.Hello{PID: 7, AppName: "app", ManagedBytes: 3 << 30, ManagedKnown: true})
	if got := s.Snapshot(7).ManagedBytes; got != 3<<30 {
		t.Fatalf("from hello = %d", got)
	}
	s.IngestAt(7, protocol.EventBatch{
		ManagedBytes: 1500 << 20,
		ManagedKnown: true,
		Allocs:       []protocol.AllocEvent{{Address: 0xabc, Size: 1 << 30, IsManaged: true, IsFree: true}},
	}, now)
	if got := s.Snapshot(7).ManagedBytes; got != 1500<<20 {
		t.Fatalf("gauge = %d", got)
	}
	s.IngestAt(7, protocol.EventBatch{ManagedBytes: 1500 << 20, ManagedKnown: true, ResidentBytes: 1 << 30}, now.Add(-3*time.Second))
	s.IngestAt(7, protocol.EventBatch{ManagedBytes: 1500 << 20, ManagedKnown: true, ResidentBytes: 200 << 20}, now)
	ov := s.Overview(0, true)
	var live ProcLive
	for _, p := range ov.Processes {
		if p.PID == 7 {
			live = p
		}
	}
	if !live.ResidentKnown || live.ResidentBytes != 200<<20 {
		t.Fatalf("resident: %+v", live)
	}
	n := len(live.Series)
	if live.Series[n-2].ResidentMB != float64(1<<30)/1e6 || live.Series[n-1].ResidentMB != float64(200<<20)/1e6 {
		t.Fatalf("carry forward: %v %v", live.Series[n-2].ResidentMB, live.Series[n-1].ResidentMB)
	}

	s.Register(protocol.Hello{PID: 8, AppName: "old"})
	s.IngestAt(8, protocol.EventBatch{
		Allocs: []protocol.AllocEvent{{Address: 0x99, Size: 8 << 20, IsManaged: true, IsFree: true}},
	}, now)
	if got := s.Snapshot(8).ManagedBytes; got != 0 {
		t.Fatalf("unknown free changed managed to %d", got)
	}
}
