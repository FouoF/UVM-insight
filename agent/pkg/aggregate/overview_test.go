package aggregate

import (
	"testing"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

func TestOverviewLanesPerProcess(t *testing.T) {
	s := New(time.Minute)
	now := time.Unix(1_700_000_000, 0)
	s.SetClock(func() time.Time { return now })
	gpu := func(limit uint64) []protocol.GpuInfo {
		return []protocol.GpuInfo{{DeviceID: 0, UUID: "GPU-a", Name: "Tesla P4", TotalMemoryBytes: limit}}
	}
	s.Register(protocol.Hello{PID: 11, AppName: "esm2", GPUs: gpu(4 << 30)})
	s.Register(protocol.Hello{PID: 12, AppName: "biogpt", GPUs: gpu(6 << 30)})

	htod := protocol.UvmEvent{Kind: protocol.KindBytesHtoD, Bytes: 3_000_000, SrcID: cpuID, DstID: 0, DeviceID: 0}
	dtoh := protocol.UvmEvent{Kind: protocol.KindBytesDtoH, Bytes: 6_000_000, SrcID: 0, DstID: cpuID, DeviceID: 0}
	s.IngestAt(11, protocol.EventBatch{Uvm: []protocol.UvmEvent{htod}}, now.Add(-800*time.Millisecond))
	s.IngestAt(12, protocol.EventBatch{Uvm: []protocol.UvmEvent{dtoh}}, now.Add(-200*time.Millisecond))
	s.IngestAt(12, protocol.EventBatch{Uvm: []protocol.UvmEvent{dtoh}}, now.Add(-1500*time.Millisecond))

	ov := s.Overview(0, true)
	if len(ov.GPUs) != 1 || ov.GPUs[0].ID != "GPU-a" {
		t.Fatalf("gpus: %+v", ov.GPUs)
	}
	if len(ov.Processes) != 2 {
		t.Fatalf("processes: %+v", ov.Processes)
	}
	bio, esm := ov.Processes[0], ov.Processes[1]
	if bio.AppName != "biogpt" || esm.AppName != "esm2" {
		t.Fatalf("order: %s %s", bio.AppName, esm.AppName)
	}
	// The window ends at a whole second, so 0.9 s has elapsed; the record from
	// 1.5 s ago has already left it.
	if esm.HtoD < 3.33 || esm.HtoD > 3.34 || bio.DtoH < 6.66 || bio.DtoH > 6.67 {
		t.Fatalf("rates: esm %+v bio %+v", esm, bio)
	}
	if len(esm.Lanes) != 1 || esm.Lanes[0].GPU != "GPU-a" || esm.Lanes[0].QuotaBytes != 4<<30 {
		t.Fatalf("lanes: %+v", esm.Lanes)
	}
	if len(esm.Series) != seriesSeconds {
		t.Fatalf("series: %d", len(esm.Series))
	}
	if len(ov.Events) != 3 || ov.Events[0].Seq == ov.Events[1].Seq || ov.Events[0].App != "esm2" {
		t.Fatalf("events: %+v", ov.Events)
	}

	next := s.Overview(ov.LastSeq, false)
	if len(next.Events) != 0 || len(next.Processes[0].Series) != 2 {
		t.Fatalf("incremental: %+v", next)
	}

	s.MarkExited(12)
	now = now.Add(3 * time.Minute)
	if got := s.Overview(0, false).Processes; len(got) != 1 || got[0].AppName != "esm2" {
		t.Fatalf("after exit: %+v", got)
	}
}
