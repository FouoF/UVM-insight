package protocol

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestCppFixture(t *testing.T) {
	path := filepath.Join("..", "..", "..", "proto", "testdata", "sample.frames")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r := bytes.NewReader(raw)
	helloEnv, err := ReadFrame(r)
	if err != nil {
		t.Fatal(err)
	}
	if helloEnv.Hello == nil {
		t.Fatal("missing hello")
	}
	h := helloEnv.Hello
	if h.PID != 42 || h.Comm != "vec" || h.AppName != "vecAdd" || h.Hostname != "n1" {
		t.Fatalf("hello identity: %+v", h)
	}
	if h.CudaVersion != "12.8" || h.DriverVersion != "12.8" {
		t.Fatalf("versions: %+v", h)
	}
	if len(h.GPUs) != 1 || h.GPUs[0].Name != "A100" || h.GPUs[0].UUID != "GPU-abc" || h.GPUs[0].TotalMemoryBytes != 80<<30 {
		t.Fatalf("gpu: %+v", h.GPUs)
	}

	batchEnv, err := ReadFrame(r)
	if err != nil {
		t.Fatal(err)
	}
	if batchEnv.Batch == nil {
		t.Fatal("missing batch")
	}
	b := batchEnv.Batch
	if len(b.Uvm) != 1 || len(b.Allocs) != 1 || len(b.Kernels) != 1 || b.Dropped != 3 {
		t.Fatalf("batch shape: %+v", b)
	}
	ev := b.Uvm[0]
	if ev.Kind != KindBytesHtoD || ev.Bytes != 4096 || ev.StartNs != 1000 || ev.EndNs != 2500 {
		t.Fatalf("uvm: %+v", ev)
	}
	if ev.Address != 0x7f00000000 || ev.SrcID != 0xffffffff || ev.DstID != 0 || ev.Cause != CauseCoherence {
		t.Fatalf("uvm ends: %+v", ev)
	}
	if ev.StreamID != 7 || ev.CorrelationID != 9 || ev.DeviceID != 0 {
		t.Fatalf("uvm ids: %+v", ev)
	}
	if !b.Allocs[0].IsManaged || b.Allocs[0].IsFree || b.Allocs[0].Size != 1<<20 || b.Allocs[0].TimestampNs != 900 {
		t.Fatalf("alloc: %+v", b.Allocs[0])
	}
	if b.Kernels[0].Name != "touch" || b.Kernels[0].StartNs != 1100 || b.Kernels[0].EndNs != 2400 || b.Kernels[0].StreamID != 7 {
		t.Fatalf("kernel: %+v", b.Kernels[0])
	}
	if r.Len() != 0 {
		t.Fatalf("trailing bytes: %d", r.Len())
	}
}

func TestRoundTrip(t *testing.T) {
	env := Envelope{Batch: &EventBatch{
		Uvm: []UvmEvent{{
			Kind: KindBytesDtoH, Bytes: 8192, Cause: CauseEviction, DeviceID: 1, SrcID: 1, DstID: 0xffffffff,
		}},
		Dropped: 2,
	}}
	var buf bytes.Buffer
	if err := WriteFrame(&buf, env); err != nil {
		t.Fatal(err)
	}
	got, err := ReadFrame(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.Batch == nil || len(got.Batch.Uvm) != 1 || got.Batch.Uvm[0].Bytes != 8192 || got.Batch.Uvm[0].Cause != CauseEviction || got.Batch.Dropped != 2 {
		t.Fatalf("%+v", got.Batch)
	}
}
