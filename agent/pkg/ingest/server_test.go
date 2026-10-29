package ingest

import (
	"context"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

func TestSocketRoundTrip(t *testing.T) {
	st := aggregate.New(time.Minute)
	sock := filepath.Join(t.TempDir(), "agent.sock")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- Serve(ctx, sock, st) }()

	var conn net.Conn
	var err error
	for i := 0; i < 50; i++ {
		conn, err = net.Dial("unix", sock)
		if err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err != nil {
		t.Fatal(err)
	}
	hello := protocol.Envelope{Hello: &protocol.Hello{PID: 9, AppName: "demo", Comm: "demo"}}
	if err := protocol.WriteFrame(conn, hello); err != nil {
		t.Fatal(err)
	}
	batch := protocol.Envelope{Batch: &protocol.EventBatch{
		Uvm: []protocol.UvmEvent{{
			Kind: protocol.KindBytesDtoH, Bytes: 1024, SrcID: 0, DstID: 0xffffffff, Cause: protocol.CauseUser,
		}},
		Dropped: 4,
	}}
	if err := protocol.WriteFrame(conn, batch); err != nil {
		t.Fatal(err)
	}
	_ = conn.Close()

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		procs := st.Processes()
		if len(procs) == 1 && procs[0].Dropped == 4 && !procs[0].Active {
			prom := st.Prom()
			if len(prom.Migrated) == 1 && prom.Migrated[0].Value == 1024 {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("processes: %+v", st.Processes())
}
