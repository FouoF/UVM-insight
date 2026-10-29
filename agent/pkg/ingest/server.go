package ingest

import (
	"context"
	"net"
	"os"
	"path/filepath"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

// Serve accepts collector connections on a Unix socket. Each connection starts
// with Hello and then carries EventBatch frames. Closing the socket marks the
// process exited. Cancel ctx to stop listening.
func Serve(ctx context.Context, path string, st *aggregate.Store) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return err
	}
	// The socket has to be reachable by CUDA processes running as other users.
	_ = os.Chmod(dir, 0o777)
	_ = os.Remove(path)
	ln, err := net.Listen("unix", path)
	if err != nil {
		return err
	}
	_ = os.Chmod(path, 0o777)
	go func() {
		<-ctx.Done()
		_ = ln.Close()
	}()
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				_ = os.Remove(path)
				return nil
			}
			continue
		}
		go handle(conn, st)
	}
}

func handle(conn net.Conn, st *aggregate.Store) {
	defer conn.Close()
	var pid uint32
	var seen bool
	defer func() {
		if seen {
			st.MarkExited(pid)
		}
	}()
	for {
		env, err := protocol.ReadFrame(conn)
		if err != nil {
			return
		}
		if env.Hello != nil {
			pid = env.Hello.PID
			seen = true
			st.Register(*env.Hello)
		}
		if env.Batch != nil && seen {
			st.Ingest(pid, *env.Batch)
		}
	}
}
