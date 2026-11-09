package ingest

import (
	"net"

	"golang.org/x/sys/unix"
)

// peerPID returns the connecting process as seen from the agent's pid
// namespace. It is 0 when that process is not visible there.
func peerPID(conn net.Conn) uint32 {
	uc, ok := conn.(*net.UnixConn)
	if !ok {
		return 0
	}
	raw, err := uc.SyscallConn()
	if err != nil {
		return 0
	}
	var pid int32
	_ = raw.Control(func(fd uintptr) {
		cred, err := unix.GetsockoptUcred(int(fd), unix.SOL_SOCKET, unix.SO_PEERCRED)
		if err == nil {
			pid = cred.Pid
		}
	})
	if pid <= 0 {
		return 0
	}
	return uint32(pid)
}
