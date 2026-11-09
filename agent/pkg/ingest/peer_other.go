//go:build !linux

package ingest

import "net"

func peerPID(net.Conn) uint32 { return 0 }
