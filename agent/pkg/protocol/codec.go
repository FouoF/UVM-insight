package protocol

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

const maxFrame = 4 << 20

var errOverflow = errors.New("protocol: varint overflow")

func ReadFrame(r io.Reader) (Envelope, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return Envelope{}, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n > maxFrame {
		return Envelope{}, fmt.Errorf("protocol: frame is %d bytes", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return Envelope{}, err
	}
	return decodeEnvelope(buf)
}

func WriteFrame(w io.Writer, env Envelope) error {
	payload := encodeEnvelope(env)
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func encodeEnvelope(env Envelope) []byte {
	var b []byte
	if env.Hello != nil {
		b = appendMessage(b, 1, encodeHello(*env.Hello))
	}
	if env.Batch != nil {
		b = appendMessage(b, 2, encodeBatch(*env.Batch))
	}
	return b
}

func encodeHello(h Hello) []byte {
	var b []byte
	b = appendVarintField(b, 1, uint64(h.PID))
	b = appendString(b, 2, h.Comm)
	b = appendString(b, 3, h.AppName)
	b = appendString(b, 4, h.Hostname)
	b = appendString(b, 5, h.CudaVersion)
	b = appendString(b, 6, h.DriverVersion)
	for _, g := range h.GPUs {
		b = appendMessage(b, 7, encodeGPU(g))
	}
	return b
}

func encodeGPU(g GpuInfo) []byte {
	var b []byte
	b = appendVarintField(b, 1, uint64(g.DeviceID))
	b = appendString(b, 2, g.UUID)
	b = appendString(b, 3, g.Name)
	b = appendVarintField(b, 4, g.TotalMemoryBytes)
	return b
}

func encodeBatch(batch EventBatch) []byte {
	var b []byte
	for _, ev := range batch.Uvm {
		b = appendMessage(b, 1, encodeUvm(ev))
	}
	for _, ev := range batch.Allocs {
		b = appendMessage(b, 2, encodeAlloc(ev))
	}
	for _, ev := range batch.Kernels {
		b = appendMessage(b, 3, encodeKernel(ev))
	}
	b = appendVarintField(b, 4, batch.Dropped)
	return b
}

func encodeUvm(ev UvmEvent) []byte {
	var b []byte
	b = appendVarintField(b, 1, uint64(ev.Kind))
	b = appendVarintField(b, 2, ev.Bytes)
	b = appendVarintField(b, 3, ev.StartNs)
	b = appendVarintField(b, 4, ev.EndNs)
	b = appendVarintField(b, 5, ev.Address)
	b = appendVarintField(b, 6, uint64(ev.SrcID))
	b = appendVarintField(b, 7, uint64(ev.DstID))
	b = appendVarintField(b, 8, uint64(ev.Cause))
	b = appendVarintField(b, 9, uint64(ev.StreamID))
	b = appendVarintField(b, 10, uint64(ev.CorrelationID))
	b = appendVarintField(b, 11, uint64(ev.DeviceID))
	return b
}

func encodeAlloc(ev AllocEvent) []byte {
	var b []byte
	b = appendVarintField(b, 1, ev.Address)
	b = appendVarintField(b, 2, ev.Size)
	if ev.IsManaged {
		b = appendVarintField(b, 3, 1)
	}
	if ev.IsFree {
		b = appendVarintField(b, 4, 1)
	}
	b = appendVarintField(b, 5, ev.TimestampNs)
	b = appendVarintField(b, 6, uint64(ev.DeviceID))
	return b
}

func encodeKernel(ev KernelEvent) []byte {
	var b []byte
	b = appendString(b, 1, ev.Name)
	b = appendVarintField(b, 2, ev.StartNs)
	b = appendVarintField(b, 3, ev.EndNs)
	b = appendVarintField(b, 4, uint64(ev.DeviceID))
	b = appendVarintField(b, 5, uint64(ev.CorrelationID))
	b = appendVarintField(b, 6, uint64(ev.StreamID))
	return b
}

func appendVarintField(b []byte, field uint32, v uint64) []byte {
	if v == 0 {
		return b
	}
	b = appendKey(b, field, 0)
	return appendVarint(b, v)
}

func appendString(b []byte, field uint32, s string) []byte {
	if s == "" {
		return b
	}
	b = appendKey(b, field, 2)
	b = appendVarint(b, uint64(len(s)))
	return append(b, s...)
}

func appendMessage(b []byte, field uint32, msg []byte) []byte {
	b = appendKey(b, field, 2)
	b = appendVarint(b, uint64(len(msg)))
	return append(b, msg...)
}

func appendKey(b []byte, field, wire uint32) []byte {
	return appendVarint(b, uint64(field)<<3|uint64(wire))
}

func appendVarint(b []byte, v uint64) []byte {
	for v >= 0x80 {
		b = append(b, byte(v)|0x80)
		v >>= 7
	}
	return append(b, byte(v))
}

func decodeEnvelope(b []byte) (Envelope, error) {
	var env Envelope
	i := 0
	for i < len(b) {
		field, wire, ni, err := readKey(b, i)
		if err != nil {
			return Envelope{}, err
		}
		i = ni
		switch field {
		case 1:
			raw, ni, err := readBytes(b, i)
			if err != nil {
				return Envelope{}, err
			}
			h, err := decodeHello(raw)
			if err != nil {
				return Envelope{}, err
			}
			env.Hello = &h
			i = ni
		case 2:
			raw, ni, err := readBytes(b, i)
			if err != nil {
				return Envelope{}, err
			}
			batch, err := decodeBatch(raw)
			if err != nil {
				return Envelope{}, err
			}
			env.Batch = &batch
			i = ni
		default:
			i, err = skip(b, i, wire)
			if err != nil {
				return Envelope{}, err
			}
		}
	}
	return env, nil
}

func decodeHello(b []byte) (Hello, error) {
	var h Hello
	i := 0
	for i < len(b) {
		field, wire, ni, err := readKey(b, i)
		if err != nil {
			return Hello{}, err
		}
		i = ni
		switch field {
		case 1:
			v, ni, err := readVarint(b, i)
			if err != nil {
				return Hello{}, err
			}
			h.PID = uint32(v)
			i = ni
		case 2:
			h.Comm, i, err = readString(b, i)
		case 3:
			h.AppName, i, err = readString(b, i)
		case 4:
			h.Hostname, i, err = readString(b, i)
		case 5:
			h.CudaVersion, i, err = readString(b, i)
		case 6:
			h.DriverVersion, i, err = readString(b, i)
		case 7:
			raw, ni, err := readBytes(b, i)
			if err != nil {
				return Hello{}, err
			}
			g, err := decodeGPU(raw)
			if err != nil {
				return Hello{}, err
			}
			h.GPUs = append(h.GPUs, g)
			i = ni
		default:
			i, err = skip(b, i, wire)
		}
		if err != nil {
			return Hello{}, err
		}
	}
	return h, nil
}

func decodeGPU(b []byte) (GpuInfo, error) {
	var g GpuInfo
	i := 0
	for i < len(b) {
		field, wire, ni, err := readKey(b, i)
		if err != nil {
			return GpuInfo{}, err
		}
		i = ni
		switch field {
		case 1:
			v, ni, err := readVarint(b, i)
			if err != nil {
				return GpuInfo{}, err
			}
			g.DeviceID = uint32(v)
			i = ni
		case 2:
			g.UUID, i, err = readString(b, i)
		case 3:
			g.Name, i, err = readString(b, i)
		case 4:
			g.TotalMemoryBytes, i, err = readVarint(b, i)
		default:
			i, err = skip(b, i, wire)
		}
		if err != nil {
			return GpuInfo{}, err
		}
	}
	return g, nil
}

func decodeBatch(b []byte) (EventBatch, error) {
	var batch EventBatch
	i := 0
	for i < len(b) {
		field, wire, ni, err := readKey(b, i)
		if err != nil {
			return EventBatch{}, err
		}
		i = ni
		switch field {
		case 1:
			raw, ni, err := readBytes(b, i)
			if err != nil {
				return EventBatch{}, err
			}
			ev, err := decodeUvm(raw)
			if err != nil {
				return EventBatch{}, err
			}
			batch.Uvm = append(batch.Uvm, ev)
			i = ni
		case 2:
			raw, ni, err := readBytes(b, i)
			if err != nil {
				return EventBatch{}, err
			}
			ev, err := decodeAlloc(raw)
			if err != nil {
				return EventBatch{}, err
			}
			batch.Allocs = append(batch.Allocs, ev)
			i = ni
		case 3:
			raw, ni, err := readBytes(b, i)
			if err != nil {
				return EventBatch{}, err
			}
			ev, err := decodeKernel(raw)
			if err != nil {
				return EventBatch{}, err
			}
			batch.Kernels = append(batch.Kernels, ev)
			i = ni
		case 4:
			batch.Dropped, i, err = readVarint(b, i)
		default:
			i, err = skip(b, i, wire)
		}
		if err != nil {
			return EventBatch{}, err
		}
	}
	return batch, nil
}

func decodeUvm(b []byte) (UvmEvent, error) {
	var ev UvmEvent
	i := 0
	for i < len(b) {
		field, wire, ni, err := readKey(b, i)
		if err != nil {
			return UvmEvent{}, err
		}
		i = ni
		var v uint64
		switch field {
		case 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11:
			v, i, err = readVarint(b, i)
		default:
			i, err = skip(b, i, wire)
		}
		if err != nil {
			return UvmEvent{}, err
		}
		switch field {
		case 1:
			ev.Kind = EventKind(v)
		case 2:
			ev.Bytes = v
		case 3:
			ev.StartNs = v
		case 4:
			ev.EndNs = v
		case 5:
			ev.Address = v
		case 6:
			ev.SrcID = uint32(v)
		case 7:
			ev.DstID = uint32(v)
		case 8:
			ev.Cause = Cause(v)
		case 9:
			ev.StreamID = uint32(v)
		case 10:
			ev.CorrelationID = uint32(v)
		case 11:
			ev.DeviceID = uint32(v)
		}
	}
	return ev, nil
}

func decodeAlloc(b []byte) (AllocEvent, error) {
	var ev AllocEvent
	i := 0
	for i < len(b) {
		field, wire, ni, err := readKey(b, i)
		if err != nil {
			return AllocEvent{}, err
		}
		i = ni
		var v uint64
		switch field {
		case 1, 2, 3, 4, 5, 6:
			v, i, err = readVarint(b, i)
		default:
			i, err = skip(b, i, wire)
		}
		if err != nil {
			return AllocEvent{}, err
		}
		switch field {
		case 1:
			ev.Address = v
		case 2:
			ev.Size = v
		case 3:
			ev.IsManaged = v != 0
		case 4:
			ev.IsFree = v != 0
		case 5:
			ev.TimestampNs = v
		case 6:
			ev.DeviceID = uint32(v)
		}
	}
	return ev, nil
}

func decodeKernel(b []byte) (KernelEvent, error) {
	var ev KernelEvent
	i := 0
	for i < len(b) {
		field, wire, ni, err := readKey(b, i)
		if err != nil {
			return KernelEvent{}, err
		}
		i = ni
		switch field {
		case 1:
			ev.Name, i, err = readString(b, i)
		case 2:
			ev.StartNs, i, err = readVarint(b, i)
		case 3:
			ev.EndNs, i, err = readVarint(b, i)
		case 4:
			v, ni, e := readVarint(b, i)
			ev.DeviceID, i, err = uint32(v), ni, e
		case 5:
			v, ni, e := readVarint(b, i)
			ev.CorrelationID, i, err = uint32(v), ni, e
		case 6:
			v, ni, e := readVarint(b, i)
			ev.StreamID, i, err = uint32(v), ni, e
		default:
			i, err = skip(b, i, wire)
		}
		if err != nil {
			return KernelEvent{}, err
		}
	}
	return ev, nil
}

func readKey(b []byte, i int) (field, wire, ni int, err error) {
	v, ni, err := readVarint(b, i)
	if err != nil {
		return 0, 0, i, err
	}
	return int(v >> 3), int(v & 7), ni, nil
}

func readVarint(b []byte, i int) (uint64, int, error) {
	var x uint64
	var s uint
	for {
		if i >= len(b) {
			return 0, i, io.ErrUnexpectedEOF
		}
		c := b[i]
		i++
		if c < 0x80 {
			if s >= 64 {
				return 0, i, errOverflow
			}
			return x | uint64(c)<<s, i, nil
		}
		if s >= 63 {
			return 0, i, errOverflow
		}
		x |= uint64(c&0x7f) << s
		s += 7
	}
}

func readBytes(b []byte, i int) ([]byte, int, error) {
	n, i, err := readVarint(b, i)
	if err != nil {
		return nil, i, err
	}
	if uint64(len(b)-i) < n {
		return nil, i, io.ErrUnexpectedEOF
	}
	end := i + int(n)
	out := make([]byte, n)
	copy(out, b[i:end])
	return out, end, nil
}

func readString(b []byte, i int) (string, int, error) {
	raw, i, err := readBytes(b, i)
	if err != nil {
		return "", i, err
	}
	return string(raw), i, nil
}

func skip(b []byte, i, wire int) (int, error) {
	switch wire {
	case 0:
		_, i, err := readVarint(b, i)
		return i, err
	case 1:
		if len(b)-i < 8 {
			return i, io.ErrUnexpectedEOF
		}
		return i + 8, nil
	case 2:
		n, i, err := readVarint(b, i)
		if err != nil {
			return i, err
		}
		if uint64(len(b)-i) < n {
			return i, io.ErrUnexpectedEOF
		}
		return i + int(n), nil
	case 5:
		if len(b)-i < 4 {
			return i, io.ErrUnexpectedEOF
		}
		return i + 4, nil
	default:
		return i, fmt.Errorf("protocol: wire type %d", wire)
	}
}
