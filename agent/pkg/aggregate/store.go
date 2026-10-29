package aggregate

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

const (
	addrBucketBytes = 2 << 20
	heatBucket      = 5 * time.Second
	maxEvents       = 2000
	maxKernels      = 80
	cpuID           = uint32(0xffffffff)
)

// Store keeps a short in-memory history of migration traffic for one machine.
type Store struct {
	mu        sync.Mutex
	retention time.Duration
	now       func() time.Time
	procs     map[uint32]*proc
}

func New(retention time.Duration) *Store {
	if retention <= 0 {
		retention = 10 * time.Minute
	}
	return &Store{
		retention: retention,
		now:       time.Now,
		procs:     map[uint32]*proc{},
	}
}

// SetClock overrides the clock. Tests and the demo generator use it to replay history.
func (s *Store) SetClock(now func() time.Time) {
	s.mu.Lock()
	s.now = now
	s.mu.Unlock()
}

func (s *Store) Register(h protocol.Hello) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.registerLocked(h, s.now())
}

func (s *Store) Ingest(pid uint32, batch protocol.EventBatch) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ingestLocked(pid, batch, s.now())
}

// IngestAt records a batch as if it arrived at the given time.
func (s *Store) IngestAt(pid uint32, batch protocol.EventBatch, at time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ingestLocked(pid, batch, at)
}

func (s *Store) MarkExited(pid uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.procs[pid]
	if p == nil || !p.active {
		return
	}
	p.active = false
	t := s.now()
	p.exited = &t
}

// Expire drops processes that have been gone longer than the retention window
// and trims old samples.
func (s *Store) Expire() {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	cutoff := now.Add(-s.retention).Unix()
	for pid, p := range s.procs {
		if !p.active && p.exited != nil && now.Sub(*p.exited) > s.retention {
			delete(s.procs, pid)
			continue
		}
		for sec := range p.buckets {
			if sec < cutoff {
				delete(p.buckets, sec)
			}
		}
		for key := range p.heat {
			if key.sec < cutoff {
				delete(p.heat, key)
			}
		}
	}
}

func (s *Store) registerLocked(h protocol.Hello, at time.Time) {
	p := s.procs[h.PID]
	if p == nil {
		p = newProc()
		s.procs[h.PID] = p
	}
	p.hello = h
	if p.hello.AppName == "" {
		p.hello.AppName = h.Comm
	}
	p.active = true
	p.exited = nil
	p.lastSeen = at
}

func (s *Store) ingestLocked(pid uint32, batch protocol.EventBatch, at time.Time) {
	p := s.procs[pid]
	if p == nil {
		p = newProc()
		p.hello.PID = pid
		p.hello.AppName = fmt.Sprintf("pid-%d", pid)
		s.procs[pid] = p
	}
	p.active = true
	p.lastSeen = at
	p.dropped += batch.Dropped
	p.observeClock(batch, at)

	for _, ev := range batch.Allocs {
		p.noteAlloc(ev)
	}
	for _, ev := range batch.Uvm {
		p.noteUvm(ev, at)
	}
	for _, ev := range batch.Kernels {
		p.noteKernel(ev, at)
	}
}

type proc struct {
	hello     protocol.Hello
	active    bool
	lastSeen  time.Time
	exited    *time.Time
	managed   uint64
	dropped   uint64
	allocs    map[uint64]uint64
	buckets   map[int64]*second
	heat      map[heatKey]uint64
	events    []EventView
	seq       uint64
	kernels   []KernelView
	migrated  map[seriesKey]uint64
	explicit  map[seriesKey]uint64
	faults    map[faultKey]uint64
	thrash    map[simpleKey]uint64
	throttle  map[simpleKey]uint64
	remote    map[simpleKey]uint64
	hists     map[seriesKey]*histogram
	offsetSet bool
	offset    int64
}

func newProc() *proc {
	return &proc{
		allocs:   map[uint64]uint64{},
		buckets:  map[int64]*second{},
		heat:     map[heatKey]uint64{},
		migrated: map[seriesKey]uint64{},
		explicit: map[seriesKey]uint64{},
		faults:   map[faultKey]uint64{},
		thrash:   map[simpleKey]uint64{},
		throttle: map[simpleKey]uint64{},
		remote:   map[simpleKey]uint64{},
		hists:    map[seriesKey]*histogram{},
	}
}

type second struct {
	flows      map[flowKey]uint64
	htod       uint64
	dtoh       uint64
	dtod       uint64
	memcpyHtod uint64
	memcpyDtoh uint64
	memcpyDtod uint64
	cpuFaults  uint64
	gpuFaults  uint64
	thrash     uint64
	throttle   uint64
	remote     uint64
}

func (p *proc) bucket(sec int64) *second {
	b := p.buckets[sec]
	if b == nil {
		b = &second{flows: map[flowKey]uint64{}}
		p.buckets[sec] = b
	}
	return b
}

type flowKey struct {
	src, dst, direction, cause string
}

type heatKey struct {
	addr      uint64
	sec       int64
	direction string
}

type seriesKey struct {
	gpu, direction, cause string
}

type faultKey struct {
	gpu, side string
}

type simpleKey struct {
	gpu string
}

func (p *proc) app() string {
	if p.hello.AppName != "" {
		return p.hello.AppName
	}
	if p.hello.Comm != "" {
		return p.hello.Comm
	}
	return fmt.Sprintf("pid-%d", p.hello.PID)
}

func (p *proc) observeClock(batch protocol.EventBatch, at time.Time) {
	if p.offsetSet {
		return
	}
	var max uint64
	for _, ev := range batch.Uvm {
		if ev.EndNs > max {
			max = ev.EndNs
		}
	}
	for _, ev := range batch.Kernels {
		if ev.EndNs > max {
			max = ev.EndNs
		}
	}
	if max == 0 {
		return
	}
	p.offset = at.UnixNano() - int64(max)
	p.offsetSet = true
}

func (p *proc) wall(cupti uint64, fallback time.Time) time.Time {
	if cupti == 0 || !p.offsetSet {
		return fallback
	}
	return time.Unix(0, int64(cupti)+p.offset)
}

func (p *proc) noteAlloc(ev protocol.AllocEvent) {
	if !ev.IsManaged {
		return
	}
	if !ev.IsFree {
		if _, ok := p.allocs[ev.Address]; !ok {
			p.managed += ev.Size
		} else {
			p.managed -= p.allocs[ev.Address]
			p.managed += ev.Size
		}
		p.allocs[ev.Address] = ev.Size
		return
	}
	if sz, ok := p.allocs[ev.Address]; ok {
		if p.managed >= sz {
			p.managed -= sz
		} else {
			p.managed = 0
		}
		delete(p.allocs, ev.Address)
		return
	}
	if p.managed >= ev.Size {
		p.managed -= ev.Size
	} else {
		p.managed = 0
	}
}

func (p *proc) noteUvm(ev protocol.UvmEvent, at time.Time) {
	src, dst, direction, kind := classify(ev)
	gpu := deviceLabel(ev)
	cause := causeName(ev.Cause)
	sec := at.Unix()
	b := p.bucket(sec)
	when := p.wall(ev.EndNs, at)

	view := EventView{
		Time:       when.UnixMilli(),
		Kind:       kind,
		Cause:      cause,
		Direction:  direction,
		Bytes:      ev.Bytes,
		Address:    fmt.Sprintf("0x%x", ev.Address),
		Src:        src,
		Dst:        dst,
		Device:     gpu,
		DurationUs: durationUs(ev),
	}
	p.pushEvent(view)

	switch ev.Kind {
	case protocol.KindBytesHtoD, protocol.KindBytesDtoH, protocol.KindBytesDtoD:
		key := seriesKey{gpu: gpu, direction: direction, cause: cause}
		p.migrated[key] += ev.Bytes
		b.flows[flowKey{src: src, dst: dst, direction: direction, cause: cause}] += ev.Bytes
		switch ev.Kind {
		case protocol.KindBytesHtoD:
			b.htod += ev.Bytes
		case protocol.KindBytesDtoH:
			b.dtoh += ev.Bytes
		default:
			b.dtod += ev.Bytes
		}
		p.observeDuration(key, ev)
		if ev.Address != 0 && ev.Bytes > 0 {
			hk := heatKey{
				addr:      ev.Address &^ (addrBucketBytes - 1),
				sec:       at.Unix() / int64(heatBucket/time.Second) * int64(heatBucket/time.Second),
				direction: direction,
			}
			p.heat[hk] += ev.Bytes
		}
	case protocol.KindMemcpyHtoD, protocol.KindMemcpyDtoH, protocol.KindMemcpyDtoD, protocol.KindMemcpyOther:
		key := seriesKey{gpu: gpu, direction: direction, cause: "explicit"}
		p.explicit[key] += ev.Bytes
		b.flows[flowKey{src: src, dst: dst, direction: direction, cause: "explicit"}] += ev.Bytes
		switch ev.Kind {
		case protocol.KindMemcpyHtoD:
			b.memcpyHtod += ev.Bytes
		case protocol.KindMemcpyDtoH:
			b.memcpyDtoh += ev.Bytes
		default:
			b.memcpyDtod += ev.Bytes
		}
	case protocol.KindCPUPageFault:
		p.faults[faultKey{gpu: gpu, side: "cpu"}]++
		b.cpuFaults++
	case protocol.KindGPUPageFault:
		n := ev.Bytes
		if n == 0 {
			n = 1
		}
		p.faults[faultKey{gpu: gpu, side: "gpu"}] += n
		b.gpuFaults += n
	case protocol.KindThrashing:
		p.thrash[simpleKey{gpu: gpu}]++
		b.thrash++
	case protocol.KindThrottling:
		p.throttle[simpleKey{gpu: gpu}]++
		b.throttle++
	case protocol.KindRemoteMap:
		p.remote[simpleKey{gpu: gpu}]++
		b.remote++
	}
}

func (p *proc) noteKernel(ev protocol.KernelEvent, at time.Time) {
	start := p.wall(ev.StartNs, at)
	end := p.wall(ev.EndNs, at)
	if end.Before(start) {
		end = start
	}
	p.kernels = append(p.kernels, KernelView{
		Name:   ev.Name,
		Device: fmt.Sprintf("gpu%d", ev.DeviceID),
		Start:  start.UnixMilli(),
		End:    end.UnixMilli(),
	})
	if len(p.kernels) > maxKernels {
		p.kernels = append([]KernelView(nil), p.kernels[len(p.kernels)-maxKernels:]...)
	}
}

func (p *proc) pushEvent(ev EventView) {
	p.seq++
	ev.Seq = p.seq
	p.events = append(p.events, ev)
	if len(p.events) > maxEvents {
		p.events = append([]EventView(nil), p.events[len(p.events)-maxEvents:]...)
	}
}

func (p *proc) observeDuration(key seriesKey, ev protocol.UvmEvent) {
	if ev.EndNs <= ev.StartNs {
		return
	}
	seconds := float64(ev.EndNs-ev.StartNs) / 1e9
	h := p.hists[key]
	if h == nil {
		h = &histogram{}
		p.hists[key] = h
	}
	h.observe(seconds)
}

var histBounds = []float64{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10}

type histogram struct {
	count   uint64
	sum     float64
	buckets [8]uint64
}

func (h *histogram) observe(v float64) {
	h.count++
	h.sum += v
	for i, bound := range histBounds {
		if v <= bound {
			h.buckets[i]++
		}
	}
}

func (h *histogram) cumulative() map[float64]uint64 {
	// observe() already stores cumulative counts: buckets[i] is the number of
	// samples less than or equal to histBounds[i].
	out := make(map[float64]uint64, len(histBounds))
	for i, bound := range histBounds {
		out[bound] = h.buckets[i]
	}
	return out
}

func durationUs(ev protocol.UvmEvent) float64 {
	switch ev.Kind {
	case protocol.KindBytesHtoD, protocol.KindBytesDtoH, protocol.KindBytesDtoD,
		protocol.KindMemcpyHtoD, protocol.KindMemcpyDtoH, protocol.KindMemcpyDtoD, protocol.KindMemcpyOther:
	default:
		// Faults and thrash records don't have a transfer duration. CPU faults
		// even leave the end timestamp unused, so subtracting it is meaningless.
		return 0
	}
	if ev.StartNs == 0 || ev.EndNs <= ev.StartNs {
		return 0
	}
	return float64(ev.EndNs-ev.StartNs) / 1e3
}

func causeName(c protocol.Cause) string {
	switch c {
	case protocol.CauseUser:
		return "user"
	case protocol.CauseCoherence:
		return "coherence"
	case protocol.CausePrefetch:
		return "prefetch"
	case protocol.CauseEviction:
		return "eviction"
	case protocol.CauseAccessCounters:
		return "access_counters"
	default:
		return "unknown"
	}
}

func gpuName(id uint32) string {
	if id == cpuID {
		return "host"
	}
	return fmt.Sprintf("gpu%d", id)
}

func deviceLabel(ev protocol.UvmEvent) string {
	switch ev.Kind {
	case protocol.KindBytesHtoD, protocol.KindMemcpyHtoD:
		if ev.DstID != cpuID {
			return gpuName(ev.DstID)
		}
	case protocol.KindBytesDtoH, protocol.KindMemcpyDtoH:
		if ev.SrcID != cpuID {
			return gpuName(ev.SrcID)
		}
	case protocol.KindBytesDtoD, protocol.KindMemcpyDtoD:
		if ev.DstID != cpuID {
			return gpuName(ev.DstID)
		}
	}
	if ev.DeviceID == cpuID {
		return "cpu"
	}
	return gpuName(ev.DeviceID)
}

func classify(ev protocol.UvmEvent) (src, dst, direction, kind string) {
	kind = kindName(ev.Kind)
	switch ev.Kind {
	case protocol.KindBytesHtoD:
		return "host", deviceLabel(ev), "htod", kind
	case protocol.KindBytesDtoH:
		return deviceLabel(ev), "host", "dtoh", kind
	case protocol.KindBytesDtoD:
		return gpuName(ev.SrcID), gpuName(ev.DstID), "dtod", kind
	case protocol.KindMemcpyHtoD:
		return "host", deviceLabel(ev), "memcpy_htod", kind
	case protocol.KindMemcpyDtoH:
		return deviceLabel(ev), "host", "memcpy_dtoh", kind
	case protocol.KindMemcpyDtoD:
		return gpuName(ev.SrcID), gpuName(ev.DstID), "memcpy_dtod", kind
	case protocol.KindMemcpyOther:
		return deviceLabel(ev), deviceLabel(ev), "memcpy_other", kind
	case protocol.KindCPUPageFault:
		return "host", deviceLabel(ev), "cpu_fault", kind
	case protocol.KindGPUPageFault:
		return deviceLabel(ev), deviceLabel(ev), "gpu_fault", kind
	default:
		return deviceLabel(ev), deviceLabel(ev), kind, kind
	}
}

func kindName(k protocol.EventKind) string {
	switch k {
	case protocol.KindBytesHtoD:
		return "htod"
	case protocol.KindBytesDtoH:
		return "dtoh"
	case protocol.KindBytesDtoD:
		return "dtod"
	case protocol.KindCPUPageFault:
		return "cpu_page_fault"
	case protocol.KindGPUPageFault:
		return "gpu_page_fault"
	case protocol.KindThrashing:
		return "thrashing"
	case protocol.KindThrottling:
		return "throttling"
	case protocol.KindRemoteMap:
		return "remote_map"
	case protocol.KindMemcpyHtoD:
		return "memcpy_htod"
	case protocol.KindMemcpyDtoH:
		return "memcpy_dtoh"
	case protocol.KindMemcpyDtoD:
		return "memcpy_dtod"
	case protocol.KindMemcpyOther:
		return "memcpy_other"
	default:
		return "unknown"
	}
}

// ---- views -----------------------------------------------------------------

type ProcessInfo struct {
	PID           uint32             `json:"pid"`
	Comm          string             `json:"comm"`
	AppName       string             `json:"appName"`
	Hostname      string             `json:"hostname"`
	CudaVersion   string             `json:"cudaVersion"`
	DriverVersion string             `json:"driverVersion"`
	GPUs          []protocol.GpuInfo `json:"gpus"`
	ManagedBytes  uint64             `json:"managedBytes"`
	Active        bool               `json:"active"`
	Dropped       uint64             `json:"dropped"`
	LastSeen      int64              `json:"lastSeen"`
}

type Flow struct {
	Src       string  `json:"src"`
	Dst       string  `json:"dst"`
	Direction string  `json:"direction"`
	Cause     string  `json:"cause"`
	MBps      float64 `json:"mbps"`
}

type Node struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Kind        string `json:"kind"`
	MemoryBytes uint64 `json:"memoryBytes"`
	UUID        string `json:"uuid"`
}

type Point struct {
	T          int64   `json:"t"`
	HtoD       float64 `json:"htod"`
	DtoH       float64 `json:"dtoh"`
	DtoD       float64 `json:"dtod"`
	MemcpyHtoD float64 `json:"memcpyHtod"`
	MemcpyDtoH float64 `json:"memcpyDtoh"`
	MemcpyDtoD float64 `json:"memcpyDtod"`
	CPUFaults  float64 `json:"cpuFaults"`
	GPUFaults  float64 `json:"gpuFaults"`
	Thrashing  float64 `json:"thrashing"`
	Throttling float64 `json:"throttling"`
	RemoteMap  float64 `json:"remoteMap"`
}

type EventView struct {
	Seq        uint64  `json:"seq"`
	Time       int64   `json:"time"`
	Kind       string  `json:"kind"`
	Cause      string  `json:"cause"`
	Direction  string  `json:"direction"`
	Bytes      uint64  `json:"bytes"`
	Address    string  `json:"address"`
	Src        string  `json:"src"`
	Dst        string  `json:"dst"`
	Device     string  `json:"device"`
	DurationUs float64 `json:"durationUs"`
}

type KernelView struct {
	Name   string `json:"name"`
	Device string `json:"device"`
	Start  int64  `json:"start"`
	End    int64  `json:"end"`
}

type Snapshot struct {
	PID          uint32       `json:"pid"`
	ManagedBytes uint64       `json:"managedBytes"`
	Dropped      uint64       `json:"dropped"`
	Nodes        []Node       `json:"nodes"`
	Flows        []Flow       `json:"flows"`
	Points       []Point      `json:"points"`
	Kernels      []KernelView `json:"kernels"`
	Events       []EventView  `json:"events"`
}

type HeatCell struct {
	Addr      string `json:"addr"`
	T         int64  `json:"t"`
	Bytes     uint64 `json:"bytes"`
	Direction string `json:"direction"`
}

type Heatmap struct {
	BucketBytes   uint64     `json:"bucketBytes"`
	TimeBucketSec int64      `json:"timeBucketSec"`
	Cells         []HeatCell `json:"cells"`
}

type Stream struct {
	Now          int64        `json:"now"`
	ManagedBytes uint64       `json:"managedBytes"`
	Dropped      uint64       `json:"dropped"`
	Active       bool         `json:"active"`
	Flows        []Flow       `json:"flows"`
	Point        *Point       `json:"point"`
	Kernels      []KernelView `json:"kernels"`
	Events       []EventView  `json:"events"`
	LastSeq      uint64       `json:"lastSeq"`
}

func (s *Store) Processes() []ProcessInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ProcessInfo, 0, len(s.procs))
	for _, p := range s.procs {
		out = append(out, p.info())
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Active != out[j].Active {
			return out[i].Active
		}
		return out[i].LastSeen > out[j].LastSeen
	})
	return out
}

func (p *proc) info() ProcessInfo {
	gpus := append([]protocol.GpuInfo(nil), p.hello.GPUs...)
	last := p.lastSeen.UnixMilli()
	return ProcessInfo{
		PID:           p.hello.PID,
		Comm:          p.hello.Comm,
		AppName:       p.app(),
		Hostname:      p.hello.Hostname,
		CudaVersion:   p.hello.CudaVersion,
		DriverVersion: p.hello.DriverVersion,
		GPUs:          gpus,
		ManagedBytes:  p.managed,
		Active:        p.active,
		Dropped:       p.dropped,
		LastSeen:      last,
	}
}

func (s *Store) selected(pid uint32) []*proc {
	if pid != 0 {
		if p := s.procs[pid]; p != nil {
			return []*proc{p}
		}
		return nil
	}
	out := make([]*proc, 0, len(s.procs))
	for _, p := range s.procs {
		if p.active {
			out = append(out, p)
		}
	}
	return out
}

func (s *Store) Snapshot(pid uint32) Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	procs := s.selected(pid)
	snap := Snapshot{
		PID:     pid,
		Nodes:   []Node{{ID: "host", Label: "主机内存", Kind: "host"}},
		Flows:   s.flowsLocked(procs, now),
		Points:  s.pointsLocked(procs, now),
		Kernels: []KernelView{},
		Events:  []EventView{},
	}
	var lastSeq uint64
	for _, p := range procs {
		snap.ManagedBytes += p.managed
		snap.Dropped += p.dropped
		snap.Nodes = mergeNodes(snap.Nodes, p.nodes())
		snap.Kernels = append(snap.Kernels, p.recentKernels(now)...)
		if p.seq > lastSeq {
			lastSeq = p.seq
		}
		snap.Events = append(snap.Events, tail(p.events, 100)...)
	}
	sort.Slice(snap.Events, func(i, j int) bool { return snap.Events[i].Seq > snap.Events[j].Seq })
	if len(snap.Events) > 100 {
		snap.Events = snap.Events[:100]
	}
	return snap
}

func (s *Store) Heatmap(pid uint32) Heatmap {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	cutoff := now.Add(-s.retention).Unix()
	heat := Heatmap{BucketBytes: addrBucketBytes, TimeBucketSec: int64(heatBucket / time.Second), Cells: []HeatCell{}}
	for _, p := range s.selected(pid) {
		for key, bytes := range p.heat {
			if key.sec < cutoff || bytes == 0 {
				continue
			}
			heat.Cells = append(heat.Cells, HeatCell{Addr: fmt.Sprintf("0x%x", key.addr), T: key.sec, Bytes: bytes, Direction: key.direction})
		}
	}
	sort.Slice(heat.Cells, func(i, j int) bool {
		if heat.Cells[i].T == heat.Cells[j].T {
			return heat.Cells[i].Addr < heat.Cells[j].Addr
		}
		return heat.Cells[i].T < heat.Cells[j].T
	})
	if len(heat.Cells) > 4000 {
		heat.Cells = heat.Cells[len(heat.Cells)-4000:]
	}
	return heat
}

func (s *Store) Stream(pid uint32, since uint64) Stream {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	procs := s.selected(pid)
	msg := Stream{
		Now:    now.UnixMilli(),
		Flows:  s.flowsLocked(procs, now),
		Events: []EventView{},
	}
	var point Point
	var havePoint bool
	for _, p := range procs {
		msg.ManagedBytes += p.managed
		msg.Dropped += p.dropped
		if p.active {
			msg.Active = true
		}
		if p.seq > msg.LastSeq {
			msg.LastSeq = p.seq
		}
		msg.Kernels = append(msg.Kernels, p.recentKernels(now)...)
		for _, ev := range p.events {
			if ev.Seq > since {
				msg.Events = append(msg.Events, ev)
			}
		}
		if pt, ok := p.pointAt(now.Unix()); ok {
			if !havePoint {
				point = pt
				havePoint = true
			} else {
				addPoint(&point, pt)
			}
		}
	}
	if havePoint {
		msg.Point = &point
	}
	sort.Slice(msg.Events, func(i, j int) bool { return msg.Events[i].Seq < msg.Events[j].Seq })
	if len(msg.Events) > 200 {
		msg.Events = msg.Events[len(msg.Events)-200:]
	}
	return msg
}

func (p *proc) nodes() []Node {
	out := make([]Node, 0, len(p.hello.GPUs))
	for _, g := range p.hello.GPUs {
		label := g.Name
		if label == "" {
			label = fmt.Sprintf("GPU %d", g.DeviceID)
		}
		out = append(out, Node{
			ID:          fmt.Sprintf("gpu%d", g.DeviceID),
			Label:       label,
			Kind:        "gpu",
			MemoryBytes: g.TotalMemoryBytes,
			UUID:        g.UUID,
		})
	}
	return out
}

func mergeNodes(dst, extra []Node) []Node {
	seen := map[string]bool{}
	for _, n := range dst {
		seen[n.ID] = true
	}
	for _, n := range extra {
		if seen[n.ID] {
			continue
		}
		seen[n.ID] = true
		dst = append(dst, n)
	}
	return dst
}

func (p *proc) recentKernels(now time.Time) []KernelView {
	cutoff := now.Add(-2 * time.Minute).UnixMilli()
	out := make([]KernelView, 0, len(p.kernels))
	for _, k := range p.kernels {
		if k.End >= cutoff {
			out = append(out, k)
		}
	}
	return out
}

func tail(events []EventView, n int) []EventView {
	if len(events) <= n {
		cp := append([]EventView(nil), events...)
		return cp
	}
	return append([]EventView(nil), events[len(events)-n:]...)
}

func (s *Store) flowsLocked(procs []*proc, now time.Time) []Flow {
	acc := map[flowKey]float64{}
	for _, p := range procs {
		for _, f := range p.rates(now) {
			acc[flowKey{f.Src, f.Dst, f.Direction, f.Cause}] += f.MBps
		}
	}
	out := make([]Flow, 0, len(acc))
	for k, mbps := range acc {
		if mbps <= 0 {
			continue
		}
		out = append(out, Flow{Src: k.src, Dst: k.dst, Direction: k.direction, Cause: k.cause, MBps: mbps})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].MBps > out[j].MBps })
	return out
}

func (p *proc) rates(now time.Time) []Flow {
	sec := now.Unix()
	elapsed := now.Sub(time.Unix(sec, 0))
	b := p.buckets[sec]
	div := elapsed.Seconds()
	if elapsed < 200*time.Millisecond || b == nil {
		b = p.buckets[sec-1]
		div = 1
	}
	if b == nil || div <= 0 {
		return nil
	}
	out := make([]Flow, 0, len(b.flows))
	for k, bytes := range b.flows {
		if bytes == 0 {
			continue
		}
		out = append(out, Flow{
			Src: k.src, Dst: k.dst, Direction: k.direction, Cause: k.cause,
			MBps: float64(bytes) / div / 1e6,
		})
	}
	return out
}

func (s *Store) pointsLocked(procs []*proc, now time.Time) []Point {
	if len(procs) == 0 {
		return []Point{}
	}
	start := now.Add(-s.retention).Unix()
	if start < now.Unix()-600 {
		start = now.Unix() - 600
	}
	out := make([]Point, 0, now.Unix()-start+1)
	for sec := start; sec <= now.Unix(); sec++ {
		var pt Point
		pt.T = sec * 1000
		for _, p := range procs {
			if one, ok := p.pointAt(sec); ok {
				addPoint(&pt, one)
			}
		}
		out = append(out, pt)
	}
	return trimLeadingQuiet(out)
}

func trimLeadingQuiet(points []Point) []Point {
	i := 0
	for i < len(points)-1 && quietPoint(points[i]) {
		i++
	}
	if i > 1 {
		i--
	}
	return points[i:]
}

func quietPoint(p Point) bool {
	return p.HtoD == 0 && p.DtoH == 0 && p.DtoD == 0 &&
		p.MemcpyHtoD == 0 && p.MemcpyDtoH == 0 && p.MemcpyDtoD == 0 &&
		p.CPUFaults == 0 && p.GPUFaults == 0 && p.Thrashing == 0 &&
		p.Throttling == 0 && p.RemoteMap == 0
}

func (p *proc) pointAt(sec int64) (Point, bool) {
	b := p.buckets[sec]
	if b == nil {
		return Point{T: sec * 1000}, false
	}
	return Point{
		T:          sec * 1000,
		HtoD:       float64(b.htod) / 1e6,
		DtoH:       float64(b.dtoh) / 1e6,
		DtoD:       float64(b.dtod) / 1e6,
		MemcpyHtoD: float64(b.memcpyHtod) / 1e6,
		MemcpyDtoH: float64(b.memcpyDtoh) / 1e6,
		MemcpyDtoD: float64(b.memcpyDtod) / 1e6,
		CPUFaults:  float64(b.cpuFaults),
		GPUFaults:  float64(b.gpuFaults),
		Thrashing:  float64(b.thrash),
		Throttling: float64(b.throttle),
		RemoteMap:  float64(b.remote),
	}, true
}

func addPoint(dst *Point, src Point) {
	dst.HtoD += src.HtoD
	dst.DtoH += src.DtoH
	dst.DtoD += src.DtoD
	dst.MemcpyHtoD += src.MemcpyHtoD
	dst.MemcpyDtoH += src.MemcpyDtoH
	dst.MemcpyDtoD += src.MemcpyDtoD
	dst.CPUFaults += src.CPUFaults
	dst.GPUFaults += src.GPUFaults
	dst.Thrashing += src.Thrashing
	dst.Throttling += src.Throttling
	dst.RemoteMap += src.RemoteMap
}

// Prometheus export ----------------------------------------------------------

type CounterObs struct {
	PID, App, GPU, Direction, Cause, Side string
	Value                                 float64
}

type HistObs struct {
	PID, App, GPU, Direction, Cause string
	Count                           uint64
	Sum                             float64
	Buckets                         map[float64]uint64
}

type PromSnap struct {
	Migrated  []CounterObs
	Explicit  []CounterObs
	Faults    []CounterObs
	Thrash    []CounterObs
	Throttle  []CounterObs
	Remote    []CounterObs
	Managed   []CounterObs
	Dropped   []CounterObs
	Durations []HistObs
}

func (s *Store) Prom() PromSnap {
	s.mu.Lock()
	defer s.mu.Unlock()
	var snap PromSnap
	for _, p := range s.procs {
		pid := fmt.Sprintf("%d", p.hello.PID)
		app := p.app()
		for k, v := range p.migrated {
			snap.Migrated = append(snap.Migrated, CounterObs{PID: pid, App: app, GPU: k.gpu, Direction: k.direction, Cause: k.cause, Value: float64(v)})
		}
		for k, v := range p.explicit {
			snap.Explicit = append(snap.Explicit, CounterObs{PID: pid, App: app, GPU: k.gpu, Direction: k.direction, Cause: k.cause, Value: float64(v)})
		}
		for k, v := range p.faults {
			snap.Faults = append(snap.Faults, CounterObs{PID: pid, App: app, GPU: k.gpu, Side: k.side, Value: float64(v)})
		}
		for k, v := range p.thrash {
			snap.Thrash = append(snap.Thrash, CounterObs{PID: pid, App: app, GPU: k.gpu, Value: float64(v)})
		}
		for k, v := range p.throttle {
			snap.Throttle = append(snap.Throttle, CounterObs{PID: pid, App: app, GPU: k.gpu, Value: float64(v)})
		}
		for k, v := range p.remote {
			snap.Remote = append(snap.Remote, CounterObs{PID: pid, App: app, GPU: k.gpu, Value: float64(v)})
		}
		snap.Managed = append(snap.Managed, CounterObs{PID: pid, App: app, Value: float64(p.managed)})
		snap.Dropped = append(snap.Dropped, CounterObs{PID: pid, App: app, Value: float64(p.dropped)})
		for k, h := range p.hists {
			snap.Durations = append(snap.Durations, HistObs{
				PID: pid, App: app, GPU: k.gpu, Direction: k.direction, Cause: k.cause,
				Count: h.count, Sum: h.sum, Buckets: h.cumulative(),
			})
		}
	}
	return snap
}
