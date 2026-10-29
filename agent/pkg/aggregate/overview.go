package aggregate

import (
	"fmt"
	"sort"
	"time"

	"github.com/FouoF/UVM-insight/agent/pkg/protocol"
)

const (
	seriesSeconds = 120
	exitedLinger  = 2 * time.Minute
)

// Lane is one process's traffic to one GPU. Rates are MB/s over rateWindowMs.
type Lane struct {
	GPU         string  `json:"gpu"`
	HtoD        float64 `json:"htod"`
	DtoH        float64 `json:"dtoh"`
	MemcpyHtoD  float64 `json:"memcpyHtod"`
	MemcpyDtoH  float64 `json:"memcpyDtoh"`
	QuotaBytes  uint64  `json:"quotaBytes"`
	DeviceLabel string  `json:"deviceLabel"`
}

type ProcLive struct {
	PID          uint32 `json:"pid"`
	AppName      string `json:"appName"`
	Hostname     string `json:"hostname"`
	CudaVersion  string `json:"cudaVersion"`
	Active       bool   `json:"active"`
	LastSeen     int64  `json:"lastSeen"`
	ManagedBytes uint64 `json:"managedBytes"`
	// Estimated by the collector; only set when ResidentKnown.
	ResidentBytes uint64  `json:"residentBytes"`
	ResidentKnown bool    `json:"residentKnown"`
	HtoD          float64 `json:"htod"`
	DtoH          float64 `json:"dtoh"`
	Memcpy        float64 `json:"memcpy"`
	GPUFaults     float64 `json:"gpuFaults"`
	CPUFaults     float64 `json:"cpuFaults"`
	Thrashing     float64 `json:"thrashing"`
	MigratedHtoD  uint64  `json:"migratedHtod"`
	MigratedDtoH  uint64  `json:"migratedDtoh"`
	Lanes         []Lane  `json:"lanes"`
	Series        []Point `json:"series"`
}

type GPUView struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	UUID        string `json:"uuid"`
	MemoryBytes uint64 `json:"memoryBytes"`
}

type Overview struct {
	Now       int64       `json:"now"`
	WindowMs  int         `json:"windowMs"`
	GPUs      []GPUView   `json:"gpus"`
	Processes []ProcLive  `json:"processes"`
	Events    []EventView `json:"events"`
	LastSeq   uint64      `json:"lastSeq"`
}

// Overview returns every live (or just exited) process side by side. With
// history set it carries the last two minutes of per-second points; otherwise
// only the last two seconds, for incremental updates.
func (s *Store) Overview(since uint64, history bool) Overview {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	out := Overview{
		Now:       now.UnixMilli(),
		WindowMs:  rateWindowMs,
		GPUs:      []GPUView{},
		Processes: []ProcLive{},
		Events:    []EventView{},
	}
	gpuSeen := map[string]bool{}
	for _, p := range s.procs {
		if !p.visible(now) {
			continue
		}
		for _, g := range p.hello.GPUs {
			id := gpuKey(g)
			if gpuSeen[id] {
				continue
			}
			gpuSeen[id] = true
			out.GPUs = append(out.GPUs, GPUView{ID: id, Name: g.Name, UUID: g.UUID, MemoryBytes: g.TotalMemoryBytes})
		}
		live := p.live(now, history)
		out.Processes = append(out.Processes, live)
		if p.seq > out.LastSeq {
			out.LastSeq = p.seq
		}
		for _, ev := range p.orderedEvents() {
			if ev.Seq > since {
				out.Events = append(out.Events, ev)
			}
		}
	}
	sort.Slice(out.GPUs, func(i, j int) bool { return out.GPUs[i].ID < out.GPUs[j].ID })
	sort.Slice(out.Processes, func(i, j int) bool {
		a, b := out.Processes[i], out.Processes[j]
		if a.Active != b.Active {
			return a.Active
		}
		if a.AppName != b.AppName {
			return a.AppName < b.AppName
		}
		return a.PID < b.PID
	})
	sort.Slice(out.Events, func(i, j int) bool { return out.Events[i].Seq < out.Events[j].Seq })
	limit := 200
	if history {
		limit = 150
	}
	if len(out.Events) > limit {
		out.Events = out.Events[len(out.Events)-limit:]
	}
	return out
}

func (p *proc) visible(now time.Time) bool {
	if p.active {
		return true
	}
	return p.exited != nil && now.Sub(*p.exited) < exitedLinger
}

// HAMi reports the per-container limit as total memory, so the physical GPU is
// identified by UUID and the limit is kept on the lane.
func gpuKey(g protocol.GpuInfo) string {
	if g.UUID != "" {
		return g.UUID
	}
	return fmt.Sprintf("gpu%d", g.DeviceID)
}

func (p *proc) live(now time.Time, history bool) ProcLive {
	from, to, div := rateSpan(now)
	lanes := map[string]*Lane{}
	order := []string{}
	byOrdinal := map[string]string{}
	for _, g := range p.hello.GPUs {
		id := gpuKey(g)
		byOrdinal[gpuName(g.DeviceID)] = id
		if lanes[id] == nil {
			lanes[id] = &Lane{GPU: id, QuotaBytes: g.TotalMemoryBytes, DeviceLabel: gpuName(g.DeviceID)}
			order = append(order, id)
		}
	}
	laneFor := func(label string) *Lane {
		id, ok := byOrdinal[label]
		if !ok {
			id = label
		}
		l := lanes[id]
		if l == nil {
			l = &Lane{GPU: id, DeviceLabel: label}
			lanes[id] = l
			order = append(order, id)
		}
		return l
	}

	out := ProcLive{
		PID:           p.hello.PID,
		AppName:       p.app(),
		Hostname:      p.hello.Hostname,
		CudaVersion:   p.hello.CudaVersion,
		Active:        p.active,
		LastSeen:      p.lastSeen.UnixMilli(),
		ManagedBytes:  p.managed,
		ResidentBytes: p.resident,
		ResidentKnown: p.managedKnown,
	}
	var faultsG, faultsC, thrash uint64
	for id := from; id <= to; id++ {
		b := p.fine[id]
		if b == nil {
			continue
		}
		faultsG += b.gpuFaults
		faultsC += b.cpuFaults
		thrash += b.thrash
		for k, v := range b.flows {
			mb := float64(v) / div / 1e6
			switch k.direction {
			case "htod":
				laneFor(k.dst).HtoD += mb
			case "dtoh":
				laneFor(k.src).DtoH += mb
			case "memcpy_htod":
				laneFor(k.dst).MemcpyHtoD += mb
			case "memcpy_dtoh":
				laneFor(k.src).MemcpyDtoH += mb
			}
		}
	}
	out.GPUFaults = float64(faultsG) / div
	out.CPUFaults = float64(faultsC) / div
	out.Thrashing = float64(thrash) / div
	for _, id := range order {
		l := *lanes[id]
		out.HtoD += l.HtoD
		out.DtoH += l.DtoH
		out.Memcpy += l.MemcpyHtoD + l.MemcpyDtoH
		out.Lanes = append(out.Lanes, l)
	}
	if out.Lanes == nil {
		out.Lanes = []Lane{}
	}
	for k, v := range p.migrated {
		switch k.direction {
		case "htod":
			out.MigratedHtoD += v
		case "dtoh":
			out.MigratedDtoH += v
		}
	}

	// Live updates resend the previous second too, so records that land after
	// the last tick of a second still reach the client.
	sec := now.Unix()
	start := sec - 1
	if history {
		start = sec - seriesSeconds + 1
	}
	out.Series = make([]Point, 0, sec-start+1)
	if !p.managedKnown {
		for sec := start; sec <= now.Unix(); sec++ {
			pt, _ := p.pointAt(sec)
			out.Series = append(out.Series, pt)
		}
		return out
	}
	// The collector only reports the resident gauge when it moves, so carry
	// the last value forward. Seconds before the first report in range take
	// that first report.
	last := p.resident
	for s := start; s <= now.Unix(); s++ {
		if b := p.buckets[s]; b != nil && b.hasResident {
			last = b.resident
			break
		}
	}
	for sec := start; sec <= now.Unix(); sec++ {
		pt, _ := p.pointAt(sec)
		if b := p.buckets[sec]; b != nil && b.hasResident {
			last = b.resident
		}
		pt.ResidentMB = float64(last) / 1e6
		out.Series = append(out.Series, pt)
	}
	return out
}
