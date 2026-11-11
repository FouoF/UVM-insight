package metrics

import (
	"github.com/prometheus/client_golang/prometheus"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
)

// Collector exports the in-memory store. Series disappear when the process
// record expires, so exited workloads do not leave stale labels behind.
//
// Every series carries namespace, pod and container (empty outside
// Kubernetes) plus pid and app. gpu is the device index inside the process;
// gpu_uuid joins with dcgm-exporter and nvidia-smi.
type Collector struct {
	store *aggregate.Store

	info     *prometheus.Desc
	migrated *prometheus.Desc
	explicit *prometheus.Desc
	duration *prometheus.Desc
	faults   *prometheus.Desc
	thrash   *prometheus.Desc
	throttle *prometheus.Desc
	remote   *prometheus.Desc
	managed          *prometheus.Desc
	resident         *prometheus.Desc
	hostResident     *prometheus.Desc
	residencySource  *prometheus.Desc
	dropped          *prometheus.Desc
}

var ident = []string{"namespace", "pod", "container", "pid", "app"}

func with(extra ...string) []string {
	out := make([]string, 0, len(ident)+len(extra))
	out = append(out, ident...)
	return append(out, extra...)
}

func identValues(id aggregate.Ident, extra ...string) []string {
	out := make([]string, 0, len(ident)+len(extra))
	out = append(out, id.Namespace, id.Pod, id.Container, id.PID, id.App)
	return append(out, extra...)
}

func New(store *aggregate.Store) *Collector {
	flow := with("gpu", "gpu_uuid", "direction", "cause")
	gpu := with("gpu", "gpu_uuid")
	return &Collector{
		store: store,
		info: prometheus.NewDesc(
			"uvm_process_info",
			"CUDA processes reporting to this exporter. cupti_api is the CUPTI API version loaded in the process.",
			with("cupti_api", "driver_version"), nil,
		),
		migrated: prometheus.NewDesc(
			"uvm_migrated_bytes_total",
			"Unified memory bytes migrated between host and device memory.",
			flow, nil,
		),
		explicit: prometheus.NewDesc(
			"uvm_explicit_copy_bytes_total",
			"Bytes moved by explicit CUDA memcpy, for comparison with unified memory migration.",
			flow, nil,
		),
		duration: prometheus.NewDesc(
			"uvm_migration_duration_seconds",
			"Duration of unified memory migrations.",
			flow, nil,
		),
		faults: prometheus.NewDesc(
			"uvm_page_faults_total",
			"CPU and GPU page faults on unified memory.",
			with("gpu", "gpu_uuid", "side"), nil,
		),
		thrash: prometheus.NewDesc(
			"uvm_thrashing_events_total",
			"Unified memory thrashing events.",
			gpu, nil,
		),
		throttle: prometheus.NewDesc(
			"uvm_throttling_events_total",
			"Unified memory throttling events.",
			gpu, nil,
		),
		remote: prometheus.NewDesc(
			"uvm_remote_map_events_total",
			"Unified memory remote map events.",
			gpu, nil,
		),
		managed: prometheus.NewDesc(
			"uvm_managed_allocated_bytes",
			"Virtual size of the process's unified memory mappings.",
			ident, nil,
		),
		resident: prometheus.NewDesc(
			"uvm_gpu_resident_bytes",
			"Estimated unified memory resident on the GPU. Part of managed = gpu + host + unknown.",
			ident, nil,
		),
		hostResident: prometheus.NewDesc(
			"uvm_host_resident_bytes",
			"Estimated unified memory resident on the host. Part of managed = gpu + host + unknown.",
			ident, nil,
		),
		residencySource: prometheus.NewDesc(
			"uvm_residency_source",
			"How resident gauges are produced for this process. source is none, driver, or delta.",
			with("source"), nil,
		),
		dropped: prometheus.NewDesc(
			"uvm_collector_dropped_events_total",
			"Events the collector dropped before the agent stored them.",
			ident, nil,
		),
	}
}

func (c *Collector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.info
	ch <- c.migrated
	ch <- c.explicit
	ch <- c.duration
	ch <- c.faults
	ch <- c.thrash
	ch <- c.throttle
	ch <- c.remote
	ch <- c.managed
	ch <- c.resident
	ch <- c.hostResident
	ch <- c.residencySource
	ch <- c.dropped
}

func (c *Collector) Collect(ch chan<- prometheus.Metric) {
	snap := c.store.Prom()
	for _, s := range snap.Info {
		ch <- prometheus.MustNewConstMetric(c.info, prometheus.GaugeValue, 1, identValues(s.Ident, s.CuptiAPI, s.Driver)...)
	}
	for _, s := range snap.Migrated {
		ch <- prometheus.MustNewConstMetric(c.migrated, prometheus.CounterValue, s.Value, identValues(s.Ident, s.GPU, s.GPUUUID, s.Direction, s.Cause)...)
	}
	for _, s := range snap.Explicit {
		ch <- prometheus.MustNewConstMetric(c.explicit, prometheus.CounterValue, s.Value, identValues(s.Ident, s.GPU, s.GPUUUID, s.Direction, s.Cause)...)
	}
	for _, s := range snap.Durations {
		ch <- prometheus.MustNewConstHistogram(c.duration, s.Count, s.Sum, s.Buckets, identValues(s.Ident, s.GPU, s.GPUUUID, s.Direction, s.Cause)...)
	}
	for _, s := range snap.Faults {
		ch <- prometheus.MustNewConstMetric(c.faults, prometheus.CounterValue, s.Value, identValues(s.Ident, s.GPU, s.GPUUUID, s.Side)...)
	}
	for _, s := range snap.Thrash {
		ch <- prometheus.MustNewConstMetric(c.thrash, prometheus.CounterValue, s.Value, identValues(s.Ident, s.GPU, s.GPUUUID)...)
	}
	for _, s := range snap.Throttle {
		ch <- prometheus.MustNewConstMetric(c.throttle, prometheus.CounterValue, s.Value, identValues(s.Ident, s.GPU, s.GPUUUID)...)
	}
	for _, s := range snap.Remote {
		ch <- prometheus.MustNewConstMetric(c.remote, prometheus.CounterValue, s.Value, identValues(s.Ident, s.GPU, s.GPUUUID)...)
	}
	for _, s := range snap.Managed {
		ch <- prometheus.MustNewConstMetric(c.managed, prometheus.GaugeValue, s.Value, identValues(s.Ident)...)
	}
	for _, s := range snap.Resident {
		ch <- prometheus.MustNewConstMetric(c.resident, prometheus.GaugeValue, s.Value, identValues(s.Ident)...)
	}
	for _, s := range snap.HostResident {
		ch <- prometheus.MustNewConstMetric(c.hostResident, prometheus.GaugeValue, s.Value, identValues(s.Ident)...)
	}
	for _, s := range snap.ResidencySource {
		ch <- prometheus.MustNewConstMetric(c.residencySource, prometheus.GaugeValue, 1, identValues(s.Ident, s.Source)...)
	}
	for _, s := range snap.Dropped {
		ch <- prometheus.MustNewConstMetric(c.dropped, prometheus.CounterValue, s.Value, identValues(s.Ident)...)
	}
}
