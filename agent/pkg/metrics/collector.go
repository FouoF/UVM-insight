package metrics

import (
	"github.com/prometheus/client_golang/prometheus"

	"github.com/FouoF/UVM-insight/agent/pkg/aggregate"
)

// Collector exports the in-memory store. Series disappear when the process
// record expires, so exited workloads do not leave stale labels behind.
type Collector struct {
	store *aggregate.Store

	migrated *prometheus.Desc
	explicit *prometheus.Desc
	duration *prometheus.Desc
	faults   *prometheus.Desc
	thrash   *prometheus.Desc
	throttle *prometheus.Desc
	remote   *prometheus.Desc
	managed  *prometheus.Desc
	dropped  *prometheus.Desc
}

func New(store *aggregate.Store) *Collector {
	labels := []string{"pid", "app", "gpu", "direction", "cause"}
	id := []string{"pid", "app"}
	return &Collector{
		store: store,
		migrated: prometheus.NewDesc(
			"uvm_migrated_bytes_total",
			"Unified memory bytes migrated between host and device memory.",
			labels, nil,
		),
		explicit: prometheus.NewDesc(
			"uvm_explicit_copy_bytes_total",
			"Bytes moved by explicit CUDA memcpy, for comparison with unified memory migration.",
			labels, nil,
		),
		duration: prometheus.NewDesc(
			"uvm_migration_duration_seconds",
			"Duration of unified memory migrations.",
			labels, nil,
		),
		faults: prometheus.NewDesc(
			"uvm_page_faults_total",
			"CPU and GPU page faults on unified memory.",
			[]string{"pid", "app", "gpu", "side"}, nil,
		),
		thrash: prometheus.NewDesc(
			"uvm_thrashing_events_total",
			"Unified memory thrashing events.",
			[]string{"pid", "app", "gpu"}, nil,
		),
		throttle: prometheus.NewDesc(
			"uvm_throttling_events_total",
			"Unified memory throttling events.",
			[]string{"pid", "app", "gpu"}, nil,
		),
		remote: prometheus.NewDesc(
			"uvm_remote_map_events_total",
			"Unified memory remote map events.",
			[]string{"pid", "app", "gpu"}, nil,
		),
		managed: prometheus.NewDesc(
			"uvm_managed_allocated_bytes",
			"Bytes currently allocated with cudaMallocManaged.",
			id, nil,
		),
		dropped: prometheus.NewDesc(
			"uvm_collector_dropped_events_total",
			"Events the collector dropped before the agent stored them.",
			id, nil,
		),
	}
}

func (c *Collector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.migrated
	ch <- c.explicit
	ch <- c.duration
	ch <- c.faults
	ch <- c.thrash
	ch <- c.throttle
	ch <- c.remote
	ch <- c.managed
	ch <- c.dropped
}

func (c *Collector) Collect(ch chan<- prometheus.Metric) {
	snap := c.store.Prom()
	for _, s := range snap.Migrated {
		ch <- prometheus.MustNewConstMetric(c.migrated, prometheus.CounterValue, s.Value, s.PID, s.App, s.GPU, s.Direction, s.Cause)
	}
	for _, s := range snap.Explicit {
		ch <- prometheus.MustNewConstMetric(c.explicit, prometheus.CounterValue, s.Value, s.PID, s.App, s.GPU, s.Direction, s.Cause)
	}
	for _, s := range snap.Durations {
		ch <- prometheus.MustNewConstHistogram(c.duration, s.Count, s.Sum, s.Buckets, s.PID, s.App, s.GPU, s.Direction, s.Cause)
	}
	for _, s := range snap.Faults {
		ch <- prometheus.MustNewConstMetric(c.faults, prometheus.CounterValue, s.Value, s.PID, s.App, s.GPU, s.Side)
	}
	for _, s := range snap.Thrash {
		ch <- prometheus.MustNewConstMetric(c.thrash, prometheus.CounterValue, s.Value, s.PID, s.App, s.GPU)
	}
	for _, s := range snap.Throttle {
		ch <- prometheus.MustNewConstMetric(c.throttle, prometheus.CounterValue, s.Value, s.PID, s.App, s.GPU)
	}
	for _, s := range snap.Remote {
		ch <- prometheus.MustNewConstMetric(c.remote, prometheus.CounterValue, s.Value, s.PID, s.App, s.GPU)
	}
	for _, s := range snap.Managed {
		ch <- prometheus.MustNewConstMetric(c.managed, prometheus.GaugeValue, s.Value, s.PID, s.App)
	}
	for _, s := range snap.Dropped {
		ch <- prometheus.MustNewConstMetric(c.dropped, prometheus.CounterValue, s.Value, s.PID, s.App)
	}
}
