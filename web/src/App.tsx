import { useEffect, useMemo, useState } from "react";
import {
  causeLabel,
  fetchHeatmap,
  fetchProcesses,
  fetchSnapshot,
  formatBytes,
  kindLabel,
  streamURL,
  type EventView,
  type Flow,
  type Heatmap,
  type KernelView,
  type NodeInfo,
  type Point,
  type ProcessInfo,
  type StreamMsg,
} from "./api";
import { BandwidthChart, FaultChart, HeatChart } from "./Charts";
import { FlowView } from "./FlowView";

const emptyHeat: Heatmap = { bucketBytes: 2 << 20, timeBucketSec: 5, cells: [] };

export function App() {
  const [processes, setProcesses] = useState<ProcessInfo[]>([]);
  const [pid, setPid] = useState<number | "all">("all");
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [flows, setFlows] = useState<Flow[]>([]);
  const [points, setPoints] = useState<Point[]>([]);
  const [kernels, setKernels] = useState<KernelView[]>([]);
  const [events, setEvents] = useState<EventView[]>([]);
  const [heat, setHeat] = useState<Heatmap>(emptyHeat);
  const [managed, setManaged] = useState(0);
  const [dropped, setDropped] = useState(0);
  const [connected, setConnected] = useState(false);
  const [kindFilter, setKindFilter] = useState("all");
  const [causeFilter, setCauseFilter] = useState("all");
  const [error, setError] = useState("");

  useEffect(() => {
    let stop = false;
    const load = () => {
      fetchProcesses()
        .then((list) => {
          if (stop) return;
          setProcesses(list);
          setError("");
        })
        .catch((err: Error) => {
          if (!stop) setError(err.message);
        });
    };
    load();
    const id = setInterval(load, 3000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let stop = false;
    setPoints([]);
    setEvents([]);
    setFlows([]);
    fetchSnapshot(pid)
      .then((snap) => {
        if (stop) return;
        setNodes(snap.nodes ?? []);
        setFlows(snap.flows ?? []);
        setPoints(snap.points ?? []);
        setKernels(snap.kernels ?? []);
        setEvents(snap.events ?? []);
        setManaged(snap.managedBytes);
        setDropped(snap.dropped);
      })
      .catch((err: Error) => setError(err.message));
    fetchHeatmap(pid)
      .then((h) => {
        if (!stop) setHeat(h);
      })
      .catch(() => setHeat(emptyHeat));
    const heatTimer = setInterval(() => {
      fetchHeatmap(pid)
        .then((h) => {
          if (!stop) setHeat(h);
        })
        .catch(() => undefined);
    }, 2000);

    const ws = new WebSocket(streamURL(pid));
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as StreamMsg;
      setFlows(msg.flows ?? []);
      setKernels(msg.kernels ?? []);
      setManaged(msg.managedBytes);
      setDropped(msg.dropped);
      if (msg.point) {
        setPoints((prev) => upsertPoint(prev, msg.point as Point));
      }
      if (msg.events?.length) {
        setEvents((prev) => mergeEvents(prev, msg.events));
      }
    };
    return () => {
      stop = true;
      clearInterval(heatTimer);
      ws.close();
    };
  }, [pid]);

  const selected = processes.find((p) => p.pid === pid);
  const visible = useMemo(
    () =>
      events.filter(
        (ev) => (kindFilter === "all" || ev.kind === kindFilter) && (causeFilter === "all" || ev.cause === causeFilter),
      ),
    [events, kindFilter, causeFilter],
  );
  const kinds = useMemo(() => unique(events.map((e) => e.kind)), [events]);
  const causes = useMemo(() => unique(events.map((e) => e.cause)), [events]);

  return (
    <div className="shell">
      <header className="top">
        <div>
          <div className="kicker">NVIDIA unified memory</div>
          <h1>UVM Insight</h1>
          <p className="lede">显存和主机内存之间的迁移、缺页和抖动。采集来自 CUPTI，指标在 /metrics。</p>
        </div>
        <div className="status">
          <span className={connected ? "dot on" : "dot"} />
          {connected ? "实时流已连接" : "正在连接"}
          {error ? <span className="err">{error}</span> : null}
        </div>
      </header>

      <div className="toolbar">
        <button className={pid === "all" ? "chip on" : "chip"} onClick={() => setPid("all")}>
          全部进程
        </button>
        {processes.map((p) => (
          <button key={p.pid} className={pid === p.pid ? "chip on" : "chip"} onClick={() => setPid(p.pid)}>
            <strong>{p.appName}</strong>
            <span>pid {p.pid}</span>
            {!p.active ? <span>已退出</span> : null}
          </button>
        ))}
        <div className="stats">
          <Stat label="托管内存" value={formatBytes(managed)} />
          <Stat label="丢弃事件" value={String(dropped)} />
          {selected ? <Stat label="GPU" value={String(selected.gpus?.length ?? 0)} /> : null}
        </div>
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2>数据流动</h2>
          <Legend />
        </div>
        {processes.length === 0 ? (
          <div className="empty">
            还没有进程。在有 GPU 的机器上用 CUDA_INJECTION64_PATH 启动应用，或者用 --mock 看演示数据。
          </div>
        ) : (
          <FlowView nodes={nodes} flows={flows} />
        )}
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panel-head">
            <h2>带宽</h2>
          </div>
          <BandwidthChart points={points} kernels={kernels} />
        </div>
        <div className="panel">
          <div className="panel-head">
            <h2>缺页 / 抖动 / 节流</h2>
          </div>
          <FaultChart points={points} />
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>地址热度</h2>
          <span className="hint">2 MB 地址桶，颜色是这个时间片里迁过的字节</span>
        </div>
        <HeatChart heat={heat} />
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>最近事件</h2>
          <div className="filters">
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
              <option value="all">全部类型</option>
              {kinds.map((k) => (
                <option key={k} value={k}>
                  {kindLabel(k)}
                </option>
              ))}
            </select>
            <select value={causeFilter} onChange={(e) => setCauseFilter(e.target.value)}>
              <option value="all">全部原因</option>
              {causes.map((c) => (
                <option key={c} value={c}>
                  {causeLabel(c)}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>类型</th>
                <th>原因</th>
                <th>路径</th>
                <th>大小</th>
                <th>耗时</th>
                <th>地址</th>
              </tr>
            </thead>
            <tbody>
              {visible.slice(0, 80).map((ev) => (
                <tr key={ev.seq}>
                  <td>{formatTime(ev.time)}</td>
                  <td>{kindLabel(ev.kind)}</td>
                  <td>{causeLabel(ev.cause)}</td>
                  <td>{route(ev)}</td>
                  <td>{ev.kind.includes("fault") || ev.kind === "throttling" ? "—" : formatBytes(ev.bytes)}</td>
                  <td>{ev.durationUs > 0 ? `${ev.durationUs.toFixed(0)} µs` : "—"}</td>
                  <td className="mono">{ev.address}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Legend() {
  const items = [
    ["coherence", "一致性"],
    ["prefetch", "推测预取"],
    ["user", "用户预取"],
    ["eviction", "驱逐"],
    ["access_counters", "访问计数"],
    ["explicit", "显式拷贝"],
  ];
  return (
    <div className="legend">
      {items.map(([id, label]) => (
        <span key={id}>
          <i style={{ background: colorOf(id) }} />
          {label}
        </span>
      ))}
    </div>
  );
}

function colorOf(id: string): string {
  const map: Record<string, string> = {
    coherence: "#6ea8fe",
    prefetch: "#3dbe7a",
    user: "#e6b450",
    eviction: "#ef6b6b",
    access_counters: "#c792ea",
    explicit: "#9aa7b8",
  };
  return map[id];
}

function upsertPoint(prev: Point[], point: Point): Point[] {
  if (prev.length === 0) return [point];
  const last = prev[prev.length - 1];
  const next = last.t === point.t ? [...prev.slice(0, -1), point] : [...prev, point];
  return next.length > 600 ? next.slice(next.length - 600) : next;
}

function mergeEvents(prev: EventView[], incoming: EventView[]): EventView[] {
  const map = new Map<number, EventView>();
  for (const ev of prev) map.set(ev.seq, ev);
  for (const ev of incoming) map.set(ev.seq, ev);
  return [...map.values()].sort((a, b) => b.seq - a.seq).slice(0, 200);
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function route(ev: EventView): string {
  if (ev.src && ev.dst && ev.src !== ev.dst) return `${ev.src} → ${ev.dst}`;
  return ev.device || ev.src || "—";
}

function formatTime(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}
