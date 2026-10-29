import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  causeLabel,
  fetchHeatmap,
  formatBytes,
  formatRate,
  kindLabel,
  liveURL,
  type EventView,
  type GPUView,
  type Heatmap,
  type Overview,
  type Point,
  type ProcLive,
} from "./api";
import { HeatChart, ProcBandwidthChart, ProcFaultChart, ResidentChart, type ProcSeries } from "./Charts";
import { FlowView } from "./FlowView";

const emptyHeat: Heatmap = { bucketBytes: 2 << 20, timeBucketSec: 5, cells: [] };
const palette = ["#6ea8fe", "#3dbe7a", "#e6b450", "#ef6b6b", "#c792ea", "#4fd1c5", "#f78c6c", "#a3be8c"];
const keepSeconds = 120;
const countKinds = new Set(["cpu_page_fault", "gpu_page_fault", "thrashing", "throttling", "remote_map"]);

export function App() {
  const [gpus, setGpus] = useState<GPUView[]>([]);
  const [procs, setProcs] = useState<ProcLive[]>([]);
  const [events, setEvents] = useState<EventView[]>([]);
  const [heat, setHeat] = useState<Heatmap>(emptyHeat);
  const [connected, setConnected] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);
  const [kindFilter, setKindFilter] = useState("all");
  const [tick, setTick] = useState(0);
  const series = useRef(new Map<number, Point[]>());
  const colors = useRef(new Map<number, string>());

  const colorOf = useCallback((pid: number) => {
    let c = colors.current.get(pid);
    if (!c) {
      c = palette[colors.current.size % palette.length];
      colors.current.set(pid, c);
    }
    return c;
  }, []);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry = 0;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(liveURL());
      let first = true;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data) as Overview;
        const seen = new Set<number>();
        for (const p of msg.processes) {
          seen.add(p.pid);
          colorOf(p.pid);
          series.current.set(p.pid, first ? p.series : upsert(series.current.get(p.pid) ?? [], p.series));
        }
        for (const pid of series.current.keys()) if (!seen.has(pid)) series.current.delete(pid);
        setGpus(msg.gpus);
        setProcs(msg.processes);
        setEvents((prev) => (first ? [...msg.events].reverse() : mergeEvents(prev, msg.events)));
        first = false;
      };
    };
    connect();
    const charts = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => {
      closed = true;
      clearTimeout(retry);
      clearInterval(charts);
      ws?.close();
    };
  }, [colorOf]);

  useEffect(() => {
    let stop = false;
    const load = () =>
      fetchHeatmap(focus ?? "all")
        .then((h) => !stop && setHeat(h))
        .catch(() => !stop && setHeat(emptyHeat));
    load();
    const id = setInterval(load, 3000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [focus]);

  useEffect(() => {
    if (focus !== null && !procs.some((p) => p.pid === focus)) setFocus(null);
  }, [procs, focus]);

  const chartData: ProcSeries[] = useMemo(
    () =>
      procs
        .filter((p) => focus === null || p.pid === focus)
        .map((p) => ({ pid: p.pid, name: p.appName, color: colorOf(p.pid), points: series.current.get(p.pid) ?? [] })),
    // Charts refresh once a second; the flow view handles the 4 Hz updates.
    [tick, focus, procs.length, colorOf],
  );

  const visible = useMemo(
    () =>
      events.filter(
        (ev) => (focus === null || ev.pid === focus) && (kindFilter === "all" || ev.kind === kindFilter),
      ),
    [events, focus, kindFilter],
  );
  const kinds = useMemo(() => [...new Set(events.map((e) => e.kind))], [events]);
  const totals = useMemo(
    () =>
      procs.reduce(
        (t, p) => ({
          managed: t.managed + p.managedBytes,
          resident: t.resident + (p.residentKnown ? p.residentBytes : 0),
          into: t.into + p.htod + p.memcpy,
          back: t.back + p.dtoh,
          faults: t.faults + p.gpuFaults + p.cpuFaults,
        }),
        { managed: 0, resident: 0, into: 0, back: 0, faults: 0 },
      ),
    [procs],
  );
  const focused = procs.find((p) => p.pid === focus);

  return (
    <div className="shell">
      <header className="top">
        <div>
          <div className="kicker">NVIDIA unified memory</div>
          <h1>UVM Insight</h1>
          <p className="lede">每个进程在显存和主机内存之间的迁移与缺页，数据来自 CUPTI，实时速率取最近 1 秒，曲线为逐秒值。</p>
        </div>
        <div className="status">
          <span className={connected ? "dot on" : "dot"} />
          {connected ? "实时" : "正在连接"}
        </div>
      </header>

      <div className="summary">
        <Stat label="进程" value={String(procs.filter((p) => p.active).length)} />
        <Stat label="显存驻留（估算）" value={formatBytes(totals.resident)} />
        <Stat label="主机侧" value={formatBytes(Math.max(0, totals.managed - totals.resident))} />
        <Stat label="迁入显存" value={formatRate(totals.into)} accent="#6ea8fe" />
        <Stat label="迁回主机" value={formatRate(totals.back)} accent="#3dbe7a" />
        <Stat label="缺页" value={`${compact(totals.faults)}/s`} accent="#e6b450" />
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2>数据流动</h2>
          <span className="hint">每行一个进程，上轨迁入显存 →，下轨迁回主机 ←；点击一行聚焦该进程</span>
        </div>
        {procs.length === 0 ? (
          <div className="empty">
            还没有进程。用 CUDA_INJECTION64_PATH 启动应用后会自动出现在这里，或者用 --mock 看演示数据。
          </div>
        ) : (
          <FlowView gpus={gpus} procs={procs} colorOf={colorOf} focus={focus} onFocus={setFocus} />
        )}
      </section>

      {procs.length > 0 ? (
        <section className="cards">
          {procs.map((p) => (
            <ProcCard
              key={p.pid}
              proc={p}
              color={colorOf(p.pid)}
              points={series.current.get(p.pid) ?? []}
              focused={focus === p.pid}
              dimmed={focus !== null && focus !== p.pid}
              onClick={() => setFocus(focus === p.pid ? null : p.pid)}
            />
          ))}
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-head">
          <h2>显存驻留（估算）</h2>
          <span className="hint">各进程驻留在显存里的统一内存，堆叠显示；一个进程换入时其他进程被挤出</span>
        </div>
        <ResidentChart procs={chartData} />
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panel-head">
            <h2>迁移带宽</h2>
            <Legend procs={chartData} />
          </div>
          <ProcBandwidthChart procs={chartData} />
        </div>
        <div className="panel">
          <div className="panel-head">
            <h2>缺页</h2>
            <Legend procs={chartData} />
          </div>
          <ProcFaultChart procs={chartData} />
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>地址热度{focused ? ` · ${focused.appName}` : ""}</h2>
          <span className="hint">2 MB 地址段，颜色表示该时间片内迁移的字节数</span>
        </div>
        <HeatChart heat={heat} />
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>最近事件{focused ? ` · ${focused.appName}` : ""}</h2>
          <div className="filters">
            <span className="hint">同一 2 MB 地址段在 200 ms 内的记录合并为一行</span>
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
              <option value="all">全部类型</option>
              {kinds.map((k) => (
                <option key={k} value={k}>
                  {kindLabel(k)}
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
                <th>进程</th>
                <th>类型</th>
                <th>原因</th>
                <th>路径</th>
                <th>大小 / 次数</th>
                <th>地址</th>
              </tr>
            </thead>
            <tbody>
              {visible.slice(0, 100).map((ev) => (
                <tr key={ev.seq}>
                  <td>{formatTime(ev.time)}</td>
                  <td>
                    <span className="swatch" style={{ background: colorOf(ev.pid) }} />
                    {ev.app}
                  </td>
                  <td>{kindLabel(ev.kind)}</td>
                  <td>{causeLabel(ev.cause)}</td>
                  <td>{route(ev)}</td>
                  <td>{countKinds.has(ev.kind) ? `${Math.max(1, ev.bytes)} 次` : formatBytes(ev.bytes)}</td>
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

function ProcCard({
  proc,
  color,
  points,
  focused,
  dimmed,
  onClick,
}: {
  proc: ProcLive;
  color: string;
  points: Point[];
  focused: boolean;
  dimmed: boolean;
  onClick: () => void;
}) {
  const quota = proc.lanes.reduce((n, l) => n + l.quotaBytes, 0);
  const resident = proc.residentKnown ? proc.residentBytes : 0;
  const cap = quota > 0 ? Math.min(quota, proc.managedBytes || quota) : proc.managedBytes;
  const ratio = cap > 0 ? resident / cap : 0;
  const host = Math.max(0, proc.managedBytes - resident);
  const gpu = proc.lanes.map((l) => l.deviceLabel).join(", ") || "—";
  return (
    <button
      className={`card${focused ? " on" : ""}${dimmed ? " dim" : ""}`}
      style={{ borderTopColor: color }}
      onClick={onClick}
    >
      <div className="card-head">
        <span className="swatch" style={{ background: color }} />
        <strong>{proc.appName}</strong>
        <span className={proc.active ? "badge" : "badge off"}>{proc.active ? "运行中" : "已退出"}</span>
      </div>
      <div className="card-meta">
        {proc.hostname || `pid ${proc.pid}`} · {gpu}
        {proc.cudaVersion ? ` · CUDA ${proc.cudaVersion}` : ""}
      </div>
      <div className="meter">
        <div className="meter-label">
          <span>显存驻留（估算）</span>
          <span>
            {proc.residentKnown ? formatBytes(resident) : "—"}
            {quota > 0 ? ` / 配额 ${formatBytes(quota)}` : ""}
          </span>
        </div>
        <div className="meter-bar">
          <i style={{ width: `${Math.min(100, ratio * 100)}%`, background: color }} />
        </div>
        <div className="meter-sub">
          申请 {formatBytes(proc.managedBytes)}，主机侧 {formatBytes(host)}
        </div>
      </div>
      <div className="card-stats">
        <Mini label="迁入" value={formatRate(proc.htod + proc.memcpy)} />
        <Mini label="迁回" value={formatRate(proc.dtoh)} />
        <Mini label="缺页" value={`${compact(proc.gpuFaults + proc.cpuFaults)}/s`} />
        <Mini label="累计迁移" value={formatBytes(proc.migratedHtod + proc.migratedDtoh)} />
      </div>
      <Spark points={points} color={color} />
    </button>
  );
}

function Spark({ points, color }: { points: Point[]; color: string }) {
  const w = 280;
  const h = 44;
  if (points.length < 2) return <svg className="spark" viewBox={`0 0 ${w} ${h}`} />;
  const into = points.map((p) => p.htod + p.memcpyHtod);
  const back = points.map((p) => p.dtoh + p.memcpyDtoh);
  const max = Math.max(1, ...into, ...back);
  const mid = h / 2;
  const path = (vals: number[], sign: number) =>
    vals
      .map((v, i) => `${i === 0 ? "M" : "L"}${((i / (vals.length - 1)) * w).toFixed(1)},${(mid - (sign * v * (mid - 2)) / max).toFixed(1)}`)
      .join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <line x1="0" x2={w} y1={mid} y2={mid} stroke="#2a3646" />
      <path d={`${path(into, 1)} L${w},${mid} L0,${mid} Z`} fill={color} opacity="0.3" />
      <path d={path(into, 1)} fill="none" stroke={color} strokeWidth="1.2" />
      <path d={`${path(back, -1)} L${w},${mid} L0,${mid} Z`} fill={color} opacity="0.12" />
      <path d={path(back, -1)} fill="none" stroke={color} strokeWidth="1" strokeDasharray="3 2" />
    </svg>
  );
}

function Legend({ procs }: { procs: ProcSeries[] }) {
  return (
    <div className="legend">
      {procs.map((p) => (
        <span key={p.pid}>
          <i style={{ background: p.color }} />
          {p.name}
        </span>
      ))}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong style={accent ? { color: accent } : undefined}>{value}</strong>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="mini">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function upsert(prev: Point[], incoming: Point[]): Point[] {
  const next = [...prev];
  for (const pt of incoming) {
    let i = next.length - 1;
    while (i >= 0 && next[i].t > pt.t) i--;
    if (i >= 0 && next[i].t === pt.t) next[i] = pt;
    else next.splice(i + 1, 0, pt);
  }
  return next.length > keepSeconds ? next.slice(next.length - keepSeconds) : next;
}

function mergeEvents(prev: EventView[], incoming: EventView[]): EventView[] {
  if (!incoming.length) return prev;
  const next = [...incoming].reverse().concat(prev);
  return next.length > 400 ? next.slice(0, 400) : next;
}

function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return n.toFixed(n >= 10 || n === 0 ? 0 : 1);
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
