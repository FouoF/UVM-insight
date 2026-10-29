import { useEffect, useRef } from "react";
import { formatBytes, formatRate, type GPUView, type ProcLive } from "./api";

type Props = {
  gpus: GPUView[];
  procs: ProcLive[];
  colorOf: (pid: number) => string;
  focus: number | null;
  onFocus: (pid: number | null) => void;
};

type LaneGeom = {
  key: string;
  proc: ProcLive;
  gpu: string;
  top: number;
  up: number;
  down: number;
  x0: number;
  x1: number;
  rateIn: number;
  rateOut: number;
  quota: number;
  device: string;
};

type GroupGeom = { gpu: GPUView; top: number; bottom: number; procs: number; quota: number; resident: number };

type Track = { shown: number; acc: number };
type Particle = { key: string; dir: 1 | -1; x: number; speed: number; r: number; jitter: number };

const PAD = 18;
const HOST_W = 176;
const GPU_W = 210;
const LANE_H = 66;
const GROUP_GAP = 14;
const HEAD = 8;

export function flowHeight(gpus: GPUView[], procs: ProcLive[]): number {
  const groups = Math.max(1, gpus.length);
  const lanes = Math.max(1, procs.reduce((n, p) => n + Math.max(1, p.lanes.length), 0));
  return Math.max(240, PAD * 2 + HEAD + lanes * LANE_H + (groups - 1) * GROUP_GAP);
}

export function FlowView({ gpus, procs, colorOf, focus, onFocus }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const state = useRef({ gpus, procs, colorOf, focus, lanes: [] as LaneGeom[] });
  state.current.gpus = gpus;
  state.current.procs = procs;
  state.current.colorOf = colorOf;
  state.current.focus = focus;

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tracks = new Map<string, Track>();
    const particles: Particle[] = [];
    let raf = 0;
    let last = performance.now();

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const s = state.current;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.textAlign = "left";
      const { lanes, groups } = layout(w, s.gpus, s.procs);
      s.lanes = lanes;
      step(lanes, tracks, particles, dt);
      drawHost(ctx, h, s.procs);
      for (const g of groups) drawGPU(ctx, w, g);
      for (const lane of lanes) drawLane(ctx, lane, s.colorOf(lane.proc.pid), s.focus, tracks);
      drawParticles(ctx, lanes, particles, s.colorOf, s.focus);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const x = e.clientX - rect.left;
    const hit = state.current.lanes.find((l) => y >= l.top && y < l.top + LANE_H && x >= l.x0 && x <= l.x1);
    if (!hit) {
      onFocus(null);
      return;
    }
    onFocus(state.current.focus === hit.proc.pid ? null : hit.proc.pid);
  };

  return (
    <div className="flow" ref={wrapRef} style={{ height: flowHeight(gpus, procs) }}>
      <canvas ref={canvasRef} onClick={onClick} />
    </div>
  );
}

function layout(w: number, gpus: GPUView[], procs: ProcLive[]): { lanes: LaneGeom[]; groups: GroupGeom[] } {
  const known = new Map(gpus.map((g) => [g.id, g]));
  const order = gpus.map((g) => g.id);
  const byGpu = new Map<string, { proc: ProcLive; lane: ProcLive["lanes"][number] | null }[]>();
  for (const p of procs) {
    const lanes = p.lanes.length ? p.lanes : [null];
    for (const lane of lanes) {
      const id = lane?.gpu ?? order[0] ?? "gpu";
      if (!known.has(id)) {
        known.set(id, { id, name: lane?.deviceLabel ?? "GPU", uuid: "", memoryBytes: 0 });
        order.push(id);
      }
      if (!byGpu.has(id)) byGpu.set(id, []);
      byGpu.get(id)!.push({ proc: p, lane });
    }
  }
  const x0 = PAD + HOST_W + 10;
  const x1 = w - PAD - GPU_W - 10;
  const lanes: LaneGeom[] = [];
  const groups: GroupGeom[] = [];
  let y = PAD + HEAD;
  for (const id of order) {
    const rows = byGpu.get(id) ?? [];
    if (!rows.length) continue;
    const top = y;
    let quota = 0;
    let resident = 0;
    for (const { proc, lane } of rows) {
      if (proc.residentKnown && (!lane || proc.lanes[0]?.gpu === lane.gpu)) resident += proc.residentBytes;
      const q = lane?.quotaBytes ?? 0;
      quota += q;
      lanes.push({
        key: `${proc.pid}|${id}`,
        proc,
        gpu: id,
        top: y,
        up: y + 36,
        down: y + 52,
        x0,
        x1,
        rateIn: lane ? lane.htod + lane.memcpyHtod : 0,
        rateOut: lane ? lane.dtoh + lane.memcpyDtoh : 0,
        quota: q,
        device: lane?.deviceLabel ?? "",
      });
      y += LANE_H;
    }
    groups.push({ gpu: known.get(id)!, top, bottom: y, procs: rows.length, quota, resident });
    y += GROUP_GAP;
  }
  return { lanes, groups };
}

// Rates arrive four times a second; easing them keeps the stream steady
// between updates and lets it fade out instead of stopping dead.
function step(lanes: LaneGeom[], tracks: Map<string, Track>, particles: Particle[], dt: number) {
  const ease = 1 - Math.exp(-dt / 0.3);
  const live = new Set<string>();
  for (const lane of lanes) {
    for (const dir of [1, -1] as const) {
      const key = `${lane.key}|${dir}`;
      live.add(key);
      const target = dir === 1 ? lane.rateIn : lane.rateOut;
      let t = tracks.get(key);
      if (!t) {
        t = { shown: target, acc: Math.random() };
        tracks.set(key, t);
        if (target >= 0.02) {
          const lg = Math.log10(1 + target);
          const span = (lane.x1 - lane.x0) / (150 + lg * 70);
          const n = Math.round(emitRate(lg) * span);
          for (let i = 0; i < n; i++) particles.push(spawn(key, dir, lg, Math.random()));
        }
      }
      t.shown += (target - t.shown) * ease;
      if (t.shown < 0.02) continue;
      const lg = Math.log10(1 + t.shown);
      t.acc += emitRate(lg) * dt;
      while (t.acc >= 1) {
        t.acc -= 1;
        particles.push(spawn(key, dir, lg, 0));
      }
    }
  }
  for (const key of tracks.keys()) if (!live.has(key)) tracks.delete(key);
  const len = lanes.length ? lanes[0].x1 - lanes[0].x0 : 1;
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += (p.speed * dt) / Math.max(1, len);
    if (p.x >= 1 || !live.has(p.key)) particles.splice(i, 1);
  }
}

function emitRate(lg: number): number {
  return Math.min(36, 2.5 + lg * 8);
}

function spawn(key: string, dir: 1 | -1, lg: number, x: number): Particle {
  return {
    key,
    dir,
    x,
    speed: 150 + lg * 70 + Math.random() * 30,
    r: 1.8 + Math.min(2.6, lg * 0.9) + Math.random() * 0.6,
    jitter: (Math.random() - 0.5) * 3,
  };
}

function drawHost(ctx: CanvasRenderingContext2D, h: number, procs: ProcLive[]) {
  const box = { x: PAD, y: PAD, w: HOST_W, h: h - PAD * 2 };
  panel(ctx, box, "#172033", "#6ea8fe");
  const host = procs.reduce((n, p) => n + Math.max(0, p.managedBytes - (p.residentKnown ? p.residentBytes : 0)), 0);
  const into = procs.reduce((n, p) => n + p.htod + p.memcpy, 0);
  const back = procs.reduce((n, p) => n + p.dtoh, 0);
  let y = box.y + 24;
  text(ctx, "HOST", box.x + 14, y, "#9ec0ff", "11px ui-monospace, monospace");
  y += 24;
  text(ctx, "主机内存", box.x + 14, y, "#f2f6fb", "15px 'Avenir Next', 'PingFang SC', sans-serif");
  y += 26;
  text(ctx, `主机侧 ${formatBytes(host)}`, box.x + 14, y, "#8ea0b5", "12px ui-monospace, monospace");
  if (box.h > 130) {
    y += 20;
    text(ctx, `→ ${formatRate(into)}`, box.x + 14, y, "#8ea0b5", "12px ui-monospace, monospace");
    y += 18;
    text(ctx, `← ${formatRate(back)}`, box.x + 14, y, "#8ea0b5", "12px ui-monospace, monospace");
  }
}

function drawGPU(ctx: CanvasRenderingContext2D, w: number, g: GroupGeom) {
  const box = { x: w - PAD - GPU_W, y: g.top, w: GPU_W, h: g.bottom - g.top };
  panel(ctx, box, "#14241c", "#3dbe7a");
  let y = box.y + 22;
  text(ctx, "GPU", box.x + 14, y, "#8fd6ae", "11px ui-monospace, monospace");
  y += 22;
  text(ctx, trim(g.gpu.name || g.gpu.id, 22), box.x + 14, y, "#f2f6fb", "15px 'Avenir Next', 'PingFang SC', sans-serif");
  if (box.h >= 90) {
    y += 20;
    text(ctx, `驻留 ${formatBytes(g.resident)}（估算）`, box.x + 14, y, "#8fd6ae", "12px ui-monospace, 'PingFang SC', monospace");
  }
  if (box.h >= 110) {
    y += 18;
    const quota = g.quota > 0 ? ` · 配额 ${formatBytes(g.quota)}` : "";
    text(ctx, `${g.procs} 个进程${quota}`, box.x + 14, y, "#8ea0b5", "12px 'PingFang SC', sans-serif");
  }
}

function drawLane(
  ctx: CanvasRenderingContext2D,
  lane: LaneGeom,
  color: string,
  focus: number | null,
  tracks: Map<string, Track>,
) {
  const dim = focus !== null && focus !== lane.proc.pid;
  ctx.globalAlpha = dim ? 0.35 : 1;
  ctx.fillStyle = hexA(color, focus === lane.proc.pid ? 0.13 : 0.06);
  roundRect(ctx, lane.x0, lane.top + 4, lane.x1 - lane.x0, LANE_H - 8, 10);
  ctx.fill();

  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.arc(lane.x0 + 16, lane.top + 18, 4, 0, Math.PI * 2);
  ctx.fill();
  let name = lane.proc.appName;
  if (lane.proc.lanes.length > 1) name += ` · ${lane.device}`;
  if (!lane.proc.active) name += "（已退出）";
  text(ctx, name, lane.x0 + 28, lane.top + 19, "#f2f6fb", "13px 'Avenir Next', 'PingFang SC', sans-serif");

  ctx.textAlign = "right";
  const parts = [`迁入 ${formatRate(lane.rateIn)}`, `迁回 ${formatRate(lane.rateOut)}`];
  if (lane.proc.residentKnown && lane.proc.lanes[0]?.gpu === lane.gpu) parts.unshift(`驻留 ${formatBytes(lane.proc.residentBytes)}`);
  const firstLane = !lane.proc.lanes.length || lane.proc.lanes[0].gpu === lane.gpu;
  if (firstLane && lane.proc.gpuFaults > 0) parts.push(`缺页 ${compact(lane.proc.gpuFaults)}/s`);
  text(ctx, parts.join("   "), lane.x1 - 14, lane.top + 19, "#b8c5d6", "12px ui-monospace, 'PingFang SC', monospace");
  ctx.textAlign = "left";

  for (const dir of [1, -1] as const) {
    const y = dir === 1 ? lane.up : lane.down;
    const shown = tracks.get(`${lane.key}|${dir}`)?.shown ?? 0;
    const glow = Math.min(1, Math.log10(1 + shown) / 3);
    ctx.strokeStyle = hexA(color, 0.16 + glow * 0.4);
    ctx.lineWidth = 2 + glow * 3;
    ctx.beginPath();
    ctx.moveTo(lane.x0 + 12, y);
    ctx.lineTo(lane.x1 - 12, y);
    ctx.stroke();
    arrow(ctx, dir === 1 ? lane.x1 - 12 : lane.x0 + 12, y, dir, hexA(color, 0.3 + glow * 0.6));
  }
  ctx.globalAlpha = 1;
}

function drawParticles(
  ctx: CanvasRenderingContext2D,
  lanes: LaneGeom[],
  particles: Particle[],
  colorOf: (pid: number) => string,
  focus: number | null,
) {
  const byKey = new Map(lanes.map((l) => [l.key, l]));
  ctx.lineCap = "round";
  for (const p of particles) {
    const laneKey = p.key.slice(0, p.key.lastIndexOf("|"));
    const lane = byKey.get(laneKey);
    if (!lane) continue;
    const color = colorOf(lane.proc.pid);
    const a = lane.x0 + 12;
    const b = lane.x1 - 12;
    const u = p.dir === 1 ? p.x : 1 - p.x;
    const x = a + (b - a) * u;
    const y = (p.dir === 1 ? lane.up : lane.down) + p.jitter;
    const fade = Math.min(1, p.x * 8, (1 - p.x) * 8);
    const dim = focus !== null && focus !== lane.proc.pid ? 0.3 : 1;
    ctx.globalAlpha = fade * dim;
    const tail = 8 + p.r * 5;
    const grad = ctx.createLinearGradient(x - p.dir * tail, y, x, y);
    grad.addColorStop(0, hexA(color, 0));
    grad.addColorStop(1, hexA(color, 0.95));
    ctx.strokeStyle = grad;
    ctx.lineWidth = p.r * 1.8;
    ctx.beginPath();
    ctx.moveTo(x - p.dir * tail, y);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.fillStyle = hexA(color, 0.35);
    ctx.beginPath();
    ctx.arc(x, y, p.r * 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#f6f9fc";
    ctx.beginPath();
    ctx.arc(x, y, p.r * 0.8, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function panel(ctx: CanvasRenderingContext2D, box: { x: number; y: number; w: number; h: number }, fill: string, stroke: string) {
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  roundRect(ctx, box.x, box.y, box.w, box.h, 12);
  ctx.fill();
  ctx.stroke();
}

function arrow(ctx: CanvasRenderingContext2D, x: number, y: number, dir: 1 | -1, color: string) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x + dir * 6, y);
  ctx.lineTo(x - dir * 3, y - 5);
  ctx.lineTo(x - dir * 3, y + 5);
  ctx.closePath();
  ctx.fill();
}

function text(ctx: CanvasRenderingContext2D, s: string, x: number, y: number, color: string, font: string) {
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.textBaseline = "middle";
  ctx.fillText(s, x, y);
}

function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return n.toFixed(n >= 10 ? 0 : 1);
}

function hexA(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function trim(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
