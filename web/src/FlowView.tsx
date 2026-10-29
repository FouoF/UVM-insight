import { useEffect, useRef } from "react";
import { causeColor, causeLabel, formatBytes, formatRate, type Flow, type NodeInfo } from "./api";

type Props = {
  nodes: NodeInfo[];
  flows: Flow[];
};

type Particle = { flow: number; u: number; speed: number; r: number };

type Geom = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export function FlowView({ nodes, flows }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const flowsRef = useRef(flows);
  const nodesRef = useRef(nodes);
  flowsRef.current = flows;
  nodesRef.current = nodes;

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const particles: Particle[] = [];
    let raf = 0;
    let last = performance.now();

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      draw(ctx, canvas, nodesRef.current, flowsRef.current, particles, dt);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="flow" ref={wrapRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}

function draw(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nodes: NodeInfo[],
  flows: Flow[],
  particles: Particle[],
  dt: number,
) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const host = nodes.find((n) => n.kind === "host") ?? {
    id: "host",
    label: "主机内存",
    kind: "host",
    memoryBytes: 0,
    uuid: "",
  };
  const gpus = nodes.filter((n) => n.kind === "gpu");
  const layout = place(w, h, host, gpus);

  for (let i = 0; i < flows.length; i++) {
    const f = flows[i];
    const a = layout.get(f.src);
    const b = layout.get(f.dst);
    if (!a || !b || f.mbps < 0.05) continue;
    const lane = laneOffset(flows, i);
    const color = causeColor[f.cause] ?? causeColor.unknown;
    strokeLink(ctx, a, b, lane, color, f);
    const mid = pointOn(a, b, 0.34 + (i % 4) * 0.1, lane);
    const label = `${formatRate(f.mbps)}  ${causeLabel(f.cause)}`;
    ctx.font = "12px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const width = ctx.measureText(label).width + 10;
    ctx.fillStyle = "rgba(12, 16, 22, 0.82)";
    roundRect(ctx, mid.x - width / 2, mid.y - 16, width, 16, 4);
    ctx.fill();
    ctx.fillStyle = "#d5deea";
    ctx.fillText(label, mid.x, mid.y - 8);
  }

  stepParticles(particles, flows, dt);
  for (const p of particles) {
    const f = flows[p.flow];
    if (!f) continue;
    const a = layout.get(f.src);
    const b = layout.get(f.dst);
    if (!a || !b) continue;
    const lane = laneOffset(flows, p.flow);
    const at = pointOn(a, b, p.u, lane);
    ctx.beginPath();
    ctx.fillStyle = causeColor[f.cause] ?? "#d5deea";
    ctx.arc(at.x, at.y, p.r, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const [id, box] of layout) {
    const node = id === host.id ? host : gpus.find((g) => g.id === id);
    if (node) drawNode(ctx, box, node);
  }
}

function place(w: number, h: number, host: NodeInfo, gpus: NodeInfo[]): Map<string, Geom> {
  const map = new Map<string, Geom>();
  const hostBox = { x: 28, y: h / 2 - 62, w: 168, h: 124 };
  map.set(host.id, hostBox);
  const gh = 86;
  const gap = 16;
  const total = gpus.length * gh + Math.max(0, gpus.length - 1) * gap;
  let y = Math.max(24, (h - total) / 2);
  for (const g of gpus) {
    map.set(g.id, { x: w - 28 - 210, y, w: 210, h: gh });
    y += gh + gap;
  }
  return map;
}

function laneOffset(flows: Flow[], index: number): number {
  const f = flows[index];
  const siblings = flows
    .map((item, i) => ({ item, i }))
    .filter((x) => x.item.src === f.src && x.item.dst === f.dst);
  const pos = siblings.findIndex((x) => x.i === index);
  return (pos - (siblings.length - 1) / 2) * 16;
}

function anchor(box: Geom, side: "left" | "right"): { x: number; y: number } {
  return { x: side === "right" ? box.x + box.w : box.x, y: box.y + box.h / 2 };
}

function ends(a: Geom, b: Geom, lane: number): [{ x: number; y: number }, { x: number; y: number }] {
  const aIsLeft = a.x <= b.x;
  const p0 = anchor(a, aIsLeft ? "right" : "left");
  const p1 = anchor(b, aIsLeft ? "left" : "right");
  p0.y += lane;
  p1.y += lane;
  return [p0, p1];
}

function pointOn(a: Geom, b: Geom, t: number, lane: number): { x: number; y: number } {
  const [p0, p1] = ends(a, b, lane);
  const c1 = { x: (p0.x + p1.x) / 2, y: p0.y };
  const c2 = { x: (p0.x + p1.x) / 2, y: p1.y };
  const u = 1 - t;
  return {
    x: u * u * u * p0.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * p1.x,
    y: u * u * u * p0.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * p1.y,
  };
}

function strokeLink(ctx: CanvasRenderingContext2D, a: Geom, b: Geom, lane: number, color: string, flow: Flow) {
  const [p0, p1] = ends(a, b, lane);
  ctx.beginPath();
  ctx.moveTo(p0.x, p0.y);
  ctx.bezierCurveTo((p0.x + p1.x) / 2, p0.y, (p0.x + p1.x) / 2, p1.y, p1.x, p1.y);
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.85;
  ctx.lineWidth = Math.max(1.5, Math.min(9, 1.2 + Math.log2(1 + flow.mbps)));
  ctx.setLineDash(flow.direction.startsWith("memcpy") ? [5, 5] : []);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}

function stepParticles(particles: Particle[], flows: Flow[], dt: number) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    const f = flows[p.flow];
    if (!f || f.mbps < 0.05) {
      particles.splice(i, 1);
      continue;
    }
    p.u += p.speed * dt;
    if (p.u >= 1) particles.splice(i, 1);
  }
  flows.forEach((f, index) => {
    if (f.mbps < 0.05) return;
    const want = Math.min(24, Math.max(2, Math.round(Math.sqrt(f.mbps) * 1.6)));
    let have = 0;
    for (const p of particles) if (p.flow === index) have++;
    for (let n = have; n < want; n++) {
      particles.push({
        flow: index,
        u: Math.random(),
        speed: 0.18 + Math.min(0.7, f.mbps / 800),
        r: 2.2 + Math.random() * 1.4,
      });
    }
  });
}

function drawNode(ctx: CanvasRenderingContext2D, box: Geom, node: NodeInfo) {
  const gpu = node.kind === "gpu";
  ctx.fillStyle = gpu ? "#14241c" : "#172033";
  ctx.strokeStyle = gpu ? "#3dbe7a" : "#6ea8fe";
  ctx.lineWidth = 1.5;
  roundRect(ctx, box.x, box.y, box.w, box.h, 12);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = gpu ? "#8fd6ae" : "#9ec0ff";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.fillText(gpu ? "DEVICE" : "HOST", box.x + 14, box.y + 22);
  ctx.fillStyle = "#f2f6fb";
  ctx.font = "15px 'Avenir Next', 'Segoe UI', sans-serif";
  ctx.fillText(trim(node.label, 22), box.x + 14, box.y + 46);
  ctx.fillStyle = "#8ea0b5";
  ctx.font = "12px ui-monospace, monospace";
  const sub = gpu && node.memoryBytes > 0 ? formatBytes(node.memoryBytes) : gpu ? node.id : "统一地址空间";
  ctx.fillText(sub, box.x + 14, box.y + 68);
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
