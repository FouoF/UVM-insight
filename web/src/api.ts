export type GpuInfo = {
  deviceId: number;
  uuid: string;
  name: string;
  totalMemoryBytes: number;
};

export type ProcessInfo = {
  pid: number;
  comm: string;
  appName: string;
  hostname: string;
  cudaVersion: string;
  driverVersion: string;
  gpus: GpuInfo[];
  managedBytes: number;
  active: boolean;
  dropped: number;
  lastSeen: number;
};

export type Flow = {
  src: string;
  dst: string;
  direction: string;
  cause: string;
  mbps: number;
};

export type NodeInfo = {
  id: string;
  label: string;
  kind: string;
  memoryBytes: number;
  uuid: string;
};

export type Point = {
  t: number;
  htod: number;
  dtoh: number;
  dtod: number;
  memcpyHtod: number;
  memcpyDtoh: number;
  memcpyDtod: number;
  cpuFaults: number;
  gpuFaults: number;
  thrashing: number;
  throttling: number;
  remoteMap: number;
};

export type EventView = {
  seq: number;
  time: number;
  kind: string;
  cause: string;
  direction: string;
  bytes: number;
  address: string;
  src: string;
  dst: string;
  device: string;
  durationUs: number;
};

export type KernelView = {
  name: string;
  device: string;
  start: number;
  end: number;
};

export type Snapshot = {
  pid: number;
  managedBytes: number;
  dropped: number;
  nodes: NodeInfo[];
  flows: Flow[];
  points: Point[];
  kernels: KernelView[];
  events: EventView[];
};

export type HeatCell = {
  addr: string;
  t: number;
  bytes: number;
  direction: string;
};

export type Heatmap = {
  bucketBytes: number;
  timeBucketSec: number;
  cells: HeatCell[];
};

export type StreamMsg = {
  now: number;
  managedBytes: number;
  dropped: number;
  active: boolean;
  flows: Flow[];
  point: Point | null;
  kernels: KernelView[];
  events: EventView[];
  lastSeq: number;
};

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${res.status} ${url}`);
  }
  return res.json() as Promise<T>;
}

export function fetchProcesses(): Promise<ProcessInfo[]> {
  return getJSON("/api/processes");
}

export function fetchSnapshot(pid: number | "all"): Promise<Snapshot> {
  const q = pid === "all" ? "all" : String(pid);
  return getJSON(`/api/snapshot?pid=${q}`);
}

export function fetchHeatmap(pid: number | "all"): Promise<Heatmap> {
  const q = pid === "all" ? "all" : String(pid);
  return getJSON(`/api/heatmap?pid=${q}`);
}

export function streamURL(pid: number | "all"): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = pid === "all" ? "all" : String(pid);
  return `${proto}://${location.host}/api/stream?pid=${q}`;
}

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v >= 10 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function formatRate(mbps: number): string {
  if (mbps >= 1000) return `${(mbps / 1000).toFixed(2)} GB/s`;
  if (mbps >= 10) return `${mbps.toFixed(0)} MB/s`;
  return `${mbps.toFixed(1)} MB/s`;
}

const kindText: Record<string, string> = {
  htod: "迁入显存",
  dtoh: "迁回主机",
  dtod: "卡间迁移",
  cpu_page_fault: "CPU 缺页",
  gpu_page_fault: "GPU 缺页",
  thrashing: "抖动",
  throttling: "节流",
  remote_map: "远端映射",
  memcpy_htod: "显式拷贝 HtoD",
  memcpy_dtoh: "显式拷贝 DtoH",
  memcpy_dtod: "显式拷贝 DtoD",
  memcpy_other: "显式拷贝",
};

const causeText: Record<string, string> = {
  user: "用户预取",
  coherence: "一致性",
  prefetch: "推测预取",
  eviction: "驱逐",
  access_counters: "访问计数",
  explicit: "显式拷贝",
  unknown: "未标记",
};

export function kindLabel(kind: string): string {
  return kindText[kind] ?? kind;
}

export function causeLabel(cause: string): string {
  return causeText[cause] ?? cause;
}

export const causeColor: Record<string, string> = {
  user: "#e6b450",
  coherence: "#6ea8fe",
  prefetch: "#3dbe7a",
  eviction: "#ef6b6b",
  access_counters: "#c792ea",
  explicit: "#9aa7b8",
  unknown: "#8ea0b5",
};
