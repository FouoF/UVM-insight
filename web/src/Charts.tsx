import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { Heatmap, KernelView, Point } from "./api";

const text = "#8ea0b5";
const split = "#243041";

function useChart(option: echarts.EChartsCoreOption) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const c = echarts.init(ref.current);
    chart.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, []);
  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true });
  }, [option]);
  return ref;
}

export function BandwidthChart({ points, kernels }: { points: Point[]; kernels: KernelView[] }) {
  const areas = kernels.slice(-12).map((k) => [
    { xAxis: k.start, name: k.name },
    { xAxis: k.end },
  ]);
  const ref = useChart({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { textStyle: { color: text }, top: 0, data: ["迁入显存", "迁回主机", "卡间", "显式拷贝"] },
    grid: { left: 52, right: 16, top: 36, bottom: 28 },
    xAxis: { type: "time", axisLabel: { color: text }, axisLine: { lineStyle: { color: split } } },
    yAxis: {
      type: "value",
      name: "MB/s",
      nameTextStyle: { color: text },
      axisLabel: { color: text },
      splitLine: { lineStyle: { color: split } },
    },
    series: [
      series("迁入显存", "#6ea8fe", points.map((p) => [p.t, round(p.htod)]), "uvm"),
      series("迁回主机", "#3dbe7a", points.map((p) => [p.t, round(p.dtoh)]), "uvm"),
      series("卡间", "#c792ea", points.map((p) => [p.t, round(p.dtod)]), ""),
      series("显式拷贝", "#e6b450", points.map((p) => [p.t, round(p.memcpyHtod + p.memcpyDtoh + p.memcpyDtod)]), ""),
      {
        name: "kernel",
        type: "line",
        data: [],
        markArea: {
          itemStyle: { color: "rgba(110,168,254,0.08)" },
          label: { color: text, fontSize: 10 },
          data: areas,
        },
      },
    ],
  });
  return <div ref={ref} className="chart" />;
}

export function FaultChart({ points }: { points: Point[] }) {
  const ref = useChart({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { textStyle: { color: text }, top: 0 },
    grid: { left: 46, right: 16, top: 36, bottom: 28 },
    xAxis: { type: "time", axisLabel: { color: text }, axisLine: { lineStyle: { color: split } } },
    yAxis: {
      type: "value",
      name: "次/秒",
      nameTextStyle: { color: text },
      axisLabel: { color: text },
      splitLine: { lineStyle: { color: split } },
    },
    series: [
      line("CPU 缺页", "#6ea8fe", points.map((p) => [p.t, p.cpuFaults])),
      line("GPU 缺页", "#e6b450", points.map((p) => [p.t, p.gpuFaults])),
      line("抖动", "#ef6b6b", points.map((p) => [p.t, p.thrashing])),
      line("节流", "#c792ea", points.map((p) => [p.t, p.throttling])),
    ],
  });
  return <div ref={ref} className="chart" />;
}

export function HeatChart({ heat }: { heat: Heatmap }) {
  const totals = new Map<string, number>();
  for (const c of heat.cells) totals.set(c.addr, (totals.get(c.addr) ?? 0) + c.bytes);
  const rows = [...totals.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 14)
    .map((e) => e[0]);
  const rowIndex = new Map(rows.map((addr, i) => [addr, i]));
  const times = [...new Set(heat.cells.map((c) => c.t))].sort((a, b) => a - b);
  const timeIndex = new Map(times.map((t, i) => [t, i]));
  const data = heat.cells
    .filter((c) => rowIndex.has(c.addr))
    .map((c) => [timeIndex.get(c.t), rowIndex.get(c.addr), +(c.bytes / 1e6).toFixed(2)]);
  const ref = useChart({
    backgroundColor: "transparent",
    tooltip: {
      formatter: (p: unknown) => {
        const item = p as { data: number[] };
        const t = times[item.data[0]] ?? 0;
        const addr = rows[item.data[1]] ?? "";
        return `${new Date(t * 1000).toLocaleTimeString()}<br/>${addr}<br/>${item.data[2]} MB`;
      },
    },
    grid: { left: 118, right: 24, top: 16, bottom: 28 },
    xAxis: {
      type: "category",
      data: times.map((t) => new Date(t * 1000).toLocaleTimeString()),
      axisLabel: { color: text, hideOverlap: true },
      axisLine: { lineStyle: { color: split } },
    },
    yAxis: {
      type: "category",
      data: rows,
      axisLabel: { color: text, fontFamily: "ui-monospace, monospace", fontSize: 11 },
    },
    visualMap: {
      min: 0,
      max: Math.max(1, ...data.map((d) => Number(d[2]))),
      calculable: false,
      orient: "horizontal",
      right: 8,
      top: 0,
      itemWidth: 10,
      itemHeight: 80,
      textStyle: { color: text },
      inRange: { color: ["#142033", "#1f6f8f", "#e6b450", "#ef6b6b"] },
    },
    series: [{ type: "heatmap", data, progressive: 2000 }],
  });
  return <div ref={ref} className="chart chart-heat" />;
}

function series(name: string, color: string, data: (number | string)[][], stack: string) {
  return {
    name,
    type: "line",
    stack: stack || undefined,
    smooth: true,
    showSymbol: false,
    color,
    areaStyle: stack ? { opacity: 0.25 } : undefined,
    data,
  };
}

function line(name: string, color: string, data: number[][]) {
  return { name, type: "line", smooth: true, showSymbol: false, color, data };
}

function round(v: number): number {
  return Math.round(v * 100) / 100;
}
