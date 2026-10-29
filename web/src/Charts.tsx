import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { Heatmap, Point } from "./api";

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
    chart.current?.setOption(option, { replaceMerge: ["series"] });
  }, [option]);
  return ref;
}

export type ProcSeries = { pid: number; name: string; color: string; points: Point[] };

function perSecond(points: Point[], pick: (p: Point) => number): [number, number][] {
  return points.map((p) => [p.t, round(pick(p))]);
}

function timeAxis() {
  return { type: "time", axisLabel: { color: text, hideOverlap: true }, axisLine: { lineStyle: { color: split } } };
}

function valueAxis(name: string) {
  return {
    type: "value",
    name,
    nameTextStyle: { color: text },
    axisLabel: { color: text },
    splitLine: { lineStyle: { color: split } },
  };
}

export function ProcBandwidthChart({ procs }: { procs: ProcSeries[] }) {
  const series = procs.flatMap((p) => [
    {
      name: `${p.name} 迁入`,
      type: "line",
      stack: "in",
      smooth: true,
      showSymbol: false,
      color: p.color,
      lineStyle: { width: 1.5 },
      areaStyle: { opacity: 0.28 },
      data: perSecond(p.points, (x) => x.htod + x.memcpyHtod),
    },
    {
      name: `${p.name} 迁回`,
      type: "line",
      stack: "out",
      smooth: true,
      showSymbol: false,
      color: p.color,
      lineStyle: { width: 1.5, type: "dashed" },
      areaStyle: { opacity: 0.1 },
      data: perSecond(p.points, (x) => -(x.dtoh + x.memcpyDtoh)),
    },
  ]);
  const ref = useChart({
    backgroundColor: "transparent",
    animation: false,
    tooltip: {
      trigger: "axis",
      valueFormatter: (v: number) => `${Math.abs(v).toFixed(1)} MB/s`,
    },
    legend: { show: false },
    grid: { left: 56, right: 16, top: 34, bottom: 28 },
    xAxis: timeAxis(),
    yAxis: {
      ...valueAxis("MB/s  ↑迁入 ↓迁回"),
      axisLabel: { color: text, formatter: (v: number) => String(Math.abs(v)) },
    },
    series,
  });
  return <div ref={ref} className="chart" />;
}

export function ResidentChart({ procs }: { procs: ProcSeries[] }) {
  const ref = useChart({
    backgroundColor: "transparent",
    animation: false,
    tooltip: {
      trigger: "axis",
      valueFormatter: (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)} GB` : `${v.toFixed(0)} MB`),
    },
    legend: { show: false },
    grid: { left: 56, right: 16, top: 30, bottom: 28 },
    xAxis: timeAxis(),
    yAxis: {
      ...valueAxis("GB"),
      axisLabel: { color: text, formatter: (v: number) => (v / 1000).toFixed(1) },
    },
    series: procs.map((p) => ({
      name: p.name,
      type: "line",
      stack: "resident",
      step: "end",
      showSymbol: false,
      color: p.color,
      lineStyle: { width: 1 },
      areaStyle: { opacity: 0.45 },
      data: perSecond(p.points, (x) => x.residentMb ?? 0),
    })),
  });
  return <div ref={ref} className="chart chart-short" />;
}

export function ProcFaultChart({ procs }: { procs: ProcSeries[] }) {
  const ref = useChart({
    backgroundColor: "transparent",
    animation: false,
    tooltip: { trigger: "axis", valueFormatter: (v: number) => `${v.toFixed(0)} 次/秒` },
    legend: { show: false },
    grid: { left: 56, right: 16, top: 34, bottom: 28 },
    xAxis: timeAxis(),
    yAxis: valueAxis("次/秒"),
    series: procs.map((p) => ({
      name: p.name,
      type: "line",
      smooth: true,
      showSymbol: false,
      color: p.color,
      data: perSecond(p.points, (x) => x.gpuFaults + x.cpuFaults),
    })),
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

function round(v: number): number {
  return Math.round(v * 100) / 100;
}
