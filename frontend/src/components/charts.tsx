import { useMemo } from "react";
import { EChart, type EOption } from "./EChart";
import { axisStyle, baseGrid, compactMoney, useChartTokens } from "../lib/chartTheme";
import type { GPStat, Series } from "../lib/types";

const line = (color: string, width = 2) => ({
  type: "line",
  showSymbol: false,
  symbolSize: 8,
  lineStyle: { width, color, cap: "round", join: "round" },
  itemStyle: { color, borderColor: "transparent" },
  emphasis: { focus: "series" },
});

/** Cumulative PnL (total + long/short legs) with the OS holdout shaded. */
export function PnlChart({ series, osStart, height = 260 }: { series: Series; osStart?: number; height?: number }) {
  const t = useChartTokens();
  const option = useMemo<EOption>(() => {
    const d = series.dates;
    const os = osStart !== undefined && osStart < d.length ? d[osStart] : undefined;
    return {
      ...baseGrid(t, { grid: { left: 62, right: 56, top: 28, bottom: 26 } }),
      legend: {
        top: 0,
        right: 4,
        itemWidth: 14,
        itemHeight: 3,
        icon: "rect",
        textStyle: { color: t.ink2, fontSize: 11 },
        data: ["Total", "Long leg", "Short leg"],
      },
      tooltip: {
        ...(baseGrid(t) as any).tooltip,
        valueFormatter: (v: number) => (typeof v === "number" ? `$${compactMoney(v)}` : "—"),
      },
      xAxis: { type: "category", data: d, boundaryGap: false, ...axisStyle(t), splitLine: { show: false } },
      yAxis: { type: "value", scale: true, ...axisStyle(t), axisLabel: { color: t.muted, fontSize: 10.5, formatter: (v: number) => compactMoney(v) } },
      dataZoom: [{ type: "inside", throttle: 30 }],
      series: [
        {
          name: "Total",
          ...line(t.series[0]),
          data: series.cum_pnl,
          z: 3,
          markArea: os
            ? {
                silent: true,
                itemStyle: { color: t.surface2, opacity: 0.9 },
                label: { show: true, position: "insideTop", color: t.muted, fontSize: 10, formatter: "OS holdout" },
                data: [[{ xAxis: os }, { xAxis: d[d.length - 1] }]],
              }
            : undefined,
          endLabel: { show: true, color: t.ink2, fontSize: 10.5, formatter: (p: any) => compactMoney(p.value) },
        },
        { name: "Long leg", ...line(t.series[1], 1.5), data: series.cum_long, z: 2 },
        { name: "Short leg", ...line(t.series[2], 1.5), data: series.cum_short, z: 2 },
      ],
    };
  }, [series, osStart, t]);
  return <EChart option={option} height={height} />;
}

/** One measure over time; optional horizontal reference line (threshold). */
export function SeriesChart({
  dates,
  data,
  name,
  height = 170,
  area = false,
  refLine,
  percent = false,
  money = false,
  smooth,
}: {
  dates: string[];
  data: (number | null)[];
  name: string;
  height?: number;
  area?: boolean;
  refLine?: { value: number; label: string };
  percent?: boolean;
  money?: boolean;
  smooth?: number;
}) {
  const t = useChartTokens();
  const option = useMemo<EOption>(() => {
    let vals = data;
    if (smooth && smooth > 1) {
      const out: (number | null)[] = [];
      let s = 0;
      let n = 0;
      const q: (number | null)[] = [];
      for (const v of data) {
        q.push(v);
        if (v !== null) {
          s += v;
          n++;
        }
        if (q.length > smooth) {
          const old = q.shift();
          if (old !== null && old !== undefined) {
            s -= old;
            n--;
          }
        }
        out.push(n ? s / n : null);
      }
      vals = out;
    }
    const fmtV = (v: number) => (percent ? `${(v * 100).toFixed(0)}%` : money ? compactMoney(v) : v.toFixed(2));
    return {
      ...baseGrid(t, { grid: { left: 52, right: 18, top: 14, bottom: 24 } }),
      tooltip: { ...(baseGrid(t) as any).tooltip, valueFormatter: (v: number) => (typeof v === "number" ? fmtV(v) : "—") },
      xAxis: { type: "category", data: dates, boundaryGap: false, ...axisStyle(t), splitLine: { show: false } },
      yAxis: { type: "value", scale: !area, ...axisStyle(t), axisLabel: { color: t.muted, fontSize: 10.5, formatter: fmtV } },
      dataZoom: [{ type: "inside", throttle: 30 }],
      series: [
        {
          name,
          ...line(t.series[0]),
          data: vals,
          areaStyle: area ? { color: t.series[0], opacity: 0.1 } : undefined,
          markLine: refLine
            ? {
                silent: true,
                symbol: "none",
                lineStyle: { color: t.muted, width: 1, type: "solid" },
                label: { color: t.muted, fontSize: 10, formatter: refLine.label, position: "insideEndTop" },
                data: [{ yAxis: refLine.value }],
              }
            : undefined,
        },
      ],
    };
  }, [dates, data, name, area, refLine, percent, money, smooth, t]);
  return <EChart option={option} height={height} />;
}

/** Two measures of the same unit (e.g. long/short position counts). */
export function PairChart({ dates, a, b, names, height = 170 }: { dates: string[]; a: number[]; b: number[]; names: [string, string]; height?: number }) {
  const t = useChartTokens();
  const option = useMemo<EOption>(
    () => ({
      ...baseGrid(t, { grid: { left: 52, right: 18, top: 28, bottom: 24 } }),
      legend: { top: 0, right: 4, itemWidth: 14, itemHeight: 3, icon: "rect", textStyle: { color: t.ink2, fontSize: 11 } },
      xAxis: { type: "category", data: dates, boundaryGap: false, ...axisStyle(t), splitLine: { show: false } },
      yAxis: { type: "value", scale: true, ...axisStyle(t) },
      dataZoom: [{ type: "inside" }],
      series: [
        { name: names[0], ...line(t.series[0]), data: a },
        { name: names[1], ...line(t.series[1]), data: b },
      ],
    }),
    [dates, a, b, names, t],
  );
  return <EChart option={option} height={height} />;
}

/** Diverging heatmap (blue <-> gray <-> red). */
export function Heatmap({
  xLabels,
  yLabels,
  cells,
  height = 240,
  onCell,
  highlight,
  valueLabel = "value",
  invert = false,
  showValues = true,
  digits = 2,
}: {
  xLabels: string[];
  yLabels: string[];
  cells: [number, number, number | null][];
  height?: number;
  onCell?: (x: number, y: number) => void;
  highlight?: [number, number] | null;
  valueLabel?: string;
  invert?: boolean;
  showValues?: boolean;
  digits?: number;
}) {
  const t = useChartTokens();
  const option = useMemo<EOption>(() => {
    const vals = cells.map((c) => c[2]).filter((v): v is number => v !== null && Number.isFinite(v));
    const m = Math.max(0.01, ...vals.map((v) => Math.abs(v)));
    const colors = invert ? [t.divPos, t.divMid, t.divNeg] : [t.divNeg, t.divMid, t.divPos];
    return {
      backgroundColor: "transparent",
      textStyle: { color: t.ink2, fontSize: 11 },
      tooltip: {
        backgroundColor: t.surface,
        borderColor: t.grid,
        textStyle: { color: t.ink, fontSize: 11 },
        formatter: (p: any) => `${yLabels[p.value[1]]} · ${xLabels[p.value[0]]}<br/>${valueLabel}: <b>${p.value[2] === null ? "—" : Number(p.value[2]).toFixed(digits)}</b>`,
      },
      grid: { left: 96, right: 14, top: 8, bottom: 52 },
      xAxis: { type: "category", data: xLabels, splitArea: { show: false }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 10.5, rotate: xLabels.length > 12 ? 60 : 0 } },
      yAxis: { type: "category", data: yLabels, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, fontSize: 10.5 } },
      visualMap: {
        min: -m,
        max: m,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        itemHeight: 120,
        itemWidth: 10,
        textStyle: { color: t.muted, fontSize: 10 },
        inRange: { color: colors },
        text: [`+${m.toFixed(digits)}`, `-${m.toFixed(digits)}`],
      },
      series: [
        {
          type: "heatmap",
          // the robust pick is outlined in ink instead of covered by a marker, so its value stays readable
          data: cells.map((c) =>
            highlight && c[0] === highlight[0] && c[1] === highlight[1]
              ? { value: c, itemStyle: { borderColor: t.ink, borderWidth: 3 } }
              : c,
          ),
          label: {
            show: showValues,
            fontSize: 10,
            color: t.ink,
            formatter: (p: any) => (p.value[2] === null ? "" : Number(p.value[2]).toFixed(digits)),
          },
          itemStyle: { borderColor: t.surface, borderWidth: 2 },
          emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 2 } },
        },
      ],
    };
  }, [xLabels, yLabels, cells, highlight, valueLabel, invert, showValues, digits, t]);
  return <EChart option={option} height={height} onClick={(p) => onCell && p.seriesType === "heatmap" && onCell(p.value[0], p.value[1])} />;
}

/** GP progress: best and median fitness per generation. */
export function GpProgressChart({ stats, height = 180 }: { stats: GPStat[]; height?: number }) {
  const t = useChartTokens();
  const option = useMemo<EOption>(
    () => ({
      ...baseGrid(t, { grid: { left: 44, right: 16, top: 28, bottom: 36 } }),
      legend: { top: 0, right: 4, itemWidth: 14, itemHeight: 3, icon: "rect", textStyle: { color: t.ink2, fontSize: 11 } },
      xAxis: { type: "category", data: stats.map((s) => s.generation), ...axisStyle(t), splitLine: { show: false }, name: "generation", nameLocation: "middle", nameGap: 22, nameTextStyle: { color: t.muted, fontSize: 10 } },
      yAxis: { type: "value", scale: true, ...axisStyle(t) },
      series: [
        { name: "Best fitness", ...line(t.series[0]), showSymbol: true, data: stats.map((s) => s.best_fitness) },
        { name: "Median fitness", ...line(t.series[1]), showSymbol: true, data: stats.map((s) => s.median_fitness) },
      ],
    }),
    [stats, t],
  );
  return <EChart option={option} height={height} />;
}

/** Pareto front: Sharpe vs turnover, color = novelty (single-hue sequential ramp). */
export function ParetoScatter({ front, height = 220 }: { front: GPStat["front"]; height?: number }) {
  const t = useChartTokens();
  const option = useMemo<EOption>(
    () => ({
      ...baseGrid(t, { grid: { left: 48, right: 84, top: 14, bottom: 36 } }),
      tooltip: {
        trigger: "item",
        backgroundColor: t.surface,
        borderColor: t.grid,
        textStyle: { color: t.ink, fontSize: 11 },
        formatter: (p: any) =>
          `<div style="max-width:320px;white-space:normal"><code>${String(p.data[4]).replace(/</g, "&lt;")}</code><br/>Sharpe ${Number(p.data[1]).toFixed(2)} · TO ${(p.data[0] * 100).toFixed(0)}% · fitness ${Number(p.data[3]).toFixed(2)} · novelty ${Number(p.data[2]).toFixed(2)}</div>`,
      },
      xAxis: { type: "value", name: "turnover", nameLocation: "middle", nameGap: 22, nameTextStyle: { color: t.muted, fontSize: 10 }, ...axisStyle(t), axisLabel: { color: t.muted, fontSize: 10.5, formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
      yAxis: { type: "value", name: "Sharpe", nameTextStyle: { color: t.muted, fontSize: 10 }, ...axisStyle(t) },
      visualMap: {
        dimension: 2,
        min: 0,
        max: 1,
        right: 0,
        top: 10,
        itemHeight: 110,
        itemWidth: 10,
        text: ["novel", "similar"],
        textStyle: { color: t.muted, fontSize: 10 },
        inRange: { color: t.seq },
        calculable: false,
      },
      series: [
        {
          type: "scatter",
          symbolSize: 10,
          itemStyle: { borderColor: t.surface, borderWidth: 2 },
          data: front.map((f) => [f.turnover, f.sharpe, f.novelty, f.fitness, f.expr]),
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: t.muted, width: 1, type: "solid" },
            label: { color: t.muted, fontSize: 10 },
            data: [
              { yAxis: 1.25, label: { formatter: "Sharpe 1.25", position: "insideStartTop" } },
              { xAxis: 0.7, label: { formatter: "TO 70%", position: "insideEndTop" } },
            ],
          },
        },
      ],
    }),
    [front, t],
  );
  return <EChart option={option} height={height} />;
}

/** Local vs BRAIN Sharpe with pass/fail status encoding (color + shape + legend label). */
export function CalibrationScatter({ points, map, height = 280 }: { points: { local: number; brain: number; passed: boolean | null }[]; map?: { slope: number; intercept: number } | null; height?: number }) {
  const t = useChartTokens();
  const option = useMemo<EOption>(() => {
    const xs = points.map((p) => p.local);
    const lo = Math.min(0, ...xs);
    const hi = Math.max(2, ...xs);
    return {
      ...baseGrid(t, { grid: { left: 48, right: 16, top: 30, bottom: 36 } }),
      legend: { top: 0, right: 4, textStyle: { color: t.ink2, fontSize: 11 } },
      tooltip: { trigger: "item", backgroundColor: t.surface, borderColor: t.grid, textStyle: { color: t.ink, fontSize: 11 }, formatter: (p: any) => `local ${Number(p.value[0]).toFixed(2)} → BRAIN ${Number(p.value[1]).toFixed(2)}` },
      xAxis: { type: "value", name: "local Sharpe", nameLocation: "middle", nameGap: 22, nameTextStyle: { color: t.muted, fontSize: 10 }, ...axisStyle(t) },
      yAxis: { type: "value", name: "BRAIN Sharpe", nameTextStyle: { color: t.muted, fontSize: 10 }, ...axisStyle(t) },
      series: [
        { name: "Passed on BRAIN", type: "scatter", symbol: "circle", symbolSize: 10, itemStyle: { color: t.good, borderColor: t.surface, borderWidth: 2 }, data: points.filter((p) => p.passed).map((p) => [p.local, p.brain]) },
        { name: "Failed on BRAIN", type: "scatter", symbol: "triangle", symbolSize: 11, itemStyle: { color: t.bad, borderColor: t.surface, borderWidth: 2 }, data: points.filter((p) => p.passed === false).map((p) => [p.local, p.brain]) },
        { name: "Unlabeled", type: "scatter", symbol: "diamond", symbolSize: 10, itemStyle: { color: t.muted, borderColor: t.surface, borderWidth: 2 }, data: points.filter((p) => p.passed === null || p.passed === undefined).map((p) => [p.local, p.brain]) },
        ...(map
          ? [{ name: "Fitted map", ...line(t.series[0]), data: [[lo, map.slope * lo + map.intercept], [hi, map.slope * hi + map.intercept]] }]
          : []),
      ],
    };
  }, [points, map, t]);
  return <EChart option={option} height={height} />;
}

/** Combined PnL of several alphas. */
export function ComboChart({ dates, cum, height = 240 }: { dates: string[]; cum: number[]; height?: number }) {
  return <SeriesChart dates={dates} data={cum} name="Combined PnL" money height={height} />;
}

/** Original vs re-engineered cumulative PnL on one money axis, OS holdout shaded, both lines direct-labeled. */
export function PnlCompareChart({ dates, original, champion, osStart, height = 250 }: { dates: string[]; original: number[]; champion: number[]; osStart?: string | null; height?: number }) {
  const t = useChartTokens();
  const option = useMemo<EOption>(() => {
    const endLabel = (name: string) => ({ show: true, color: t.ink2, fontSize: 10.5, formatter: (p: any) => `${name} ${compactMoney(p.value)}` });
    return {
      ...baseGrid(t, { grid: { left: 62, right: 118, top: 28, bottom: 26 } }),
      legend: { top: 0, right: 4, itemWidth: 14, itemHeight: 3, icon: "rect", textStyle: { color: t.ink2, fontSize: 11 }, data: ["Re-engineered", "Original"] },
      tooltip: { ...(baseGrid(t) as any).tooltip, valueFormatter: (v: number) => (typeof v === "number" ? `$${compactMoney(v)}` : "—") },
      xAxis: { type: "category", data: dates, boundaryGap: false, ...axisStyle(t), splitLine: { show: false } },
      yAxis: { type: "value", scale: true, ...axisStyle(t), axisLabel: { color: t.muted, fontSize: 10.5, formatter: (v: number) => compactMoney(v) } },
      dataZoom: [{ type: "inside", throttle: 30 }],
      series: [
        {
          name: "Re-engineered",
          ...line(t.series[0]),
          data: champion,
          z: 3,
          endLabel: endLabel("Re-engineered"),
          markArea: osStart
            ? {
                silent: true,
                itemStyle: { color: t.surface2, opacity: 0.9 },
                label: { show: true, position: "insideTop", color: t.muted, fontSize: 10, formatter: "OS holdout (never searched)" },
                data: [[{ xAxis: osStart }, { xAxis: dates[dates.length - 1] }]],
              }
            : undefined,
        },
        { name: "Original", ...line(t.series[1]), data: original, z: 2, endLabel: endLabel("Original") },
      ],
    };
  }, [dates, original, champion, osStart, t]);
  return <EChart option={option} height={height} />;
}

/** IS Sharpe and fitness of the best version after each search stage (one unitless axis, bars as reference lines). */
export function ReTrajectoryChart({ steps, sharpeBar = 1.25, fitnessBar = 1.0, height = 200 }: { steps: { label: string; sharpe: number | null; fitness: number | null }[]; sharpeBar?: number; fitnessBar?: number; height?: number }) {
  const t = useChartTokens();
  const option = useMemo<EOption>(() => {
    const endLabel = (name: string) => ({ show: true, color: t.ink2, fontSize: 10.5, formatter: (p: any) => `${name} ${Number(p.value).toFixed(2)}` });
    // the two bars sit close together, so their labels go at opposite ends and opposite sides of the line
    const ref = (value: number, label: string, position: string) => ({ yAxis: value, label: { formatter: label, position } });
    return {
      ...baseGrid(t, { grid: { left: 40, right: 96, top: 28, bottom: 40 } }),
      legend: { top: 0, right: 4, itemWidth: 14, itemHeight: 3, icon: "rect", textStyle: { color: t.ink2, fontSize: 11 }, data: ["Sharpe", "Fitness"] },
      tooltip: { ...(baseGrid(t) as any).tooltip, valueFormatter: (v: number) => (typeof v === "number" ? v.toFixed(2) : "—") },
      xAxis: { type: "category", data: steps.map((s) => s.label), ...axisStyle(t), splitLine: { show: false }, axisLabel: { color: t.muted, fontSize: 10, rotate: steps.length > 8 ? 30 : 0, hideOverlap: true } },
      yAxis: { type: "value", scale: true, ...axisStyle(t) },
      series: [
        {
          name: "Sharpe",
          ...line(t.series[0]),
          showSymbol: true,
          data: steps.map((s) => s.sharpe),
          endLabel: endLabel("Sharpe"),
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: t.muted, width: 1, type: "solid" },
            label: { color: t.muted, fontSize: 10 },
            data: [ref(sharpeBar, `Sharpe bar ${sharpeBar}`, "insideStartTop"), ref(fitnessBar, `fitness bar ${fitnessBar}`, "insideEndBottom")],
          },
        },
        { name: "Fitness", ...line(t.series[1]), showSymbol: true, data: steps.map((s) => s.fitness), endLabel: endLabel("Fitness") },
      ],
    };
  }, [steps, sharpeBar, fitnessBar, t]);
  return <EChart option={option} height={height} />;
}
