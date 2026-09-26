import { useMemo } from "react";
import { useUI } from "./store";

export interface ChartTokens {
  surface: string;
  surface2: string;
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  series: string[];
  divNeg: string;
  divMid: string;
  divPos: string;
  good: string;
  bad: string;
  warn: string;
  seq: string[];
}

export function readTokens(): ChartTokens {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  return {
    surface: v("--surface-1"),
    surface2: v("--surface-2"),
    ink: v("--text-primary"),
    ink2: v("--text-secondary"),
    muted: v("--text-muted"),
    grid: v("--hairline"),
    axis: v("--axis"),
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v(`--series-${i}`)),
    divNeg: v("--div-neg"),
    divMid: v("--div-mid"),
    divPos: v("--div-pos"),
    good: v("--status-good"),
    bad: v("--status-critical"),
    warn: v("--status-warning"),
    seq: [1, 2, 3, 4, 5].map((i) => v(`--seq-${i}`)),
  };
}

/** Tokens re-read whenever the theme changes. */
export function useChartTokens(): ChartTokens {
  const theme = useUI((s) => s.theme);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => readTokens(), [theme]);
}

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

export function baseGrid(t: ChartTokens, extra: Record<string, unknown> = {}) {
  return {
    backgroundColor: "transparent",
    textStyle: { color: t.ink2, fontFamily: FONT, fontSize: 11 },
    animationDuration: 250,
    tooltip: {
      trigger: "axis",
      backgroundColor: t.surface,
      borderColor: t.grid,
      borderWidth: 1,
      textStyle: { color: t.ink, fontSize: 11 },
      axisPointer: { type: "line", lineStyle: { color: t.axis, width: 1 } },
      confine: true,
    },
    grid: { left: 62, right: 18, top: 28, bottom: 26, containLabel: false },
    ...extra,
  };
}

export function axisStyle(t: ChartTokens) {
  return {
    axisLine: { lineStyle: { color: t.axis, width: 1 } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: 10.5, hideOverlap: true },
    splitLine: { show: true, lineStyle: { color: t.grid, width: 1, type: "solid" } },
  };
}

export function compactMoney(v: number): string {
  const a = Math.abs(v);
  const s = v < 0 ? "-" : "";
  if (a >= 1e9) return `${s}${(a / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${s}${(a / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${s}${(a / 1e3).toFixed(0)}K`;
  return `${s}${a.toFixed(0)}`;
}
