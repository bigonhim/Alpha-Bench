export const fmt = {
  num(v: number | null | undefined, d = 2): string {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return v.toFixed(d);
  },
  pct(v: number | null | undefined, d = 1): string {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return `${(v * 100).toFixed(d)}%`;
  },
  bps(v: number | null | undefined): string {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return `${v.toFixed(1)}‱`;
  },
  money(v: number | null | undefined): string {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const a = Math.abs(v);
    const s = v < 0 ? "-" : "";
    if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(2)}B`;
    if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(1)}K`;
    return `${s}$${a.toFixed(0)}`;
  },
  compact(v: number | null | undefined): string {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (a >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return v.toLocaleString();
  },
  ago(iso?: string): string {
    if (!iso) return "";
    const t = new Date(iso).getTime();
    const s = (Date.now() - t) / 1000;
    if (s < 60) return `${Math.max(0, Math.round(s))}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  },
};

export function settingsLabel(s: { region?: string; universe?: string; delay?: number; decay?: number; neutralization?: string; truncation?: number }): string {
  return `${s.region ?? "USA"} · ${s.universe ?? "TOP3000"} · D${s.delay ?? 1} · decay ${s.decay ?? 0} · ${(s.neutralization ?? "").toLowerCase()} · trunc ${s.truncation ?? 0.08}`;
}

export function clsx(...a: (string | false | null | undefined)[]): string {
  return a.filter(Boolean).join(" ");
}
