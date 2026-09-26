import { Copy } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { copyText } from "../lib/api";
import { clsx, fmt } from "../lib/format";
import type { Check, SimResult } from "../lib/types";
import { PairChart, PnlChart, SeriesChart } from "./charts";
import { Card, Kv, Spinner, StatusBadge, Tabs, Tip } from "./ui";

const HARD = new Set(["LOW_SHARPE", "LOW_FITNESS", "LOW_TURNOVER", "HIGH_TURNOVER", "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE", "SELF_CORRELATION"]);

function checkOf(r: SimResult, name: string): Check | undefined {
  return r.checks?.checks.find((c) => c.name === name);
}

function Metric({ label, value, os, status, hint }: { label: string; value: string; os?: string; status?: string; hint?: string }) {
  return (
    <div className="card flex min-w-[112px] flex-1 flex-col gap-0.5 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10.5px] font-semibold uppercase tracking-wide text-muted" title={hint}>
          {label}
        </span>
        {status && <StatusBadge status={status} compact label="" />}
      </div>
      <div className="text-[20px] font-semibold leading-tight text-ink">{value}</div>
      {os !== undefined && <div className="tnum text-[11px] text-muted">OS {os}</div>}
    </div>
  );
}

export function MetricStrip({ r }: { r: SimResult }) {
  const is = r.metrics?.is;
  const os = r.metrics?.os;
  if (!is) return null;
  const st = (n: string) => checkOf(r, n)?.result;
  const toStatus = st("HIGH_TURNOVER") === "FAIL" || st("LOW_TURNOVER") === "FAIL" ? "FAIL" : st("HIGH_TURNOVER") ? "PASS" : undefined;
  return (
    <div className="flex flex-wrap gap-2">
      <Metric label="Sharpe" value={fmt.num(is.sharpe)} os={fmt.num(os?.sharpe)} status={st("LOW_SHARPE")} hint="Annualized mean/std of daily PnL" />
      <Metric label="Fitness" value={fmt.num(is.fitness)} os={fmt.num(os?.fitness)} status={st("LOW_FITNESS")} hint="Sharpe × sqrt(|returns| / max(turnover, 12.5%))" />
      <Metric label="Turnover" value={fmt.pct(is.turnover)} os={fmt.pct(os?.turnover)} status={toStatus} />
      <Metric label="Returns" value={fmt.pct(is.returns, 2)} os={fmt.pct(os?.returns, 2)} />
      <Metric label="Drawdown" value={fmt.pct(is.drawdown, 2)} os={fmt.pct(os?.drawdown, 2)} />
      <Metric label="Margin" value={fmt.bps(is.margin_bps)} os={fmt.bps(os?.margin_bps)} hint="PnL per dollar traded (basis points)" />
      <div className="card flex min-w-[150px] flex-1 flex-col gap-0.5 px-3 py-2">
        <span className="text-[10.5px] font-semibold uppercase tracking-wide text-muted">BRAIN pass likelihood</span>
        <div className="text-[20px] font-semibold leading-tight">{r.pass_prob !== undefined ? `${Math.round(r.pass_prob * 100)}%` : "—"}</div>
        <div className="text-[11px] text-muted">
          {r.expected_brain_sharpe != null ? `expected BRAIN Sharpe ${fmt.num(r.expected_brain_sharpe)}` : "calibrates as you import BRAIN results"}
        </div>
      </div>
    </div>
  );
}

export function ChecksList({ r, loading }: { r: SimResult; loading?: boolean }) {
  const checks = r.checks?.checks ?? [];
  const hard = checks.filter((c) => HARD.has(c.name) || c.name === "LOW_2Y_SHARPE");
  const gates = checks.filter((c) => !HARD.has(c.name) && c.name !== "LOW_2Y_SHARPE");
  const row = (c: Check) => (
    <div key={c.name} className="flex items-start justify-between gap-2 border-b border-line py-1.5 last:border-0">
      <div className="min-w-0">
        <div className="mono text-[11.5px] font-medium text-ink">{c.name}</div>
        <div className="text-[11.5px] text-muted">{c.message}</div>
      </div>
      <StatusBadge status={c.result} />
    </div>
  );
  return (
    <Card
      title="Submission checks"
      actions={
        <>
          {loading && <Spinner />}
          {r.checks && <StatusBadge status={r.checks.status} label={r.checks.status === "PASS" ? "All hard checks pass" : r.checks.status === "FAIL" ? `${r.checks.failed.length} failing` : "Pending"} />}
        </>
      }
    >
      <div>{hard.map(row)}</div>
      {gates.length > 0 && (
        <>
          <div className="mt-2 text-[10.5px] font-semibold uppercase tracking-wide text-muted">Robustness gates (local)</div>
          <div>{gates.map(row)}</div>
        </>
      )}
    </Card>
  );
}

export function YearlyTable({ r }: { r: SimResult }) {
  const rows = r.yearly ?? [];
  return (
    <div className="max-h-[300px] overflow-auto">
      <table className="tbl tnum">
        <thead>
          <tr>
            <th>Year</th>
            <th>Period</th>
            <th className="text-right">Sharpe</th>
            <th className="text-right">Fitness</th>
            <th className="text-right">Turnover</th>
            <th className="text-right">Returns</th>
            <th className="text-right">Drawdown</th>
            <th className="text-right">Margin</th>
            <th className="text-right">Long</th>
            <th className="text-right">Short</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((y) => (
            <tr key={y.year}>
              <td>{y.year}</td>
              <td>
                <span className="chip">{y.period}</span>
              </td>
              <td className={clsx("text-right", y.sharpe < 0 && "text-[var(--bad-text)]")}>{fmt.num(y.sharpe)}</td>
              <td className="text-right">{fmt.num(y.fitness)}</td>
              <td className="text-right">{fmt.pct(y.turnover)}</td>
              <td className="text-right">{fmt.pct(y.returns, 2)}</td>
              <td className="text-right">{fmt.pct(y.drawdown, 2)}</td>
              <td className="text-right">{fmt.bps(y.margin_bps)}</td>
              <td className="text-right">{Math.round(y.long_count)}</td>
              <td className="text-right">{Math.round(y.short_count)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ChartsCard({ r }: { r: SimResult }) {
  const [tab, setTab] = useState("pnl");
  const s = r.series;
  const osStart = r.periods?.os?.[0];
  const tabs = useMemo(() => {
    if (!s) return [];
    return [
      { id: "pnl", label: "PnL", content: <PnlChart series={s} osStart={osStart} /> },
      { id: "dd", label: "Drawdown", content: <SeriesChart dates={s.dates} data={s.drawdown} name="Drawdown" area money height={260} /> },
      {
        id: "rs",
        label: "Rolling Sharpe (1y)",
        content: <SeriesChart dates={s.dates} data={s.rolling_sharpe} name="Rolling 1y Sharpe" refLine={{ value: 1.25, label: "1.25" }} height={260} />,
      },
      {
        id: "to",
        label: "Turnover (20d avg)",
        content: <SeriesChart dates={s.dates} data={s.turnover} name="Turnover" percent smooth={20} refLine={{ value: 0.7, label: "70% limit" }} height={260} />,
      },
      { id: "pos", label: "Positions", content: <PairChart dates={s.dates} a={s.long_count} b={s.short_count} names={["Long names", "Short names"]} height={260} /> },
      { id: "years", label: "Yearly table", content: <YearlyTable r={r} /> },
    ];
  }, [s, osStart, r]);
  if (!s) return null;
  return (
    <Card title={`Performance · ${r.local_universe ?? ""} (~${Math.round(r.universe_size ?? 0)} names)`} bodyClass="px-1 pb-2">
      <Tabs tabs={tabs} value={tab} onChange={setTab} />
    </Card>
  );
}

export function RobustnessCard({ r, loading }: { r: SimResult; loading?: boolean }) {
  const ex = r.extras;
  const sub = checkOf(r, "LOW_SUB_UNIVERSE_SHARPE");
  return (
    <Card title="Robustness" actions={loading ? <Spinner /> : null}>
      {!ex ? (
        <div className="text-[12px] text-muted">{loading ? "Running sub-universe and parameter-stability tests…" : "Not computed yet."}</div>
      ) : (
        <div className="flex flex-col gap-0.5">
          <Kv k={`Sub-universe (${ex.sub_universe ?? "—"}) Sharpe`} v={`${fmt.num(ex.sub_sharpe)} ${sub?.limit != null ? `(needs ≥ ${fmt.num(sub.limit)})` : ""}`} />
          <Kv k="Parameter stability" v={ex.stability != null ? fmt.pct(ex.stability, 0) : "—"} />
          <Kv k="Deflated Sharpe probability" v={r.dsr ? fmt.pct(r.dsr.dsr, 0) : "—"} />
          {(ex.variants ?? []).length > 0 && (
            <div className="mt-1.5">
              <div className="lbl mb-1">Window perturbations</div>
              {(ex.variants ?? []).map((v) => (
                <div key={v.expr} className="flex justify-between gap-2 py-0.5 text-[11.5px]">
                  <span className="mono truncate text-ink2" title={v.expr}>
                    {v.expr}
                  </span>
                  <span className="tnum shrink-0">S {fmt.num(v.sharpe)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

export function CorrelationCard({ r, onOpen }: { r: SimResult; onOpen?: (id: number) => void }) {
  const top = r.correlation?.top ?? [];
  return (
    <Card title="Most correlated library alphas">
      {top.length === 0 ? (
        <div className="text-[12px] text-muted">Library is empty or has no overlapping PnL yet.</div>
      ) : (
        top.map((c) => (
          <button key={c.alpha_id} className="flex w-full items-center gap-2 py-0.5 text-left text-[12px] hover:text-ink" onClick={() => onOpen?.(c.alpha_id)}>
            <span className="w-12 text-ink2">#{c.alpha_id}</span>
            <div className="relative h-1.5 flex-1 rounded bg-[var(--surface-3)]">
              <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${Math.abs(c.corr) * 100}%`, background: Math.abs(c.corr) >= 0.7 ? "var(--div-neg)" : "var(--series-1)" }} />
            </div>
            <span className="tnum w-12 text-right">{c.corr.toFixed(2)}</span>
          </button>
        ))
      )}
    </Card>
  );
}

export function AttributionCard({ r }: { r: SimResult }) {
  const sec = Object.entries(r.sector_pnl ?? {}).sort((a, b) => b[1] - a[1]);
  const m = Math.max(1, ...sec.map(([, v]) => Math.abs(v)));
  return (
    <Card title="PnL attribution (IS)">
      <div className="flex flex-col gap-0.5">
        {sec.map(([k, v]) => (
          <div key={k} className="flex items-center gap-2 text-[11.5px]">
            <span className="w-36 truncate text-ink2" title={k}>
              {k}
            </span>
            <div className="relative h-2.5 flex-1">
              <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--axis)]" />
              <div
                className="absolute inset-y-0 rounded-sm"
                style={{
                  left: v >= 0 ? "50%" : `${50 - (Math.abs(v) / m) * 50}%`,
                  width: `${(Math.abs(v) / m) * 50}%`,
                  background: v >= 0 ? "var(--div-pos)" : "var(--div-neg)",
                }}
              />
            </div>
            <span className="tnum w-16 text-right">{fmt.money(v)}</span>
          </div>
        ))}
      </div>
      {r.top_names && (
        <div className="mt-2 grid grid-cols-2 gap-3 text-[11.5px]">
          <div>
            <div className="lbl mb-0.5">Top contributors</div>
            {r.top_names.best.slice(0, 5).map(([t, v]) => (
              <Kv key={t} k={t} v={fmt.money(v)} />
            ))}
          </div>
          <div>
            <div className="lbl mb-0.5">Top detractors</div>
            {r.top_names.worst.slice(0, 5).map(([t, v]) => (
              <Kv key={t} k={t} v={fmt.money(v)} />
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

export function DescriptionCard({ r }: { r: SimResult }) {
  const d = r.description;
  if (!d) return null;
  const text = `Idea: ${d.idea}\n\nData: ${d.data}\n\nOperators:\n${d.operators.map((o) => `- ${o}`).join("\n")}`;
  return (
    <Card
      title="Description draft (for BRAIN)"
      actions={
        <button
          className="btn btn-ghost !px-1.5 !py-0.5"
          onClick={() => {
            copyText(text);
            toast.success("Description copied");
          }}
        >
          <Copy size={12} /> Copy
        </button>
      }
    >
      <div className="flex flex-col gap-2 text-[12px] leading-relaxed">
        <div className="text-ink">{d.summary}</div>
        <div>
          <span className="lbl">Idea</span>
          <div className="text-ink2">{d.idea}</div>
        </div>
        <div>
          <span className="lbl">Data</span>
          <div className="text-ink2">{d.data}</div>
        </div>
        <div>
          <span className="lbl">Operators</span>
          <ul className="list-disc pl-4 text-ink2">
            {d.operators.map((o) => (
              <li key={o}>{o}</li>
            ))}
          </ul>
        </div>
        <div className="flex flex-wrap gap-1">
          <span className="chip">idea: {d.tags.idea}</span>
          <span className="chip">data: {d.tags.category}</span>
          <span className="chip">horizon: {d.tags.horizon}</span>
        </div>
      </div>
    </Card>
  );
}

export function BrainOnlyNotice({ r }: { r: SimResult }) {
  return (
    <Card title="BRAIN-only alpha">
      <div className="text-[12.5px] text-ink2">{r.message}</div>
      <ul className="mt-1.5 list-disc pl-4 text-[12px] text-muted">
        {r.analysis.brain_only_reasons.map((x) => (
          <li key={x}>{x}</li>
        ))}
      </ul>
      <div className="mt-2 text-[12px] text-muted">
        <Tip content="Save it to the library, then export the BRAIN simulation payload from the Library page.">
          <span className="cursor-help underline decoration-dotted">How do I test it?</span>
        </Tip>
      </div>
    </Card>
  );
}
