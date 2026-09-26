import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Clipboard, FlaskConical, Library, Square, Wand2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api, copyText } from "../lib/api";
import { clsx, fmt, settingsLabel } from "../lib/format";
import { useJobs, useUI } from "../lib/store";
import type { ReFinalRow, ReMetrics, ReReport, ReStep, Settings } from "../lib/types";
import { PnlCompareChart, ReTrajectoryChart } from "./charts";
import { Card, Empty, Field, Progress, Spinner, StatusBadge, Toggle } from "./ui";

export interface ReConfig {
  time_limit_min: number;
  target_sharpe: number;
  target_fitness: number;
  allow_blend: boolean;
  allow_structure: boolean;
  evolve: boolean;
}

export const RE_DEFAULTS: ReConfig = { time_limit_min: 6, target_sharpe: 2.0, target_fitness: 1.5, allow_blend: true, allow_structure: true, evolve: true };

export async function startReengineer(expr: string, settings: Partial<Settings>, cfg: Partial<ReConfig> = {}, alphaId?: number): Promise<number> {
  const r = await api.startJob("reengineer", { ...RE_DEFAULTS, ...cfg, expr, settings, ...(alphaId ? { alpha_id: alphaId } : {}) });
  return r.id;
}

const LIVE = new Set(["running", "paused", "queued"]);
const VERDICT_STATUS: Record<string, string> = { good: "PASS", warn: "WARNING", bad: "FAIL" };
const DIAG_STATUS: Record<string, string> = { bad: "FAIL", warn: "WARNING", info: "PENDING" };

function Num({ label, value, onChange, step = 1, min, hint }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <input className="input w-24 tnum" type="number" step={step} min={min} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

/** Search options for a re-engineer run (shared by the Studio panel and the Miner card). */
export function ReengineerOptions({ value, onChange }: { value: ReConfig; onChange: (v: ReConfig) => void }) {
  const set = <K extends keyof ReConfig>(k: K) => (v: ReConfig[K]) => onChange({ ...value, [k]: v });
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-3">
        <Num label="Time budget (min)" value={value.time_limit_min} onChange={set("time_limit_min")} min={0.5} step={0.5} />
        <Num label="Target Sharpe" value={value.target_sharpe} onChange={set("target_sharpe")} step={0.1} hint="The search stops early once the best version reaches both targets with no failing checks" />
        <Num label="Target fitness" value={value.target_fitness} onChange={set("target_fitness")} step={0.1} />
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-2">
        <Toggle checked={value.allow_structure} onChange={set("allow_structure")} label="Rewrite structure (component surgery, conditioning)" />
        <Toggle checked={value.allow_blend} onChange={set("allow_blend")} label="Allow blending one decorrelated companion signal" />
        <Toggle checked={value.evolve} onChange={set("evolve")} label="Finish with genetic programming if time remains" />
      </div>
    </div>
  );
}

/** Config form shown before a run starts. */
export function ReengineerLauncher({ expr, onStart, busy }: { expr: string; onStart: (cfg: ReConfig) => void; busy?: boolean }) {
  const [cfg, setCfg] = useState<ReConfig>(RE_DEFAULTS);
  return (
    <div className="flex flex-col gap-3">
      <div className="text-[12px] text-ink2">
        Diagnoses why the alpha is weak, then searches, stage by stage: direction → shape → neutralization → horizon → turnover → conditioning → blend → settings. It keeps only changes that beat their parent in-sample and records each one as a step in a recipe. The last 2 years stay a hidden holdout that judges the result.
      </div>
      <div className="mono truncate rounded-md bg-[var(--surface-2)] px-2 py-1.5 text-[11.5px]" title={expr}>
        {expr}
      </div>
      <ReengineerOptions value={cfg} onChange={setCfg} />
      <div className="flex justify-end">
        <button className="btn btn-primary" disabled={busy || !expr.trim()} onClick={() => onStart(cfg)}>
          {busy ? <Spinner size={13} /> : <Wand2 size={13} />} Start re-engineering
        </button>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- report pieces

type Better = "up" | "down" | "none";

function Delta({ a, b, better = "up", pct = false, digits = 2 }: { a?: number | null; b?: number | null; better?: Better; pct?: boolean; digits?: number }) {
  if (a == null || b == null) return <span className="text-muted">—</span>;
  const d = b - a;
  if (Math.abs(d) < 1e-9) return <span className="text-muted">no change</span>;
  const good = better === "none" ? null : better === "up" ? d > 0 : d < 0;
  const pp = d * 100;
  const txt = `${d > 0 ? "+" : ""}${pct ? `${pp.toFixed(Math.abs(pp) < 1 ? 1 : 0)}pp` : d.toFixed(digits)}`;
  return (
    <span className={clsx("whitespace-nowrap", good === null ? "text-muted" : good ? "text-[var(--good-text)]" : "text-[var(--bad-text)]")}>
      {d > 0 ? "▲" : "▼"} {txt}
    </span>
  );
}

function CompareTable({ orig, cur, curLabel, final }: { orig: ReFinalRow; cur: { is: ReMetrics; os?: ReMetrics; sub_sharpe?: number | null; failed?: string[] }; curLabel: string; final: boolean }) {
  const rows: { k: string; a?: number | null; b?: number | null; f: (v?: number | null) => string; better: Better; pct?: boolean }[] = [
    { k: "IS Sharpe", a: orig.is.sharpe, b: cur.is.sharpe, f: (v) => fmt.num(v), better: "up" },
    { k: "IS Fitness", a: orig.is.fitness, b: cur.is.fitness, f: (v) => fmt.num(v), better: "up" },
    { k: "Turnover", a: orig.is.turnover, b: cur.is.turnover, f: (v) => fmt.pct(v, 0), better: "none", pct: true },
    { k: "Returns", a: orig.is.returns, b: cur.is.returns, f: (v) => fmt.pct(v, 1), better: "up", pct: true },
    { k: "Drawdown", a: orig.is.drawdown, b: cur.is.drawdown, f: (v) => fmt.pct(v, 1), better: "down", pct: true },
    { k: "Max weight", a: orig.is.max_weight, b: cur.is.max_weight, f: (v) => fmt.pct(v, 1), better: "down", pct: true },
  ];
  if (final) {
    rows.push({ k: "Sub-universe Sharpe", a: orig.sub_sharpe, b: cur.sub_sharpe, f: (v) => fmt.num(v), better: "up" });
    rows.push({ k: "OS Sharpe (holdout)", a: orig.os?.sharpe, b: cur.os?.sharpe, f: (v) => fmt.num(v), better: "up" });
  }
  return (
    <table className="tbl tnum">
      <thead>
        <tr>
          <th>Metric</th>
          <th className="text-right">Original</th>
          <th className="text-right">{curLabel}</th>
          <th className="text-right">Change</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.k}>
            <td className="text-ink2">{r.k}</td>
            <td className="text-right">{r.f(r.a)}</td>
            <td className="text-right font-semibold">{r.f(r.b)}</td>
            <td className="text-right">
              <Delta a={r.a} b={r.b} better={r.better} pct={r.pct} />
            </td>
          </tr>
        ))}
        {cur.failed && (
          <tr>
            <td className="text-ink2">Failing hard checks</td>
            <td className="text-right">{orig.failed.length}</td>
            <td className="text-right font-semibold">{cur.failed.length}</td>
            <td className="text-right">
              <Delta a={orig.failed.length} b={cur.failed.length} better="down" digits={0} />
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

function Recipe({ steps, onApply }: { steps: ReStep[]; onApply: (expr: string, s: Partial<Settings>) => void }) {
  if (!steps.length) return <div className="py-2 text-[12px] text-muted">No accepted changes yet.</div>;
  return (
    <ol className="flex flex-col divide-y divide-[var(--hairline)]">
      {steps.map((s, i) => (
        <li key={i} className="flex items-start gap-3 py-2">
          <span className="tnum mt-0.5 w-5 shrink-0 text-right text-[12px] font-semibold text-muted">{i + 1}</span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="chip">{s.stage_title}</span>
              <span className="text-[12.5px] font-semibold text-ink">{s.label}</span>
            </div>
            <div className="text-[11.5px] text-muted">{s.reason}</div>
            <div className="mono mt-0.5 truncate text-[11.5px] text-ink2" title={s.expr}>
              {s.expr}
            </div>
            <div className="text-[11px] text-muted">{settingsLabel(s.settings)}</div>
          </div>
          <div className="tnum grid shrink-0 grid-cols-3 gap-x-3 text-right text-[11.5px]">
            <span className="text-muted">Sharpe</span>
            <span className="text-muted">Fitness</span>
            <span className="text-muted">TO</span>
            <span>{fmt.num(s.metrics.sharpe)}</span>
            <span>{fmt.num(s.metrics.fitness)}</span>
            <span>{fmt.pct(s.metrics.turnover, 0)}</span>
            <Delta a={0} b={s.delta.sharpe} />
            <Delta a={0} b={s.delta.fitness} />
            <Delta a={0} b={s.delta.turnover} better="none" pct />
          </div>
          <button className="btn shrink-0" title="Load this intermediate version into the editor" onClick={() => onApply(s.expr, s.settings)}>
            Load
          </button>
        </li>
      ))}
    </ol>
  );
}

function Exposures({ before, after }: { before: Record<string, number>; after?: Record<string, number> }) {
  const names: Record<string, string> = { size: "Size", momentum: "12-month momentum", reversal: "Short-term reversal", volatility: "Volatility", liquidity: "Share turnover" };
  const keys = Object.keys(before);
  if (!keys.length) return null;
  return (
    <table className="tbl tnum">
      <thead>
        <tr>
          <th>Style factor</th>
          <th className="text-right">PnL corr. before</th>
          {after && <th className="text-right">after</th>}
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => (
          <tr key={k}>
            <td className="text-ink2">{names[k] ?? k}</td>
            <td className={clsx("text-right", Math.abs(before[k]) >= 0.4 && "font-semibold")}>{fmt.num(before[k])}</td>
            {after && <td className={clsx("text-right", Math.abs(after[k] ?? 0) >= 0.4 && "font-semibold")}>{fmt.num(after[k])}</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Alternatives({ rows, onApply, onOpen }: { rows: ReFinalRow[]; onApply: (expr: string, s: Partial<Settings>) => void; onOpen: (id: number) => void }) {
  return (
    <table className="tbl tnum">
      <thead>
        <tr>
          <th>Status</th>
          <th className="text-right">IS Sh.</th>
          <th className="text-right">Fitness</th>
          <th className="text-right">TO</th>
          <th className="text-right">OS Sh.</th>
          <th>Expression · recipe</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td>
              <StatusBadge status={r.status} compact />
            </td>
            <td className="text-right">{fmt.num(r.is.sharpe)}</td>
            <td className="text-right">{fmt.num(r.is.fitness)}</td>
            <td className="text-right">{fmt.pct(r.is.turnover, 0)}</td>
            <td className="text-right">{fmt.num(r.os?.sharpe)}</td>
            <td className="max-w-[420px]">
              <div className="mono truncate text-[11.5px]" title={r.expr}>
                {r.expr}
              </div>
              <div className="truncate text-[11px] text-muted" title={r.recipe}>
                {r.recipe}
              </div>
            </td>
            <td className="whitespace-nowrap text-right">
              <button className="btn btn-ghost" onClick={() => onApply(r.expr, r.settings)}>
                Load
              </button>
              {r.alpha_id ? (
                <button className="btn btn-ghost" onClick={() => onOpen(r.alpha_id!)}>
                  #{r.alpha_id}
                </button>
              ) : null}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// --------------------------------------------------------------------------- main view

/** Live and final view of a re-engineer job: diagnosis, before/after, recipe, holdout verdict, alternatives. */
export function ReengineerView({ jobId, onApply }: { jobId: number; onApply?: (expr: string, s: Partial<Settings>) => void }) {
  const nav = useNavigate();
  const setStudio = useUI((s) => s.setStudio);
  const live = useJobs((s) => s.re[jobId]);
  const job = useJobs((s) => s.jobs[jobId]);
  const isLive = job ? LIVE.has(job.status) : false;
  const { data: fetched, isFetching, error } = useQuery({
    queryKey: ["reengineer", jobId, job?.status],
    queryFn: () => api.reengineer(jobId),
    enabled: !live || !isLive,
    retry: (n) => n < 20,
    retryDelay: 1500,
  });
  const rep: ReReport | undefined = live && (isLive || !fetched || live.final) ? live : fetched ?? live;
  const apply = onApply ?? ((expr: string, s: Partial<Settings>) => {
    setStudio(expr, s);
    nav("/studio");
  });
  const openAlpha = (id: number) => nav(`/library?open=${id}`);

  if (!rep) {
    if (job?.status === "error") return <Card><div className="text-[12px] text-[var(--bad-text)]">{job.error}</div></Card>;
    return (
      <Card>
        <div className="flex items-center gap-2 py-4 text-[12px] text-muted">
          <Spinner /> {error ? "Waiting for the first report…" : isFetching ? "Loading report…" : "Diagnosing the original alpha…"}
        </div>
      </Card>
    );
  }
  const orig = rep.final?.original ?? rep.original;
  const fin = rep.final;
  const champ = fin?.champion;
  const best = rep.best;
  const p = job?.progress || {};
  const steps = [...(rep.trajectory ?? [])];
  const cfgBar = 1.25;

  return (
    <div className="flex flex-col gap-3">
      <Card
        title={
          <span className="flex items-center gap-2">
            <Wand2 size={13} /> Re-engineer · job #{jobId}
            {isLive ? <StatusBadge status="PENDING" label={job?.status ?? "running"} compact /> : <StatusBadge status={rep.status === "done" ? "PASS" : "WARNING"} label={rep.status} compact />}
          </span>
        }
        actions={
          isLive && (
            <button className="btn" onClick={() => api.jobAction(jobId, "cancel")} title="Stop searching and evaluate the best versions found so far on the holdout">
              <Square size={12} /> Stop & keep best
            </button>
          )
        }
      >
        {isLive && (
          <>
            <div className="mb-1 flex items-center justify-between text-[12px]">
              <span className="text-ink2">{p.phase || "running"}</span>
              <span className="tnum text-muted">
                {rep.n_evaluated ?? 0} variants tested · {Math.round(rep.elapsed_s ?? 0)}s
              </span>
            </div>
            <Progress value={p.total ? (p.done || 0) / p.total : 0} />
          </>
        )}
        {fin && (
          <div className={clsx("mt-1 flex items-start gap-2 rounded-md border px-3 py-2", fin.verdict.level === "good" ? "border-[var(--status-good)]" : fin.verdict.level === "warn" ? "border-[var(--status-warning)]" : "border-[var(--status-critical)]")}>
            <StatusBadge status={VERDICT_STATUS[fin.verdict.level]} label="" />
            <div className="min-w-0">
              <div className="text-[13px] font-semibold">{fin.verdict.title}</div>
              <div className="text-[12px] text-ink2">{fin.verdict.text}</div>
              <div className="mt-0.5 text-[11.5px] text-muted">
                {fin.n_evaluated} variants tested in {Math.round(fin.elapsed_s)}s
                {fin.kinship != null && ` · shares ${Math.round(fin.kinship * 100)}% of its PnL pattern with the original${fin.kinship_flipped ? " (after flipping its sign)" : ""}`}
              </div>
            </div>
          </div>
        )}
        {champ && (
          <div className="mt-3">
            <div className="lbl mb-1">Re-engineered alpha</div>
            <div className="mono whitespace-pre-wrap break-all rounded-md bg-[var(--surface-2)] px-2 py-1.5 text-[12.5px]">{champ.expr}</div>
            <div className="mt-1 text-[11.5px] text-muted">{settingsLabel(champ.settings)}</div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <button className="btn btn-primary" onClick={() => apply(champ.expr, champ.settings)}>
                <FlaskConical size={13} /> Load into Studio
              </button>
              {champ.alpha_id ? (
                <button className="btn" onClick={() => openAlpha(champ.alpha_id!)}>
                  <Library size={13} /> Library #{champ.alpha_id}
                </button>
              ) : null}
              <button
                className="btn"
                onClick={() => {
                  copyText(champ.expr);
                  toast.success("Expression copied");
                }}
              >
                <Clipboard size={13} /> Copy
              </button>
            </div>
          </div>
        )}
        {!fin && best && best.steps > 0 && (
          <div className="mt-3">
            <div className="lbl mb-1">Best so far</div>
            <div className="mono truncate text-[12px]" title={best.expr}>
              {best.expr}
            </div>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        {orig && (
          <Card title={fin ? "Before → after" : "Original → best so far"}>
            <CompareTable orig={orig} cur={champ ?? { is: best?.metrics ?? orig.is }} curLabel={champ ? "Re-engineered" : "Best so far"} final={!!champ} />
          </Card>
        )}
        <Card title="Why it was weak">
          <div className="flex flex-col divide-y divide-[var(--hairline)]">
            {(rep.diagnosis ?? []).map((d) => (
              <div key={d.code} className="flex items-start gap-2 py-1.5">
                <StatusBadge status={DIAG_STATUS[d.severity]} label="" compact />
                <div className="min-w-0">
                  <div className="text-[12.5px] font-medium">{d.title}</div>
                  <div className="text-[11.5px] text-muted">
                    {d.detail} <ArrowRight size={10} className="inline" /> {d.stage} stage
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {fin?.series && (
        <Card title="Cumulative PnL: original vs re-engineered">
          <PnlCompareChart dates={fin.series.dates} original={fin.series.original} champion={fin.series.champion} osStart={fin.series.os_start} />
        </Card>
      )}

      <Card title={`Recipe · ${(champ?.lineage ?? best?.lineage ?? []).length} accepted steps`}>
        <Recipe steps={champ?.lineage ?? best?.lineage ?? []} onApply={apply} />
      </Card>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        {steps.length > 1 && (
          <Card title="Best IS Sharpe and fitness after each stage">
            <ReTrajectoryChart steps={steps} sharpeBar={cfgBar} />
          </Card>
        )}
        <Card title="Search log" bodyClass="px-0 pb-1">
          {rep.stages.length === 0 ? (
            <Empty title="Search starting" />
          ) : (
            <div className="max-h-[260px] overflow-auto">
              <table className="tbl tnum">
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th className="text-right">Tried</th>
                    <th className="text-right">Kept</th>
                    <th className="text-right">Gain</th>
                    <th>New best</th>
                  </tr>
                </thead>
                <tbody>
                  {rep.stages.map((s, i) => (
                    <tr key={i}>
                      <td className="whitespace-nowrap">{s.pass ? `Pass ${s.pass} · ` : ""}{s.title}</td>
                      <td className="text-right">{s.tried}</td>
                      <td className="text-right">{s.accepted}</td>
                      <td className="text-right">{s.gain > 0 ? `+${s.gain.toFixed(2)}` : "—"}</td>
                      <td className="max-w-[240px] truncate text-ink2" title={s.best_label ?? ""}>
                        {s.best_label ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {rep.exposures && Object.keys(rep.exposures).length > 0 && (
        <Card title="Style exposure (IS PnL correlation with plain factors)">
          <Exposures before={rep.exposures} after={fin?.exposures_after} />
        </Card>
      )}

      {fin && fin.alternatives.length > 0 && (
        <Card title="Alternatives (compare before submitting; simpler often holds up better)" bodyClass="px-0 pb-1">
          <Alternatives rows={fin.alternatives} onApply={apply} onOpen={openAlpha} />
        </Card>
      )}
    </div>
  );
}
