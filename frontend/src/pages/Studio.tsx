import * as Popover from "@radix-ui/react-popover";
import { useQueryClient } from "@tanstack/react-query";
import { Braces, Clipboard, CloudUpload, Grid3x3, History, Play, Save, Stethoscope, Wand2, X } from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Heatmap } from "../components/charts";
import { ExpressionEditor } from "../components/ExpressionEditor";
import { type ReConfig, ReengineerLauncher, ReengineerView, startReengineer } from "../components/Reengineer";
import {
  AttributionCard,
  BrainOnlyNotice,
  ChartsCard,
  ChecksList,
  CorrelationCard,
  DescriptionCard,
  MetricStrip,
  QualityCard,
  RobustnessCard,
} from "../components/ResultView";
import { SettingsBar } from "../components/SettingsBar";
import { Card, Empty, Select, Spinner, StatusBadge, Tip } from "../components/ui";
import { api, copyText, type DoctorFix } from "../lib/api";
import { clsx, fmt, settingsLabel } from "../lib/format";
import { useHotkey } from "../lib/hooks";
import { useUI } from "../lib/store";
import type { Analysis, Settings, SimResult } from "../lib/types";

interface SweepCell {
  decay: number;
  neutralization: string;
  truncation: number;
  sharpe: number;
  fitness: number;
  turnover: number;
}

export function brainPayload(text: string, s: Settings) {
  return {
    type: "REGULAR",
    settings: {
      instrumentType: s.instrumentType ?? "EQUITY",
      region: s.region,
      universe: s.universe,
      delay: s.delay,
      decay: s.decay,
      neutralization: s.neutralization,
      truncation: s.truncation,
      pasteurization: s.pasteurization,
      unitHandling: s.unitHandling ?? "VERIFY",
      nanHandling: s.nanHandling,
      language: "FASTEXPR",
      visualization: false,
    },
    regular: text.trim(),
  };
}

function AnalysisStrip({ a }: { a: Analysis | null }) {
  if (!a) return <div className="h-5" />;
  const errs = a.diagnostics.filter((d) => d.severity === "error");
  const warns = a.diagnostics.filter((d) => d.severity === "warning");
  const infos = a.diagnostics.filter((d) => d.severity === "info");
  return (
    <div className="flex min-h-5 flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]">
      {errs.length ? (
        <span className="flex items-center gap-1 text-[var(--bad-text)]">
          <StatusBadge status="FAIL" label="" compact />
          {errs[0].message}
        </span>
      ) : (
        <>
          <StatusBadge status="PASS" label="Valid" compact />
          <span className={clsx("chip", !a.local && "!text-[var(--warn-text)]")}>{a.local ? "simulates locally" : "BRAIN-only"}</span>
          {a.tags && (
            <>
              <span className="chip">idea: {a.tags.idea}</span>
              <span className="chip">data: {a.tags.category}</span>
              <span className="chip">horizon: {a.tags.horizon}</span>
            </>
          )}
          <span className="text-muted">
            {a.op_count} ops · depth {a.depth} · lookback {a.lookback}d · fields: {a.fields.join(", ")}
          </span>
        </>
      )}
      {warns.map((w, i) => (
        <span key={`w${i}`} className="text-[var(--warn-text)]">
          ⚠ {w.message}
        </span>
      ))}
      {infos.map((w, i) => (
        <span key={`i${i}`} className="text-muted">
          ℹ {w.message}
        </span>
      ))}
    </div>
  );
}

function DoctorPanel({ fixes, loading, failed, onApply, onClose }: { fixes: DoctorFix[]; loading: boolean; failed: string[]; onApply: (f: DoctorFix) => void; onClose: () => void }) {
  return (
    <Card
      title={
        <span className="flex items-center gap-1.5">
          <Stethoscope size={13} /> Doctor {failed.length ? `· fixing ${failed.join(", ")}` : ""}
        </span>
      }
      actions={
        <>
          {loading && <Spinner />}
          <button className="btn btn-ghost !p-1" onClick={onClose} aria-label="Close doctor">
            <X size={14} />
          </button>
        </>
      }
    >
      {!loading && fixes.length === 0 && <Empty title="Nothing to fix">All hard checks and robustness gates already pass.</Empty>}
      <div className="flex flex-col divide-y divide-[var(--hairline)]">
        {fixes.map((f, i) => (
          <div key={i} className="flex items-start gap-3 py-2">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[12.5px] font-semibold text-ink">{f.label}</span>
                <StatusBadge status={f.n_failed === 0 ? "PASS" : "FAIL"} label={f.n_failed === 0 ? "passes all hard checks" : `${f.n_failed} still failing`} compact />
              </div>
              <div className="text-[11.5px] text-muted">{f.reason}</div>
              <div className="mono mt-0.5 truncate text-[11.5px] text-ink2" title={f.expr}>
                {f.expr}
              </div>
              {f.kind === "settings" && <div className="text-[11px] text-muted">{settingsLabel(f.settings)}</div>}
            </div>
            <div className="tnum grid shrink-0 grid-cols-3 gap-x-3 text-right text-[11.5px]">
              <span className="text-muted">Sharpe</span>
              <span className="text-muted">Fitness</span>
              <span className="text-muted">TO</span>
              <span>{fmt.num(f.metrics?.sharpe)}</span>
              <span>{fmt.num(f.metrics?.fitness)}</span>
              <span>{fmt.pct(f.metrics?.turnover, 0)}</span>
              <span className={f.delta.sharpe >= 0 ? "text-[var(--good-text)]" : "text-[var(--bad-text)]"}>{f.delta.sharpe >= 0 ? "+" : ""}{f.delta.sharpe.toFixed(2)}</span>
              <span className={f.delta.fitness >= 0 ? "text-[var(--good-text)]" : "text-[var(--bad-text)]"}>{f.delta.fitness >= 0 ? "+" : ""}{f.delta.fitness.toFixed(2)}</span>
              <span className="text-muted">{f.delta.turnover >= 0 ? "+" : ""}{(f.delta.turnover * 100).toFixed(0)}pp</span>
            </div>
            <button className="btn btn-primary shrink-0" onClick={() => onApply(f)}>
              Apply
            </button>
          </div>
        ))}
      </div>
    </Card>
  );
}

function SweepPanel({ cells, running, best, onApply, onClose }: { cells: SweepCell[]; running: boolean; best: SweepCell | null; onApply: (c: SweepCell) => void; onClose: () => void }) {
  const truncs = [...new Set(cells.map((c) => c.truncation))].sort((a, b) => a - b);
  const [tr, setTr] = useState<number | null>(null);
  const cur = tr ?? best?.truncation ?? truncs[0];
  const decays = [...new Set(cells.map((c) => c.decay))].sort((a, b) => a - b);
  const neuts = ["MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"].filter((n) => cells.some((c) => c.neutralization === n));
  const sub = cells.filter((c) => c.truncation === cur);
  const data: [number, number, number | null][] = [];
  for (const c of sub) data.push([decays.indexOf(c.decay), neuts.indexOf(c.neutralization), c.fitness]);
  const hl: [number, number] | null = best && best.truncation === cur ? [decays.indexOf(best.decay), neuts.indexOf(best.neutralization)] : null;
  const ranked = [...cells].sort((a, b) => b.fitness - a.fitness).slice(0, 8);
  return (
    <Card
      title={
        <span className="flex items-center gap-1.5">
          <Grid3x3 size={13} /> Settings sweep · IS fitness {running ? `(${cells.length} cells…)` : ""}
        </span>
      }
      actions={
        <>
          {running && <Spinner />}
          <span className="text-[11px] text-muted">truncation</span>
          <Select value={cur ?? ""} onChange={(v) => setTr(Number(v))} options={truncs.map((t) => ({ value: t, label: String(t) }))} />
          <button className="btn btn-ghost !p-1" onClick={onClose} aria-label="Close sweep">
            <X size={14} />
          </button>
        </>
      }
    >
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <Heatmap
            xLabels={decays.map((d) => `decay ${d}`)}
            yLabels={neuts}
            cells={data}
            height={220}
            valueLabel="fitness"
            highlight={hl}
            onCell={(x, y) => {
              const c = sub.find((k) => k.decay === decays[x] && k.neutralization === neuts[y]);
              if (c) onApply(c);
            }}
          />
          <div className="text-[11px] text-muted">Click a cell to apply its settings. The outlined cell is the robust pick (best neighbourhood), not just the peak.</div>
        </div>
        <div className="lg:col-span-2">
          <table className="tbl tnum">
            <thead>
              <tr>
                <th>Decay</th>
                <th>Neut.</th>
                <th>Trunc</th>
                <th className="text-right">Sharpe</th>
                <th className="text-right">Fitness</th>
                <th className="text-right">TO</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((c, i) => (
                <tr key={i} className="cursor-pointer" onClick={() => onApply(c)}>
                  <td>{c.decay}</td>
                  <td>{c.neutralization.toLowerCase()}</td>
                  <td>{c.truncation}</td>
                  <td className="text-right">{fmt.num(c.sharpe)}</td>
                  <td className="text-right">{fmt.num(c.fitness)}</td>
                  <td className="text-right">{fmt.pct(c.turnover, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  );
}

export default function Studio() {
  const { studioText, studioSettings, setStudio, history, pushHistory, studioReJob, setStudioReJob } = useUI();
  const [re, setRe] = useState<{ open: boolean; starting: boolean }>({ open: false, starting: false });
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [result, setResult] = useState<SimResult | null>(null);
  const [running, setRunning] = useState(false);
  const [extrasLoading, setExtrasLoading] = useState(false);
  const [doctor, setDoctor] = useState<{ open: boolean; loading: boolean; fixes: DoctorFix[]; failed: string[] }>({ open: false, loading: false, fixes: [], failed: [] });
  const [sweep, setSweep] = useState<{ open: boolean; running: boolean; cells: SweepCell[]; best: SweepCell | null }>({ open: false, running: false, cells: [], best: null });
  const runId = useRef(0);
  const sweepAbort = useRef<AbortController | null>(null);
  const nav = useNavigate();
  const qc = useQueryClient();

  const run = useCallback(
    async (text = studioText, settings = studioSettings) => {
      if (!text.trim()) return;
      const id = ++runId.current;
      setRunning(true);
      try {
        const r = await api.simulate(text, settings, false);
        if (id !== runId.current) return;
        setResult(r);
        if (!r.ok) {
          toast.error(r.error || r.analysis?.diagnostics?.find((d) => d.severity === "error")?.message || "Invalid expression");
          return;
        }
        if (r.brain_only) return;
        const m = r.metrics?.is;
        pushHistory({ text, settings, sharpe: m?.sharpe, fitness: m?.fitness, turnover: m?.turnover, status: r.checks?.status, at: Date.now() });
        setExtrasLoading(true);
        const ex = await api.extras(text, settings);
        if (id !== runId.current) return;
        if (ex.ok) setResult((prev) => (prev ? { ...prev, extras: ex.extras, checks: ex.checks, pass_prob: ex.pass_prob, quality: ex.quality ?? prev.quality } : prev));
      } catch (e) {
        toast.error(String((e as Error).message || e));
      } finally {
        if (id === runId.current) {
          setRunning(false);
          setExtrasLoading(false);
        }
      }
    },
    [studioText, studioSettings, pushHistory],
  );

  const save = useCallback(async () => {
    try {
      const r = await api.save(studioText, studioSettings);
      if (r.ok) {
        toast.success(`Saved as alpha #${r.id}${r.status ? ` (${r.status})` : ""}`, { action: { label: "Library", onClick: () => nav(`/library?open=${r.id}`) } });
        qc.invalidateQueries({ queryKey: ["alphas"] });
      } else toast.error("Could not save: fix the expression first");
    } catch (e) {
      toast.error(String((e as Error).message));
    }
  }, [studioText, studioSettings, nav, qc]);

  const runDoctor = async () => {
    setDoctor({ open: true, loading: true, fixes: [], failed: [] });
    try {
      const d = await api.doctor(studioText, studioSettings);
      setDoctor({ open: true, loading: false, fixes: d.fixes || [], failed: d.base_failed || [] });
    } catch (e) {
      setDoctor({ open: true, loading: false, fixes: [], failed: [] });
      toast.error(String((e as Error).message));
    }
  };

  const runSweep = async () => {
    sweepAbort.current?.abort();
    const ac = new AbortController();
    sweepAbort.current = ac;
    setSweep({ open: true, running: true, cells: [], best: null });
    try {
      await api.sweep(studioText, studioSettings, undefined, (ev) => {
        if (ev.type === "cell") setSweep((s) => ({ ...s, cells: [...s.cells, ev.cell] }));
        if (ev.type === "best") setSweep((s) => ({ ...s, best: ev.cell }));
      }, ac.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") toast.error(String((e as Error).message));
    } finally {
      setSweep((s) => ({ ...s, running: false }));
    }
  };

  useHotkey("Enter", () => run());
  useHotkey("s", () => save());

  const applyFix = (f: DoctorFix) => {
    setStudio(f.expr, f.settings);
    run(f.expr, { ...studioSettings, ...f.settings });
  };
  const startRe = async (cfg: ReConfig) => {
    setRe((r) => ({ ...r, starting: true }));
    try {
      const id = await startReengineer(studioText, studioSettings, cfg);
      setStudioReJob(id);
      toast.success(`Re-engineering started (job #${id})`);
    } catch (e) {
      toast.error(String((e as Error).message));
    } finally {
      setRe((r) => ({ ...r, starting: false }));
    }
  };
  const applyRe = (expr: string, s: Partial<Settings>) => {
    setStudio(expr, s);
    run(expr, { ...studioSettings, ...s });
  };
  const applyCell = (c: SweepCell) => {
    const s = { ...studioSettings, decay: c.decay, neutralization: c.neutralization, truncation: c.truncation };
    setStudio(undefined, s);
    run(studioText, s);
  };

  const hasErr = analysis?.diagnostics.some((d) => d.severity === "error") ?? false;
  const recent = useMemo(() => history.slice(0, 25), [history]);

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="card flex flex-col gap-2 p-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <button className="btn btn-primary" onClick={() => run()} disabled={running || hasErr}>
            {running ? <Spinner size={13} /> : <Play size={13} />} Simulate <span className="opacity-70">Ctrl+Enter</span>
          </button>
          <button className="btn" onClick={save} disabled={hasErr}>
            <Save size={13} /> Save
          </button>
          <button className="btn" onClick={runDoctor} disabled={hasErr || !analysis?.local}>
            <Stethoscope size={13} /> Doctor
          </button>
          <button className="btn" onClick={runSweep} disabled={hasErr || !analysis?.local}>
            <Grid3x3 size={13} /> Settings sweep
          </button>
          <Tip content="Diagnose why this alpha is weak and search for a much stronger version, step by step">
            <button className="btn" onClick={() => setRe((r) => ({ ...r, open: !r.open }))} disabled={!re.open && (hasErr || !analysis?.local)}>
              <Wand2 size={13} /> Re-engineer
            </button>
          </Tip>
          <div className="mx-1 h-5 w-px bg-[var(--hairline)]" />
          <Tip content="Copy the expression to paste into BRAIN">
            <button
              className="btn btn-ghost"
              onClick={() => {
                copyText(studioText.trim());
                toast.success("Expression copied");
              }}
            >
              <Clipboard size={13} /> Copy
            </button>
          </Tip>
          <Tip content="Copy a BRAIN simulation payload (settings + expression) as JSON">
            <button
              className="btn btn-ghost"
              onClick={() => {
                copyText(JSON.stringify(brainPayload(studioText, studioSettings), null, 2));
                toast.success("BRAIN payload copied");
              }}
            >
              <Braces size={13} /> BRAIN JSON
            </button>
          </Tip>
          <Tip content="Simulate this alpha on BRAIN itself (needs the BRAIN connection) and store the real results">
            <button
              className="btn btn-ghost"
              disabled={hasErr}
              onClick={async () => {
                try {
                  const st = await api.brain.status();
                  if (!st.connected) {
                    toast.error("Not connected to BRAIN. Sign in on the BRAIN page first.");
                    nav("/brain");
                    return;
                  }
                  const r = await api.brain.simulate({ exprs: [studioText.trim()], settings: studioSettings });
                  toast.success(`Sent to BRAIN (job #${r.id}); results appear in the Library`);
                } catch (e) {
                  toast.error(String((e as Error).message));
                }
              }}
            >
              <CloudUpload size={13} /> Run on BRAIN
            </button>
          </Tip>
          <Popover.Root>
            <Popover.Trigger asChild>
              <button className="btn btn-ghost">
                <History size={13} /> History
              </button>
            </Popover.Trigger>
            <Popover.Portal>
              <Popover.Content align="start" sideOffset={4} className="card z-50 max-h-[420px] w-[560px] overflow-auto p-1 shadow-xl">
                {recent.length === 0 && <div className="p-3 text-[12px] text-muted">No runs yet.</div>}
                {recent.map((h, i) => (
                  <Popover.Close asChild key={i}>
                    <button
                      className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-[var(--surface-3)]"
                      onClick={() => {
                        setStudio(h.text, h.settings);
                        run(h.text, h.settings);
                      }}
                    >
                      <span className="mono min-w-0 flex-1 truncate text-[11.5px]">{h.text}</span>
                      <span className="tnum shrink-0 text-[11px] text-muted">
                        S {fmt.num(h.sharpe)} · F {fmt.num(h.fitness)} · {fmt.pct(h.turnover, 0)}
                      </span>
                    </button>
                  </Popover.Close>
                ))}
              </Popover.Content>
            </Popover.Portal>
          </Popover.Root>
          <div className="ml-auto text-[11px] text-muted">
            {result?.elapsed_ms !== undefined && `simulated in ${Math.round(result.elapsed_ms)} ms · ${result.data?.source} data`}
          </div>
        </div>
        <ExpressionEditor value={studioText} onChange={(v) => setStudio(v)} onRun={() => run()} onSave={save} onAnalysis={setAnalysis} />
        <AnalysisStrip a={analysis} />
        <SettingsBar settings={studioSettings} onChange={(p) => setStudio(undefined, p)} />
      </div>

      {doctor.open && <DoctorPanel {...doctor} onApply={applyFix} onClose={() => setDoctor((d) => ({ ...d, open: false }))} />}
      {re.open &&
        (studioReJob === null ? (
          <Card
            title={
              <span className="flex items-center gap-1.5">
                <Wand2 size={13} /> Re-engineer this alpha
              </span>
            }
            actions={
              <button className="btn btn-ghost !p-1" onClick={() => setRe((r) => ({ ...r, open: false }))} aria-label="Close re-engineer">
                <X size={14} />
              </button>
            }
          >
            <ReengineerLauncher expr={studioText} onStart={startRe} busy={re.starting} />
          </Card>
        ) : (
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-end gap-1.5">
              <button className="btn" onClick={() => setStudioReJob(null)} disabled={hasErr || !analysis?.local}>
                <Wand2 size={13} /> New run on the editor's alpha
              </button>
              <button className="btn btn-ghost !p-1" onClick={() => setRe((r) => ({ ...r, open: false }))} aria-label="Close re-engineer">
                <X size={14} />
              </button>
            </div>
            <ReengineerView jobId={studioReJob} onApply={applyRe} />
          </div>
        ))}
      {sweep.open && (
        <SweepPanel
          {...sweep}
          onApply={applyCell}
          onClose={() => {
            sweepAbort.current?.abort();
            setSweep((s) => ({ ...s, open: false }));
          }}
        />
      )}

      {!result && !running && (
        <Card>
          <Empty title="Write an alpha and press Ctrl+Enter">
            The local proxy simulator runs your Fast Expression on the active dataset and reports BRAIN-style checks, PnL, robustness and a description draft. Try{" "}
            <code className="mono">group_rank(ts_backfill(operating_income, 120) / cap, subindustry)</code>.
          </Empty>
        </Card>
      )}
      {result && result.ok && result.brain_only && <BrainOnlyNotice r={result} />}
      {result && result.ok && !result.brain_only && (
        <div className={clsx("grid grid-cols-1 gap-3 transition-opacity xl:grid-cols-12", running && "opacity-60")}>
          <div className="flex flex-col gap-3 xl:col-span-8">
            <MetricStrip r={result} />
            <ChartsCard r={result} />
            <DescriptionCard r={result} />
          </div>
          <div className="flex flex-col gap-3 xl:col-span-4">
            <QualityCard r={result} loading={extrasLoading} />
            <ChecksList r={result} loading={extrasLoading} />
            <RobustnessCard r={result} loading={extrasLoading} />
            <CorrelationCard r={result} onOpen={(id) => nav(`/library?open=${id}`)} />
            <AttributionCard r={result} />
          </div>
        </div>
      )}
    </div>
  );
}
