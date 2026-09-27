import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  Blocks,
  Braces,
  Check,
  CircleDashed,
  Clipboard,
  CloudUpload,
  Dna,
  FileUp,
  FlaskConical,
  GitMerge,
  Library,
  Lightbulb,
  Loader2,
  PenLine,
  RotateCcw,
  SkipForward,
  Sparkles,
  Square,
  Stethoscope,
  ThumbsDown,
  ThumbsUp,
  Wand2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { ChartsCard, ChecksList, DescriptionCard } from "../components/ResultView";
import { SettingsBar } from "../components/SettingsBar";
import { Card, Empty, GradeBadge, Kv, Progress, Spinner, StatusBadge, Tip, Toggle } from "../components/ui";
import { api, copyText, fileToBase64 } from "../lib/api";
import { clsx, fmt } from "../lib/format";
import { useDebounced, useHotkey, useStatus } from "../lib/hooks";
import { DEFAULT_SETTINGS, useJobs, useUI } from "../lib/store";
import type { ForgeAlpha, ForgeResult, ForgeRow, ForgeStage, IdeaSpec, JobSnapshot, Settings } from "../lib/types";

const EXAMPLES = [
  "Stocks that drop sharply on heavy volume tend to bounce back within a week.",
  "Profitable companies with low debt outperform their industry peers.",
  "Cheap stocks by cash-flow yield that are starting to trend up keep outperforming.",
  "Momentum works better among low-volatility stocks.",
  "After earnings announcements, stocks with improving EPS keep drifting up for about a month.",
  "Signal = 12-month return skipping the last month, ranked within industry; neutralize by industry, decay 4",
  "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))",
];

const FORMAT_LABEL: Record<string, string> = {
  english: "plain English", fastexpr: "Fast Expression", paper: "paper / 101-Alphas formula", python: "Python / pandas code",
  json: "JSON record", yaml: "YAML record", list: "list of ideas", document: "long document", formula: "written formula",
  mixed: "text + code", described: "described computation",
};
const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|ya?ml|py|r|sql|tex|log)$/i;

const EFFORTS = [
  { id: "quick", label: "Quick", hint: "up to ~5 min: fewer drafts, short evolution" },
  { id: "standard", label: "Standard", hint: "up to ~12 min: the balanced default" },
  { id: "deep", label: "Deep", hint: "up to ~28 min: more drafts, longer two-island evolution" },
] as const;

const STAGES = [
  { id: "interpret", label: "Interpret", icon: Lightbulb, desc: "Read the idea" },
  { id: "draft", label: "Draft", icon: PenLine, desc: "Build and screen first drafts" },
  { id: "combine", label: "Combine", icon: GitMerge, desc: "Blends, gates, neutralizations" },
  { id: "refine", label: "Refine", icon: Stethoscope, desc: "Doctor fixes, fitness shaping and settings sweep" },
  { id: "compose", label: "Compose", icon: Blocks, desc: "Complex (multi-statement) alphas from the leaders" },
  { id: "evolve", label: "Evolve", icon: Dna, desc: "Genetic programming on the leaders" },
  { id: "polish", label: "Polish", icon: Sparkles, desc: "Full checks, champion, runners-up" },
] as const;

const LOCAL_FAMILIES = ["reversion", "momentum", "seasonality", "value", "quality", "growth", "accruals", "investment", "leverage", "liquidity", "pv_divergence", "volatility"];
const LIVE = new Set(["running", "paused", "queued"]);
const DRAFT_KEY = "af-forge-idea";

function loadDraft(): string {
  try {
    return localStorage.getItem(DRAFT_KEY) ?? "";
  } catch {
    return "";
  }
}
function saveDraft(t: string) {
  try {
    localStorage.setItem(DRAFT_KEY, t);
  } catch {
    /* storage unavailable */
  }
}

function brainJson(expr: string, s: Settings) {
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
    regular: expr.trim(),
  };
}

/* ------------------------------------------------------------------ interpretation */

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-wide text-muted">{label}</span>
      {children}
    </div>
  );
}

function Interpretation({
  spec,
  loading,
  families,
  onFamilies,
  horizon,
  onHorizon,
  onOpen,
}: {
  spec: IdeaSpec | undefined;
  loading: boolean;
  families: string[] | null;
  onFamilies: (f: string[] | null) => void;
  horizon: string | null;
  onHorizon: (h: string | null) => void;
  onOpen: (expr: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  if (!spec || !spec.text.trim()) {
    return (
      <Card title="How the forge reads it">
        <Empty title="Describe a market effect in plain English">
          Say what kind of stocks, what data, and what happens next, e.g. <em>"stocks that drop on heavy volume bounce back within a week"</em>. You can also paste Fast Expressions; they become seeds.
        </Empty>
      </Card>
    );
  }
  const selected = families ?? spec.families.filter((f) => !f.brain_only).map((f) => f.id);
  const conf = spec.confidence;
  const labelOf = (id: string) => spec.all_families?.find((f) => f.id === id)?.label ?? id;
  return (
    <Card
      title={
        <span className="flex items-center gap-1.5">
          <Lightbulb size={13} /> How the forge reads it
        </span>
      }
      actions={loading && <Spinner />}
    >
      <div className={clsx("flex flex-col gap-3 transition-opacity", loading && "opacity-60")}>
        <div className="flex items-center gap-2">
          <Progress value={conf} className="max-w-[160px]" />
          <span className="text-[11.5px] text-ink2">{conf >= 0.75 ? "Clear reading" : conf >= 0.5 ? "Reasonable reading" : "Loose reading: add detail for a sharper search"}</span>
        </div>

        {spec.input_format && spec.input_format !== "english" && (
          <Section label="Read as">
            <div className="flex flex-wrap gap-1">
              {(spec.input_formats ?? [spec.input_format]).map((f) => (
                <span key={f} className="chip">{FORMAT_LABEL[f] ?? f}</span>
              ))}
              {Object.entries(spec.settings_hints ?? {}).map(([k, v]) => (
                <span key={k} className="chip !text-[var(--accent)]">{k} {String(v)}</span>
              ))}
            </div>
          </Section>
        )}
        {spec.compiled && spec.compiled.length > 0 && (
          <Section label="Formulas built from your description">
            <div className="flex flex-col divide-y divide-[var(--hairline)]">
              {spec.compiled.map((c) => (
                <button key={c.expr} className="group flex items-center gap-2 py-1 text-left" onClick={() => onOpen(c.expr)} title="Open in Studio">
                  <span className="min-w-0 flex-1">
                    <span className="mono block truncate text-[11.5px] text-ink">{c.expr}</span>
                    <span className="block truncate text-[10.5px] text-muted">{c.label}</span>
                  </span>
                  <ArrowRight size={12} className="shrink-0 text-muted opacity-0 group-hover:opacity-100" />
                </button>
              ))}
            </div>
          </Section>
        )}
        {spec.key_sentences && spec.key_sentences.length > 0 && (
          <details className="text-[12px]">
            <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-wide text-muted">Key sentences used from the document ({spec.key_sentences.length})</summary>
            <ul className="mt-1 flex flex-col gap-1 text-ink2">
              {spec.key_sentences.map((k) => (
                <li key={k}>“{k}”</li>
              ))}
            </ul>
          </details>
        )}
        {spec.sub_ideas && spec.sub_ideas.length > 1 && (
          <Section label={`${spec.sub_ideas.length} ideas, combined into one search`}>
            <ul className="flex flex-col gap-0.5 text-[12px] text-ink2">
              {spec.sub_ideas.map((x) => (
                <li key={x}>• {x}</li>
              ))}
            </ul>
          </Section>
        )}

        <Section label="Mechanism and direction">
          <div className="flex flex-col gap-1">
            {spec.families.map((f) => (
              <div key={f.id} className="flex flex-wrap items-center gap-1.5 text-[12px]">
                <span className={clsx("chip", selected.includes(f.id) && !f.brain_only && "!bg-[var(--accent)] !text-white")}>{f.label}</span>
                {!f.brain_only && (
                  <span className="text-ink2">
                    long <b>{f.direction > 0 ? "high" : "low"}</b> {f.amount}
                    <span className="text-muted"> · {f.stated ? (f.flipped ? "as stated (against the usual finding)" : "as stated") : "usual direction, reverse also tested"}</span>
                  </span>
                )}
              </div>
            ))}
          </div>
          <div className="mt-1 flex items-center gap-2">
            <button className="btn btn-ghost !px-1.5 !py-0.5 text-[11px]" onClick={() => setEditing((e) => !e)}>
              {editing ? "Done" : "Change mechanisms"}
            </button>
            {families && (
              <button className="btn btn-ghost !px-1.5 !py-0.5 text-[11px]" onClick={() => onFamilies(null)}>
                <RotateCcw size={11} /> Reset to reading
              </button>
            )}
          </div>
          {editing && (
            <div className="flex flex-wrap gap-1 fade-in">
              {LOCAL_FAMILIES.map((id) => {
                const on = selected.includes(id);
                return (
                  <button
                    key={id}
                    type="button"
                    className={clsx("chip cursor-pointer", on && "!bg-[var(--accent)] !text-white")}
                    onClick={() => {
                      const next = on ? selected.filter((x) => x !== id) : [...selected, id];
                      onFamilies(next.length ? next : null);
                    }}
                  >
                    {labelOf(id)}
                  </button>
                );
              })}
            </div>
          )}
        </Section>

        <Section label="Data">
          <div className="flex flex-wrap gap-1">
            {spec.fields.map((f) => (
              <Tip key={f.id} content={f.description || f.id}>
                <span
                  className={clsx(
                    "chip mono",
                    !f.local && "!text-[var(--warn-text)]",
                    f.local && !f.mentioned && "!bg-transparent border border-dashed border-[var(--border)]",
                  )}
                >
                  {f.id}
                  {!f.local && " · BRAIN-only"}
                </span>
              </Tip>
            ))}
          </div>
          <span className="text-[10.5px] text-muted">Solid: named in your idea · dashed: typical inputs for the mechanism</span>
        </Section>

        <div className="grid grid-cols-2 gap-3">
          <Section label="Horizon">
            <div className="flex gap-1">
              {(["short", "medium", "long"] as const).map((h) => {
                const on = (horizon ?? spec.horizon) === h;
                return (
                  <button key={h} className={clsx("chip cursor-pointer", on && "!bg-[var(--accent)] !text-white")} onClick={() => onHorizon(h === spec.horizon && !horizon ? null : h)}>
                    {h}
                  </button>
                );
              })}
            </div>
            <span className="tnum text-[10.5px] text-muted">windows {spec.windows.join(", ")}</span>
          </Section>
          <Section label="Compared within">
            <span className="text-[12px] text-ink2">{spec.groups.length ? spec.groups.map((g) => g.replace("capbucket", "size buckets")).join(", ") : "default peer groups"}</span>
          </Section>
        </div>

        {(spec.conditions.length > 0 || spec.interaction || spec.smooth || spec.seeds.length > 0) && (
          <Section label="Conditions and seeds">
            <ul className="flex flex-col gap-0.5 text-[12px] text-ink2">
              {spec.conditions.map((c) => (
                <li key={c.id}>• {c.label}</li>
              ))}
              {spec.interaction && <li>• Applies the main signal within a subset of stocks (interaction), not just alongside it</li>}
              {spec.smooth && <li>• Prefers slow, low-turnover positions</li>}
              {spec.seeds.map((s) => (
                <li key={s} className="mono truncate text-[11.5px]" title={s}>
                  • seed: {s}
                </li>
              ))}
            </ul>
          </Section>
        )}

        {spec.warnings.map((w) => (
          <div key={w} className="rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1.5 text-[11.5px] text-[var(--warn-text)]">
            {w}
          </div>
        ))}

        {spec.preview && spec.preview.length > 0 && (
          <Section label="First drafts it will try">
            <div className="flex flex-col divide-y divide-[var(--hairline)]">
              {spec.preview.map((p) => (
                <button key={p.expr} className="group flex items-center gap-2 py-1 text-left" onClick={() => onOpen(p.expr)} title="Open in Studio">
                  <span className="min-w-0 flex-1">
                    <span className="mono block truncate text-[11.5px] text-ink">{p.expr}</span>
                    <span className="block truncate text-[10.5px] text-muted">{p.label}</span>
                  </span>
                  <ArrowRight size={12} className="shrink-0 text-muted opacity-0 group-hover:opacity-100" />
                </button>
              ))}
            </div>
          </Section>
        )}

        {spec.templates && spec.templates.length > 0 && (
          <Section label="Closest library templates">
            <div className="flex flex-col gap-0.5">
              {spec.templates.slice(0, 3).map((t) => (
                <div key={t.id} className="text-[11.5px]">
                  <span className="mono text-ink">{t.id}</span> <span className="text-muted">· {t.rationale}</span>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ run view */

function StageStepper({ stages, current, live }: { stages: Record<string, ForgeStage>; current?: string; live: boolean }) {
  return (
    <div className="grid grid-cols-4 gap-1.5 md:grid-cols-7">
      {STAGES.map((s) => {
        const st = stages[s.id]?.status ?? "pending";
        const active = live && current === s.id;
        const Icon = s.icon;
        return (
          <Tip key={s.id} content={`${s.desc}${stages[s.id]?.note ? ` · ${stages[s.id]?.note}` : ""}`}>
            <div
              className={clsx(
                "flex flex-col gap-0.5 rounded-md border px-2 py-1.5",
                active ? "border-[var(--accent)] bg-[var(--surface-2)]" : "border-[var(--border)]",
                st === "pending" && !active && "opacity-60",
              )}
            >
              <div className="flex items-center gap-1.5 text-[11.5px] font-semibold">
                <Icon size={13} className={active ? "text-[var(--accent)]" : st === "done" ? "text-[var(--good-text)]" : "text-muted"} />
                {s.label}
              </div>
              <div className="flex items-center gap-1 text-[10.5px] text-muted">
                {active ? (
                  <>
                    <Loader2 size={10} className="animate-spin" /> running
                  </>
                ) : st === "done" ? (
                  <>
                    <Check size={10} /> {stages[s.id]?.candidates != null ? `${stages[s.id]?.candidates} cand.` : "done"}
                    {stages[s.id]?.seconds != null && ` · ${Math.round(stages[s.id]!.seconds!)}s`}
                  </>
                ) : st === "skipped" ? (
                  <>
                    <SkipForward size={10} /> skipped
                  </>
                ) : (
                  <>
                    <CircleDashed size={10} /> waiting
                  </>
                )}
              </div>
            </div>
          </Tip>
        );
      })}
    </div>
  );
}

function Leaderboard({ rows, onOpen }: { rows: ForgeRow[]; onOpen: (r: ForgeRow) => void }) {
  if (!rows.length) return <Empty title="Candidates appear here as they are simulated" />;
  return (
    <table className="tbl tnum">
      <thead>
        <tr>
          <th className="text-right">Score</th>
          <th className="text-right">Sharpe</th>
          <th className="text-right">Fitness</th>
          <th className="text-right">TO</th>
          <th>Stage</th>
          <th>Expression (IS)</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="cursor-pointer" onClick={() => onOpen(r)} title="Open in Studio">
            <td className="text-right">{fmt.num(r.score)}</td>
            <td className="text-right">{fmt.num(r.sharpe)}</td>
            <td className="text-right">{fmt.num(r.fitness)}</td>
            <td className="text-right">{fmt.pct(r.turnover, 0)}</td>
            <td className="text-muted">{r.stage}</td>
            <td className="mono max-w-[340px] truncate text-ink2">{r.expr}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function BigMetric({ label, value, sub, hint }: { label: string; value: string; sub?: string; hint?: string }) {
  return (
    <div className="flex min-w-[96px] flex-1 flex-col gap-0.5 rounded-md bg-[var(--surface-2)] px-2.5 py-1.5" title={hint}>
      <span className="text-[10px] font-semibold uppercase tracking-wide text-muted">{label}</span>
      <span className="tnum text-[18px] font-semibold leading-tight">{value}</span>
      {sub && <span className="tnum text-[10.5px] text-muted">{sub}</span>}
    </div>
  );
}

function AlphaActions({ a, onOpen }: { a: ForgeAlpha; onOpen: (expr: string, s: Settings) => void }) {
  const nav = useNavigate();
  return (
    <div className="flex flex-wrap gap-1.5">
      <button className="btn btn-primary" onClick={() => onOpen(a.expr, a.settings)}>
        <FlaskConical size={13} /> Open in Studio
      </button>
      <button
        className="btn"
        onClick={() => {
          copyText(a.expr);
          toast.success("Expression copied");
        }}
      >
        <Clipboard size={13} /> Copy
      </button>
      <Tip content="Copy a BRAIN simulation payload (settings + expression)">
        <button
          className="btn"
          onClick={() => {
            copyText(JSON.stringify(brainJson(a.expr, a.settings), null, 2));
            toast.success("BRAIN payload copied");
          }}
        >
          <Braces size={13} /> BRAIN JSON
        </button>
      </Tip>
      {a.id != null && (
        <Tip content="Simulate on BRAIN itself (needs the BRAIN connection)">
          <button
            className="btn"
            onClick={async () => {
              try {
                const r = await api.brain.simulate({ alpha_ids: [a.id!] });
                toast.success(`Sent to BRAIN (job #${r.id})`);
              } catch (e) {
                toast.error(String((e as Error).message));
              }
            }}
          >
            <CloudUpload size={13} /> Run on BRAIN
          </button>
        </Tip>
      )}
      {a.id != null && (
        <button className="btn btn-ghost" onClick={() => nav(`/library?open=${a.id}`)}>
          <Library size={13} /> Library #{a.id}
        </button>
      )}
    </div>
  );
}

function ChampionCard({ a, result, onOpen }: { a: ForgeAlpha; result: ForgeResult; onOpen: (expr: string, s: Settings) => void }) {
  const { data, isFetching } = useQuery({ queryKey: ["alpha", a.id], queryFn: () => api.alpha(a.id!), enabled: a.id != null, staleTime: 60_000 });
  const hyp = result.hypothesis;
  return (
    <div className="flex flex-col gap-3 fade-in">
      <section className="card overflow-hidden">
        <div className="flex items-center gap-2 border-b border-[var(--hairline)] bg-[var(--surface-2)] px-3 py-2">
          <Wand2 size={15} className="text-[var(--accent)]" />
          <span className="text-[13px] font-semibold">Champion alpha</span>
          <GradeBadge grade={a.grade} />
          {a.complex && <span className="chip">complex</span>}
          <StatusBadge status={a.status === "PASS" ? "PASS" : "FAIL"} label={a.status === "PASS" ? "passes all local checks" : `fails ${a.failed.join(", ")}`} compact />
          <span className="chip ml-auto">{a.family_label}</span>
          <Tip content="How recognisably the alpha still expresses your idea: the data you named, the mechanism, and no unrelated data">
            <span className="chip">idea fidelity {Math.round(a.fidelity * 100)}%</span>
          </Tip>
        </div>
        <div className="flex flex-col gap-3 p-3">
          <div className="mono select-all whitespace-pre-wrap break-words rounded-md border border-[var(--border)] bg-[var(--editor-bg)] px-3 py-2.5 text-[13px] leading-relaxed text-ink">{a.expr}</div>
          {(a.quality_reasons?.length || a.quality_evidence?.length) ? (
            <div className="flex flex-col gap-0.5 text-[11.5px]">
              {a.quality_evidence && a.quality_evidence.length > 0 && <span className="text-[var(--good-text)]">✓ {a.quality_evidence.join(" · ")}</span>}
              {(a.quality_reasons ?? []).slice(0, 3).map((x) => (
                <span key={x} className="text-[var(--warn-text)]">• {x}</span>
              ))}
            </div>
          ) : null}
          <div className="text-[11px] text-muted">
            decay {a.settings.decay} · {a.settings.neutralization.toLowerCase()} neutralization · truncation {a.settings.truncation} · {a.settings.universe} · delay {a.settings.delay}
          </div>
          <div className="flex flex-wrap gap-2">
            <BigMetric label="Sharpe" value={fmt.num(a.sharpe)} sub={`OS ${fmt.num(a.os_sharpe)}`} hint="In-sample; OS is the held-out period the forge never selected on" />
            <BigMetric label="Fitness" value={fmt.num(a.fitness)} sub={`OS ${fmt.num(a.os_fitness)}`} />
            <BigMetric label="Turnover" value={fmt.pct(a.turnover, 1)} />
            <BigMetric label="Returns" value={fmt.pct(a.returns, 2)} sub={`drawdown ${fmt.pct(a.drawdown, 1)}`} />
            <BigMetric label="Sub-universe" value={fmt.num(a.sub_sharpe)} sub={a.stability != null ? `stability ${fmt.pct(a.stability, 0)}` : undefined} hint="Sharpe in the smaller sub-universe; stability = Sharpe kept when windows move ±25%" />
            <BigMetric label="P(pass)" value={`${Math.round((a.pass_prob ?? 0) * 100)}%`} sub={a.expected_brain_sharpe != null ? `BRAIN Sharpe ~${fmt.num(a.expected_brain_sharpe)}` : "calibrates with imports"} />
          </div>
          <AlphaActions a={a} onOpen={onOpen} />
          {result.message && <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-2 text-[12px] text-[var(--warn-text)]">{result.message}</div>}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">How it was forged</div>
              <ol className="flex flex-col gap-1">
                {a.lineage.map((l, i) => (
                  <li key={i} className="flex gap-2 text-[12px]">
                    <span className="tnum mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[var(--surface-3)] text-[10px] font-semibold">{i + 1}</span>
                    <span className="text-ink2">{l}</span>
                  </li>
                ))}
              </ol>
            </div>
            {hyp && hyp.text && (
              <div>
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">Hypothesis check</div>
                <div className="flex gap-2 text-[12px]">
                  {hyp.holds ? <ThumbsUp size={14} className="mt-0.5 shrink-0 text-[var(--good-text)]" /> : <ThumbsDown size={14} className="mt-0.5 shrink-0 text-[var(--warn-text)]" />}
                  <span className="text-ink2">{hyp.text}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
      {isFetching && !data && (
        <div className="flex items-center gap-2 text-[12px] text-muted">
          <Spinner /> loading charts and checks…
        </div>
      )}
      {data?.result && (
        <div className="grid grid-cols-1 gap-3 2xl:grid-cols-5">
          <div className="flex flex-col gap-3 2xl:col-span-3">
            <ChartsCard r={data.result} />
            <DescriptionCard r={data.result} />
          </div>
          <div className="2xl:col-span-2">
            <ChecksList r={data.result} />
          </div>
        </div>
      )}
    </div>
  );
}

function AlphaRowCard({ title, a, note, onOpen }: { title: string; a: ForgeAlpha; note?: string; onOpen: (expr: string, s: Settings) => void }) {
  return (
    <div className="flex flex-col gap-1.5 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[12px] font-semibold">{title}</span>
        <GradeBadge grade={a.grade} />
        <StatusBadge status={a.status === "PASS" ? "PASS" : "FAIL"} label={a.status === "PASS" ? "pass" : a.failed.join(", ")} compact />
        <span className="tnum ml-auto text-[11.5px] text-muted">
          Sharpe {fmt.num(a.sharpe)} · fitness {fmt.num(a.fitness)} · TO {fmt.pct(a.turnover, 0)} · OS {fmt.num(a.os_sharpe)}
        </span>
      </div>
      <div className="mono whitespace-pre-wrap break-words text-[11.5px] text-ink2">{a.expr}</div>
      {note && <div className="text-[11px] text-muted">{note}</div>}
      <div className="flex gap-1.5">
        <button className="btn !py-1" onClick={() => onOpen(a.expr, a.settings)}>
          <FlaskConical size={12} /> Studio
        </button>
        <button
          className="btn btn-ghost !py-1"
          onClick={() => {
            copyText(a.expr);
            toast.success("Expression copied");
          }}
        >
          <Clipboard size={12} /> Copy
        </button>
      </div>
    </div>
  );
}

function ForgeRun({ job, onOpen }: { job: JobSnapshot; onOpen: (expr: string, s?: Settings) => void }) {
  const live = useJobs((s) => s.jobs[job.id]);
  const j = { ...job, ...(live || {}) };
  const p = j.progress || {};
  const isLive = LIVE.has(j.status);
  const result = p.forge as ForgeResult | undefined;
  const stages = (p.stages || {}) as Record<string, ForgeStage>;
  const board = (p.leaderboard || []) as ForgeRow[];
  const frac = p.total ? (p.done || 0) / p.total : 0;
  const idea = String(j.config?.idea ?? "");
  return (
    <div className="flex flex-col gap-3">
      <Card
        title={
          <span className="flex items-center gap-2">
            Forge #{j.id} · {String(j.config?.effort ?? "standard")}
            <StatusBadge status={j.status === "done" ? "PASS" : j.status === "error" ? "FAIL" : isLive ? "PENDING" : "WARNING"} label={j.status} compact />
          </span>
        }
        actions={
          isLive && (
            <button className="btn btn-danger" onClick={() => api.jobAction(j.id, "cancel")}>
              <Square size={12} /> Stop
            </button>
          )
        }
      >
        <blockquote className="mb-3 border-l-2 border-[var(--accent)] pl-2.5 text-[12.5px] italic text-ink2">{idea}</blockquote>
        <StageStepper stages={stages} current={p.stage} live={isLive} />
        {isLive && (
          <div className="mt-2">
            <div className="mb-1 flex justify-between text-[11.5px]">
              <span className="text-ink2">{p.phase}</span>
              <span className="tnum text-muted">{p.total ? `${p.done ?? 0} / ${p.total}` : ""}</span>
            </div>
            <Progress value={frac} />
          </div>
        )}
        <div className="mt-2 grid grid-cols-2 gap-x-6 md:grid-cols-4">
          <Kv k="Simulated" v={j.stats?.evaluated ?? 0} />
          <Kv k="Elapsed" v={`${Math.round(j.stats?.elapsed_s ?? result?.seconds ?? 0)}s`} />
          <Kv k="Saved" v={j.stats?.saved ?? 0} />
          <Kv k="Errors" v={j.stats?.errors ?? 0} />
        </div>
        {j.error && <div className="mt-2 text-[12px] text-[var(--bad-text)]">{j.error}</div>}
      </Card>

      {result?.champion ? (
        <ChampionCard a={result.champion} result={result} onOpen={onOpen} />
      ) : result && !result.champion ? (
        <Card>
          <Empty title="No champion this time">{result.message}</Empty>
        </Card>
      ) : null}

      {result && (result.faithful || result.runners.length > 0 || result.brain_only.length > 0 || result.simple || result.complex) && (
        <Card title="More from this forge">
          <div className="flex flex-col divide-y divide-[var(--hairline)]">
            {result.simple && result.champion && result.simple.expr !== result.champion.expr && (
              <AlphaRowCard title="Best simple (one-line) alpha" a={result.simple} note="The strongest single-expression version." onOpen={onOpen} />
            )}
            {result.complex && result.champion && result.complex.expr !== result.champion.expr && (
              <AlphaRowCard title="Best complex (multi-statement) alpha" a={result.complex} note={result.complex.lineage[result.complex.lineage.length - 1] ?? "A composition of the leading signals."} onOpen={onOpen} />
            )}
            {result.faithful && (
              <AlphaRowCard title="Most faithful variant" a={result.faithful} note="Uses everything your idea names, in the stated direction; it scored below the champion." onOpen={onOpen} />
            )}
            {result.runners.map((r, i) => (
              <AlphaRowCard key={i} title={`Runner-up ${i + 1}`} a={r} note={`Decorrelated from the champion (PnL correlation < 0.7) · ${r.lineage[r.lineage.length - 1] ?? ""}`} onOpen={onOpen} />
            ))}
            {result.brain_only.length > 0 && (
              <div className="flex flex-col gap-1 py-2">
                <div className="flex items-center gap-1.5 text-[12px] font-semibold">
                  <CloudUpload size={13} /> BRAIN-only variants (saved unscored; simulate them on BRAIN)
                </div>
                {result.brain_only.map((b) => (
                  <button key={b.id} className="mono truncate text-left text-[11.5px] text-ink2 hover:text-ink" title="Open in Studio" onClick={() => onOpen(b.expr, b.settings)}>
                    {b.expr}
                  </button>
                ))}
              </div>
            )}
          </div>
        </Card>
      )}

      <Card title={isLive ? "Live leaderboard (in-sample)" : "Final leaderboard (in-sample)"} bodyClass="px-0 pb-1">
        <Leaderboard rows={board} onOpen={(r) => onOpen(r.expr, r.settings)} />
        {result && (
          <div className="px-3 pt-1.5 text-[11px] text-muted">
            {result.evaluated} simulations, {result.pool} distinct candidates in {Math.round(result.seconds)}s. Selection used the in-sample period only; the OS holdout is reported, never optimized.
          </div>
        )}
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ page */

export default function Forge() {
  const [params, setParams] = useSearchParams();
  const [text, setText] = useState<string>(() => params.get("idea") || loadDraft());
  const [effort, setEffort] = useState<string>("standard");
  const [settings, setSettings] = useState<Settings>({ ...DEFAULT_SETTINGS });
  const [families, setFamilies] = useState<string[] | null>(null);
  const [horizon, setHorizon] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(params.get("job") ? Number(params.get("job")) : null);
  const [starting, setStarting] = useState(false);
  const setStudio = useUI((s) => s.setStudio);
  const nav = useNavigate();
  const { data: status } = useStatus();
  const demo = status?.data?.source === "demo";
  const [allowDemo, setAllowDemo] = useState(false);
  const [reading, setReading] = useState(false);

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setReading(true);
    try {
      if (TEXT_EXT.test(f.name) || f.type.startsWith("text/")) {
        setText(await f.text());
      } else {
        const r = await api.forgeExtract(f.name, await fileToBase64(f));
        setText(r.text);
        toast.success(`Read ${f.name}: ${r.chars.toLocaleString()} characters${r.key_sentences.length ? `, ${r.key_sentences.length} key sentences` : ""}`);
      }
    } catch (e) {
      toast.error(String((e as Error).message));
    } finally {
      setReading(false);
    }
  };
  const qc = useQueryClient();
  const liveJobs = useJobs((s) => s.jobs);

  const dText = useDebounced(text, 350);
  const { data: spec, isFetching } = useQuery({
    queryKey: ["forge-interpret", dText, families, horizon],
    queryFn: () => api.forgeInterpret(dText, { families: families ?? undefined, horizon: horizon ?? undefined }),
    enabled: dText.trim().length > 0,
    placeholderData: (prev) => prev,
  });

  useEffect(() => saveDraft(text), [text]);
  // a new idea resets manual overrides
  useEffect(() => {
    setFamilies(null);
    setHorizon(null);
  }, [dText]);

  const { data: jobsData } = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 10_000 });
  const forges = useMemo(() => {
    const rows = (jobsData?.jobs ?? []).filter((j) => j.kind === "forge").map((j) => ({ ...j, ...(liveJobs[j.id] || {}) }));
    for (const lj of Object.values(liveJobs)) if (lj.kind === "forge" && !rows.some((r) => r.id === lj.id)) rows.unshift(lj);
    return rows.sort((a, b) => b.id - a.id);
  }, [jobsData, liveJobs]);
  const sel = forges.find((j) => j.id === selected) || forges[0];

  const openStudio = (expr: string, s?: Settings) => {
    setStudio(expr, s ?? settings);
    nav("/studio");
  };

  const start = async () => {
    if (!text.trim() || starting) return;
    setStarting(true);
    try {
      const r = await api.startJob("forge", { idea: text, effort, settings, families: families ?? undefined, horizon: horizon ?? undefined, allow_demo: allowDemo || undefined });
      toast.success(`Forging your idea (job #${r.id})`);
      setSelected(r.id);
      setParams({ job: String(r.id) });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    } catch (e) {
      toast.error(String((e as Error).message));
    } finally {
      setStarting(false);
    }
  };
  useHotkey("Enter", () => start());

  return (
    <div className="grid grid-cols-1 gap-3 p-3 xl:grid-cols-12">
      <div className="flex flex-col gap-3 xl:col-span-5">
        <Card
          title={
            <span className="flex items-center gap-1.5">
              <Wand2 size={13} /> Your alpha idea
            </span>
          }
        >
          <div className="flex flex-col gap-2.5">
            <textarea
              className="input min-h-[112px] resize-y text-[13px] leading-relaxed"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="e.g. Stocks that drop sharply on heavy volume tend to bounce back within a week. Or paste notes, a paper, a formula (101-Alphas notation too), pandas code, JSON, a list of ideas or Fast Expressions."
              aria-label="Alpha idea"
            />
            <div className="flex items-center gap-2">
              <label className="btn cursor-pointer !py-1">
                {reading ? <Spinner size={12} /> : <FileUp size={12} />} Open a file
                <input type="file" className="hidden" accept=".pdf,.docx,.txt,.md,.html,.htm,.rtf,.ipynb,.py,.json,.yaml,.yml,.csv,.tex" onChange={(e) => onFile(e.target.files?.[0])} />
              </label>
              <span className="text-[10.5px] text-muted">PDF, Word, HTML, notebooks, code or text. Long documents are reduced to the sentences that state the idea.</span>
            </div>
            <div className="flex flex-wrap gap-1">
              <span className="text-[10.5px] text-muted">Try:</span>
              {EXAMPLES.map((e) => (
                <button key={e} className="chip max-w-full cursor-pointer truncate hover:!text-ink" title={e} onClick={() => setText(e)}>
                  {e.length > 52 ? `${e.slice(0, 50)}…` : e}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-end gap-3 border-t border-line pt-2.5">
              <div className="flex flex-col gap-1">
                <label className="lbl">Effort</label>
                <div className="flex gap-1">
                  {EFFORTS.map((x) => (
                    <Tip key={x.id} content={x.hint}>
                      <button className={clsx("btn !py-1", effort === x.id && "!border-[var(--accent)] !text-[var(--accent)]")} onClick={() => setEffort(x.id)}>
                        {x.label}
                      </button>
                    </Tip>
                  ))}
                </div>
              </div>
              {demo && <Toggle checked={allowDemo} onChange={setAllowDemo} label="Run on demo data anyway" />}
              <button className="btn btn-primary ml-auto" onClick={start} disabled={!text.trim() || starting}>
                {starting ? <Spinner size={13} /> : <Sparkles size={13} />} Forge alpha <span className="opacity-70">Ctrl+Enter</span>
              </button>
            </div>
            <details className="text-[12px]">
              <summary className="cursor-pointer text-muted">BRAIN settings for the search</summary>
              <div className="pt-2">
                <SettingsBar settings={settings} onChange={(p) => setSettings((s) => ({ ...s, ...p }))} compact />
              </div>
            </details>
          </div>
        </Card>

        <Interpretation
          spec={dText.trim() ? spec : undefined}
          loading={isFetching}
          families={families}
          onFamilies={setFamilies}
          horizon={horizon}
          onHorizon={setHorizon}
          onOpen={(e) => openStudio(e)}
        />

        <Card title="Past forges" bodyClass="px-0 pb-1">
          {forges.length === 0 ? (
            <Empty title="No forges yet" />
          ) : (
            <div className="max-h-[260px] overflow-auto">
              {forges.map((j) => {
                const champ = (j.progress?.forge as ForgeResult | undefined)?.champion;
                return (
                  <button
                    key={j.id}
                    onClick={() => {
                      setSelected(j.id);
                      setParams({ job: String(j.id) });
                    }}
                    className={clsx("flex w-full items-center gap-2 border-b border-line px-3 py-1.5 text-left text-[12px] hover:bg-[var(--surface-2)]", sel?.id === j.id && "bg-[var(--surface-2)]")}
                  >
                    <span className="w-9 shrink-0 text-muted">#{j.id}</span>
                    <span className="min-w-0 flex-1 truncate text-ink2">{String(j.config?.idea ?? "")}</span>
                    {LIVE.has(j.status) ? (
                      <Loader2 size={12} className="animate-spin text-muted" />
                    ) : champ ? (
                      <span className="tnum shrink-0 text-[11px]">
                        <StatusBadge status={champ.status === "PASS" ? "PASS" : "FAIL"} label="" compact /> S {fmt.num(champ.sharpe)}
                      </span>
                    ) : (
                      <span className="shrink-0 text-[11px] text-muted">{j.status}</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </Card>
      </div>
      <div className="xl:col-span-7">
        {sel ? (
          <ForgeRun job={sel} onOpen={openStudio} />
        ) : (
          <Card>
            <Empty title="Paste an idea and press Forge alpha">
              The forge reads your idea, drafts dozens of Fast Expressions that implement it, screens them on the local simulator, blends and repairs the best, evolves them with genetic programming, composes complex multi-statement versions, and hands back graded champions (a simple and a complex one) with BRAIN-ready settings. Nothing is sent to BRAIN unless you press Run on BRAIN.
            </Empty>
          </Card>
        )}
      </div>
    </div>
  );
}
