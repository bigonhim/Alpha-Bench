import { useQuery, useQueryClient } from "@tanstack/react-query";
import { BookMarked, CloudUpload, Dna, LayoutTemplate, ListChecks, Rocket, Shuffle, SlidersHorizontal, Sparkles, Wand2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { JobPanel, jobStatusBadge } from "../components/JobPanel";
import { RE_DEFAULTS, ReengineerOptions } from "../components/Reengineer";
import { SettingsBar } from "../components/SettingsBar";
import { Card, Empty, Field, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { clsx, fmt } from "../lib/format";
import { DEFAULT_SETTINGS, useJobs } from "../lib/store";
import type { Settings } from "../lib/types";

const METHODS = [
  { id: "automine", title: "Auto-Mine", icon: Sparkles, desc: "Bandit-steered loop: templates → screen → Doctor fixes → GP refinement. Saves passing, mutually decorrelated candidates." },
  { id: "gp", title: "Genetic programming", icon: Dna, desc: "NSGA-II over expression trees (fitness × novelty × simplicity) with islands and successive halving." },
  { id: "templates", title: "Template grid", icon: LayoutTemplate, desc: "Expand the idea-tagged template library over fields, windows, groups and settings." },
  { id: "grammar", title: "Random grammar", icon: Shuffle, desc: "Typed, unit-aware random expressions for wide exploration." },
  { id: "alpha101", title: "101 Alphas", icon: BookMarked, desc: "Kakushadze's price-volume formulas as seeds (verify translations against the paper)." },
  { id: "batch", title: "Batch list", icon: ListChecks, desc: "Paste many expressions; every one gets the full check suite and is saved." },
  { id: "settings_opt", title: "Settings optimizer", icon: SlidersHorizontal, desc: "Robust decay × neutralization × truncation search for chosen library alphas." },
  { id: "brain_only", title: "BRAIN-only ideas", icon: CloudUpload, desc: "Candidates on data not available locally (news, options, social, imported fields) to test on BRAIN." },
  { id: "reengineer", title: "Re-engineer", icon: Wand2, desc: "Diagnose a weak alpha and rebuild it stage by stage into a strong one, with a step-by-step recipe and a holdout verdict." },
] as const;

const IDEAS = ["reversion", "momentum", "seasonality", "value", "quality", "growth", "accruals", "investment", "leverage", "liquidity", "pv_divergence", "volatility"];

function Num({ label, value, onChange, step = 1, min, max, hint }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <input className="input w-24 tnum" type="number" step={step} min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

function FamilyPicker({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  return (
    <Field label="Idea families (none = all)">
      <div className="flex flex-wrap gap-1">
        {IDEAS.map((f) => {
          const on = value.includes(f);
          return (
            <button key={f} type="button" className={clsx("chip cursor-pointer", on && "!bg-[var(--accent)] !text-white")} onClick={() => onChange(on ? value.filter((x) => x !== f) : [...value, f])}>
              {f}
            </button>
          );
        })}
      </div>
    </Field>
  );
}

export default function Miner() {
  const [params, setParams] = useSearchParams();
  const [method, setMethod] = useState<string>(params.get("method") || "automine");
  const [settings, setSettings] = useState<Settings>({ ...DEFAULT_SETTINGS });
  const [cfg, setCfg] = useState<Record<string, any>>({});
  const [selected, setSelected] = useState<number | null>(params.get("job") ? Number(params.get("job")) : null);
  const qc = useQueryClient();
  const initialParams = useRef(params);
  const liveJobs = useJobs((s) => s.jobs);
  const { data } = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 10_000 });

  useEffect(() => {
    const params = initialParams.current;
    const defaults: Record<string, Record<string, any>> = {
      automine: { time_limit_min: 20, target_candidates: 20, save_min_sharpe: 1.2, families: [], include_brain_only: false, round_size: 24, gp_every: 3 },
      gp: { population: 60, generations: 15, islands: 2, max_depth: 6, time_limit_min: 30, min_sharpe: 1.25, min_fitness: 1.0, halving: true, seed_from_library: true, library_seeds: 10, pv_weight: 0.6, seed_exprs: "" },
      templates: { families: params.get("family") ? [params.get("family")] : [], per_template: 12, settings_per_expr: 2, max_candidates: 400, save_min_sharpe: 1.0 },
      grammar: { n: 300, max_depth: 4, save_min_sharpe: 1.0, randomize_settings: true, pv_weight: 0.6 },
      alpha101: { save_min_sharpe: 0.5 },
      batch: { exprs: "" },
      settings_opt: { alpha_ids: params.get("ids") || "" },
      brain_only: { per_template: 6, n_field_candidates: 40 },
      reengineer: { ...RE_DEFAULTS, expr: params.get("expr") || "" },
    };
    setCfg(defaults[method] || {});
  }, [method]);

  const jobs = useMemo(() => {
    const rows = (data?.jobs ?? []).map((j) => ({ ...j, ...(liveJobs[j.id] || {}) }));
    for (const lj of Object.values(liveJobs)) if (!rows.some((r) => r.id === lj.id)) rows.unshift(lj);
    return rows.sort((a, b) => b.id - a.id);
  }, [data, liveJobs]);
  const sel = jobs.find((j) => j.id === selected) || jobs[0];

  const start = async () => {
    const c: Record<string, any> = { ...cfg, settings };
    if ("pv_weight" in c) {
      c.category_weights = { pv: c.pv_weight, fundamental: Math.max(0, 1 - c.pv_weight) };
      delete c.pv_weight;
    }
    if (method === "batch") c.exprs = String(c.exprs || "").split("\n");
    if (method === "gp") c.seed_exprs = String(c.seed_exprs || "").split("\n").filter((x: string) => x.trim());
    if (method === "settings_opt") c.alpha_ids = String(c.alpha_ids || "").split(/[\s,]+/).filter(Boolean).map(Number);
    if (method === "reengineer") {
      c.expr = String(c.expr || "").trim();
      if (!c.expr) {
        toast.error("Enter the alpha expression to re-engineer");
        return;
      }
    }
    try {
      const r = await api.startJob(method, c);
      toast.success(`Started job #${r.id}`);
      setSelected(r.id);
      setParams({ job: String(r.id) });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    } catch (e) {
      toast.error(String((e as Error).message));
    }
  };

  const set = (k: string) => (v: any) => setCfg((c) => ({ ...c, [k]: v }));
  const m = METHODS.find((x) => x.id === method)!;

  return (
    <div className="grid grid-cols-1 gap-3 p-3 xl:grid-cols-12">
      <div className="flex flex-col gap-3 xl:col-span-5">
        <div className="grid grid-cols-2 gap-2">
          {METHODS.map((x) => (
            <button
              key={x.id}
              onClick={() => setMethod(x.id)}
              className={clsx("card flex items-start gap-2 p-2.5 text-left transition-colors", method === x.id ? "!border-[var(--accent)]" : "hover:bg-[var(--surface-2)]")}
            >
              <x.icon size={16} className={method === x.id ? "mt-0.5 text-[var(--accent)]" : "mt-0.5 text-muted"} />
              <div className="min-w-0">
                <div className="text-[12.5px] font-semibold">{x.title}</div>
                <div className="line-clamp-2 text-[11px] text-muted">{x.desc}</div>
              </div>
            </button>
          ))}
        </div>
        <Card title={`${m.title} · configuration`}>
          <div className="flex flex-col gap-3">
            {method === "automine" && (
              <>
                <div className="flex flex-wrap gap-3">
                  <Num label="Time limit (min)" value={cfg.time_limit_min ?? 20} onChange={set("time_limit_min")} min={1} />
                  <Num label="Target passing" value={cfg.target_candidates ?? 20} onChange={set("target_candidates")} min={1} />
                  <Num label="Save if Sharpe ≥" value={cfg.save_min_sharpe ?? 1.2} onChange={set("save_min_sharpe")} step={0.05} />
                  <Num label="Round size" value={cfg.round_size ?? 24} onChange={set("round_size")} min={4} />
                  <Num label="GP every N rounds" value={cfg.gp_every ?? 3} onChange={set("gp_every")} min={1} />
                </div>
                <FamilyPicker value={cfg.families ?? []} onChange={set("families")} />
                <Toggle checked={!!cfg.include_brain_only} onChange={set("include_brain_only")} label="Also generate BRAIN-only ideas at the end" />
              </>
            )}
            {method === "gp" && (
              <>
                <div className="flex flex-wrap gap-3">
                  <Num label="Population" value={cfg.population ?? 60} onChange={set("population")} min={8} />
                  <Num label="Generations" value={cfg.generations ?? 15} onChange={set("generations")} min={1} />
                  <Num label="Islands" value={cfg.islands ?? 2} onChange={set("islands")} min={1} max={8} />
                  <Num label="Max depth" value={cfg.max_depth ?? 6} onChange={set("max_depth")} min={3} max={9} />
                  <Num label="Time limit (min)" value={cfg.time_limit_min ?? 30} onChange={set("time_limit_min")} min={1} />
                  <Num label="Hall-of-fame Sharpe ≥" value={cfg.min_sharpe ?? 1.25} onChange={set("min_sharpe")} step={0.05} />
                  <Num label="Hall-of-fame fitness ≥" value={cfg.min_fitness ?? 1} onChange={set("min_fitness")} step={0.05} />
                  <Num label="Price-volume share" value={cfg.pv_weight ?? 0.6} onChange={set("pv_weight")} step={0.1} min={0} max={1} hint="Share of price-volume vs fundamental leaves" />
                </div>
                <div className="flex flex-wrap gap-4">
                  <Toggle checked={!!cfg.halving} onChange={set("halving")} label="Successive halving (screen on 3y, promote top 40%)" />
                  <Toggle checked={!!cfg.seed_from_library} onChange={set("seed_from_library")} label="Seed from best library alphas" />
                </div>
                <Field label="Extra seed expressions (one per line)">
                  <textarea className="input mono h-20" value={cfg.seed_exprs ?? ""} onChange={(e) => set("seed_exprs")(e.target.value)} />
                </Field>
              </>
            )}
            {method === "templates" && (
              <>
                <div className="flex flex-wrap gap-3">
                  <Num label="Expansions / template" value={cfg.per_template ?? 12} onChange={set("per_template")} min={1} />
                  <Num label="Settings / expression" value={cfg.settings_per_expr ?? 2} onChange={set("settings_per_expr")} min={1} />
                  <Num label="Max candidates" value={cfg.max_candidates ?? 400} onChange={set("max_candidates")} min={10} />
                  <Num label="Save if Sharpe ≥" value={cfg.save_min_sharpe ?? 1} onChange={set("save_min_sharpe")} step={0.05} />
                </div>
                <FamilyPicker value={cfg.families ?? []} onChange={set("families")} />
              </>
            )}
            {method === "grammar" && (
              <>
                <div className="flex flex-wrap gap-3">
                  <Num label="Expressions" value={cfg.n ?? 300} onChange={set("n")} min={10} />
                  <Num label="Max depth" value={cfg.max_depth ?? 4} onChange={set("max_depth")} min={2} max={7} />
                  <Num label="Save if Sharpe ≥" value={cfg.save_min_sharpe ?? 1} onChange={set("save_min_sharpe")} step={0.05} />
                  <Num label="Price-volume share" value={cfg.pv_weight ?? 0.6} onChange={set("pv_weight")} step={0.1} min={0} max={1} />
                </div>
                <Toggle checked={!!cfg.randomize_settings} onChange={set("randomize_settings")} label="Randomize decay and neutralization" />
              </>
            )}
            {method === "alpha101" && <Num label="Save if Sharpe ≥" value={cfg.save_min_sharpe ?? 0.5} onChange={set("save_min_sharpe")} step={0.05} />}
            {method === "batch" && (
              <Field label="Expressions (one per line; # comments ignored)">
                <textarea className="input mono h-48" value={cfg.exprs ?? ""} onChange={(e) => set("exprs")(e.target.value)} placeholder={"rank(-ts_delta(close, 5))\ngroup_rank(ts_backfill(sales, 120) / cap, subindustry)"} />
              </Field>
            )}
            {method === "settings_opt" && (
              <Field label="Alpha ids (comma or space separated)">
                <input className="input mono" value={cfg.alpha_ids ?? ""} onChange={(e) => set("alpha_ids")(e.target.value)} placeholder="12, 15, 40" />
              </Field>
            )}
            {method === "reengineer" && (
              <>
                <Field label="Alpha to re-engineer (simulated with the base settings below)">
                  <textarea className="input mono h-20" value={cfg.expr ?? ""} onChange={(e) => set("expr")(e.target.value)} placeholder="ts_delta(close, 5)" />
                </Field>
                <ReengineerOptions value={{ ...RE_DEFAULTS, ...cfg }} onChange={(v) => setCfg((c) => ({ ...c, ...v }))} />
              </>
            )}
            {method === "brain_only" && (
              <div className="flex flex-wrap gap-3">
                <Num label="Expansions / template" value={cfg.per_template ?? 6} onChange={set("per_template")} min={1} />
                <Num label="Imported-field ideas" value={cfg.n_field_candidates ?? 40} onChange={set("n_field_candidates")} min={0} />
              </div>
            )}
            <div className="border-t border-line pt-2">
              <div className="lbl mb-1.5">Base BRAIN settings</div>
              <SettingsBar settings={settings} onChange={(p) => setSettings((s) => ({ ...s, ...p }))} compact />
            </div>
            <div className="flex justify-end">
              <button className="btn btn-primary" onClick={start}>
                <Rocket size={13} /> Start {m.title}
              </button>
            </div>
          </div>
        </Card>
        <Card title="Jobs" bodyClass="px-0 pb-1">
          {jobs.length === 0 ? (
            <Empty title="No jobs yet" />
          ) : (
            <div className="max-h-[320px] overflow-auto">
              {jobs.map((j) => (
                <button
                  key={j.id}
                  onClick={() => setSelected(j.id)}
                  className={clsx("flex w-full items-center gap-2 border-b border-line px-3 py-1.5 text-left text-[12px] hover:bg-[var(--surface-2)]", sel?.id === j.id && "bg-[var(--surface-2)]")}
                >
                  <span className="w-10 text-muted">#{j.id}</span>
                  <span className="w-28 font-medium">{j.kind}</span>
                  {jobStatusBadge(j.status)}
                  <span className="tnum ml-auto text-muted">
                    {j.stats?.saved ?? 0} saved · {j.stats?.passed ?? 0} pass · {fmt.ago(j.created_at)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </Card>
      </div>
      <div className="xl:col-span-7">{sel ? <JobPanel job={sel} /> : <Card><Empty title="Start a job to see live progress" /></Card>}</div>
    </div>
  );
}
