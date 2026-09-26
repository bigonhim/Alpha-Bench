import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Save } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Card, Field, Select, Spinner } from "../components/ui";
import { api } from "../lib/api";
import { useUI } from "../lib/store";

function NumIn({ label, value, onChange, step = 0.01, hint }: { label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <input className="input w-28 tnum" type="number" step={step} value={value ?? ""} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

export default function SettingsPage() {
  const { data: checks } = useQuery({ queryKey: ["checks"], queryFn: api.checks });
  const { data: settings } = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const [c, setC] = useState<any>(null);
  const [s, setS] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const { theme, setTheme } = useUI();
  const qc = useQueryClient();
  useEffect(() => setC(checks ? structuredClone(checks) : null), [checks]);
  useEffect(() => setS(settings ? { ...settings } : null), [settings]);
  if (!c || !s) return <div className="p-4"><Spinner /></div>;

  const upd = (path: string[], v: number) =>
    setC((prev: any) => {
      const n = structuredClone(prev);
      let o = n;
      for (const k of path.slice(0, -1)) o = o[k];
      o[path[path.length - 1]] = v;
      return n;
    });

  const saveChecks = async () => {
    setBusy(true);
    try {
      await api.saveChecks(c);
      toast.success("Check thresholds saved");
      qc.invalidateQueries();
    } finally {
      setBusy(false);
    }
  };
  const saveSettings = async () => {
    setBusy(true);
    try {
      await api.saveSettings(s);
      toast.success("Settings saved (dataset periods reloaded)");
      qc.invalidateQueries();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid grid-cols-1 gap-3 p-3 xl:grid-cols-2">
      <Card title="Submission check thresholds" actions={<button className="btn btn-primary" onClick={saveChecks} disabled={busy}><Save size={13} /> Save</button>}>
        <div className="mb-2 text-[12px] text-muted">Defaults mirror BRAIN's published delay-1 rules; values marked "verify" in docs/research-notes.md could not be confirmed. Adjust to match what you see on BRAIN.</div>
        <div className="flex flex-wrap gap-3">
          <NumIn label="D1 Sharpe ≥" value={c.delay1.sharpe_min} onChange={(v) => upd(["delay1", "sharpe_min"], v)} />
          <NumIn label="D1 Fitness ≥" value={c.delay1.fitness_min} onChange={(v) => upd(["delay1", "fitness_min"], v)} />
          <NumIn label="D0 Sharpe ≥" value={c.delay0.sharpe_min} onChange={(v) => upd(["delay0", "sharpe_min"], v)} />
          <NumIn label="D0 Fitness ≥" value={c.delay0.fitness_min} onChange={(v) => upd(["delay0", "fitness_min"], v)} />
          <NumIn label="Turnover min" value={c.turnover_min} onChange={(v) => upd(["turnover_min"], v)} />
          <NumIn label="Turnover max" value={c.turnover_max} onChange={(v) => upd(["turnover_max"], v)} />
          <NumIn label="Max weight" value={c.max_weight} onChange={(v) => upd(["max_weight"], v)} />
          <NumIn label="Sub-universe factor" value={c.sub_universe_factor} onChange={(v) => upd(["sub_universe_factor"], v)} />
          <NumIn label="Self-corr max" value={c.self_corr_max} onChange={(v) => upd(["self_corr_max"], v)} />
          <NumIn label="Sharpe improvement" value={c.self_corr_sharpe_improvement} onChange={(v) => upd(["self_corr_sharpe_improvement"], v)} />
          <NumIn label="Self-corr years" value={c.self_corr_years} step={1} onChange={(v) => upd(["self_corr_years"], v)} />
          <NumIn label="2y Sharpe (soft)" value={c.two_year_sharpe_min} onChange={(v) => upd(["two_year_sharpe_min"], v)} />
        </div>
        <div className="lbl mb-1 mt-3">Local robustness gates (warnings)</div>
        <div className="flex flex-wrap gap-3">
          <NumIn label="OS/IS ratio ≥" value={c.local_gates.os_is_ratio_min} onChange={(v) => upd(["local_gates", "os_is_ratio_min"], v)} />
          <NumIn label="Positive years ≥" value={c.local_gates.positive_years_min} onChange={(v) => upd(["local_gates", "positive_years_min"], v)} />
          <NumIn label="Stability ≥" value={c.local_gates.stability_min} onChange={(v) => upd(["local_gates", "stability_min"], v)} />
          <NumIn label="Drawdown ≤" value={c.local_gates.drawdown_max} onChange={(v) => upd(["local_gates", "drawdown_max"], v)} />
        </div>
      </Card>
      <div className="flex flex-col gap-3">
        <Card title="Simulation" actions={<button className="btn btn-primary" onClick={saveSettings} disabled={busy}><Save size={13} /> Save</button>}>
          <div className="flex flex-wrap gap-3">
            <NumIn label="IS starts after (years)" value={s.is_years_warmup} step={1} onChange={(v) => setS({ ...s, is_years_warmup: v })} hint="Warm-up history before the in-sample period" />
            <NumIn label="OS holdout (years)" value={s.os_years} step={0.5} onChange={(v) => setS({ ...s, os_years: v })} />
            <NumIn label="BRAIN window (years)" value={s.brain_window_years} step={1} onChange={(v) => setS({ ...s, brain_window_years: v })} />
            <NumIn label="Book size ($)" value={s.booksize} step={1000000} onChange={(v) => setS({ ...s, booksize: v })} />
            <Field label="Returns basis">
              <Select value={s.returns_basis} onChange={(v) => setS({ ...s, returns_basis: v })} options={[{ value: "half_book", label: "PnL / (book/2) (verify)" }, { value: "full_book", label: "PnL / book" }]} />
            </Field>
            <Field label="History start">
              <input className="input w-32" value={s.history_start} onChange={(e) => setS({ ...s, history_start: e.target.value })} />
            </Field>
          </div>
        </Card>
        <Card title="Performance">
          <div className="flex flex-wrap gap-3">
            <NumIn label="Miner worker processes" value={s.workers} step={1} onChange={(v) => setS({ ...s, workers: v })} hint="0 = evaluate in the server process" />
            <NumIn label="Server cache (MB)" value={s.api_cache_mb} step={50} onChange={(v) => setS({ ...s, api_cache_mb: v })} />
            <NumIn label="Worker cache (MB)" value={s.worker_cache_mb} step={50} onChange={(v) => setS({ ...s, worker_cache_mb: v })} />
          </div>
          <div className="mt-2 text-[11.5px] text-muted">On a 2-core / 8 GB laptop, 2 workers with 300 MB caches is a good balance. Save with the Simulation card's button.</div>
        </Card>
        <Card title="Appearance">
          <div className="flex items-center gap-2 text-[12px]">
            <span className="text-muted">Theme</span>
            <Select value={theme} onChange={(v) => setTheme(v as "dark" | "light")} options={["dark", "light"]} />
          </div>
        </Card>
      </div>
    </div>
  );
}
