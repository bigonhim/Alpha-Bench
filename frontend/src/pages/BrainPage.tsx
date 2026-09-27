import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CloudDownload, ExternalLink, KeyRound, Link2, LogOut, RefreshCw, Rocket, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { JobPanel, jobStatusBadge } from "../components/JobPanel";
import { Card, Empty, Field, Kv, Select, Spinner, StatusBadge, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { clsx, fmt } from "../lib/format";
import { useJobs } from "../lib/store";

const BRAIN_KINDS = new Set(["brain_sim", "brain_fields", "brain_sync", "brain_mine"]);

function Num({ label, value, onChange, min, max, hint }: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <input className="input w-24 tnum" type="number" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

function HowTo() {
  return (
    <Card title="How the BRAIN connection works">
      <ol className="flex list-decimal flex-col gap-1.5 pl-4 text-[12px] text-ink2">
        <li>
          There is <b>no separate API key</b>. BRAIN's API (<span className="mono">api.worldquantbrain.com</span>) accepts the same email and password you use on{" "}
          <a className="text-[var(--accent)]" href="https://platform.worldquantbrain.com" target="_blank" rel="noreferrer">
            platform.worldquantbrain.com
          </a>
          . Sign in below.
        </li>
        <li>
          If your account uses biometric sign-in, BRAIN answers with a verification link. Open it, complete the check in the browser, then press <b>Complete sign-in</b>.
        </li>
        <li>
          The session cookie is kept in <span className="mono">%LOCALAPPDATA%\AlphaFoundry</span>, outside the OneDrive folder. With "remember me" the password goes into Windows Credential Manager, never into
          the project, the database or the logs. You can instead set <span className="mono">BRAIN_EMAIL</span> and <span className="mono">BRAIN_PASSWORD</span> environment variables.
        </li>
        <li>
          BRAIN does not let anyone download its raw datasets. The connection is used to <b>simulate on BRAIN</b> (real Sharpe, fitness, turnover and every check), to <b>pull your alphas and PnL</b> (self-correlation), and to{" "}
          <b>import the data-field catalog</b> (so the generator uses real field ids, preferring less-crowded ones).
        </li>
        <li>
          Simulations count against your BRAIN limits. The daily budget below caps what this tool may spend. Nothing is ever <b>submitted</b>: submission stays a manual click on the BRAIN website.
        </li>
      </ol>
    </Card>
  );
}

export default function BrainPage() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const { data: st, isFetching, refetch } = useQuery({ queryKey: ["brain-status"], queryFn: api.brain.status, refetchInterval: 60_000 });
  const { data: ds } = useQuery({ queryKey: ["brain-datasets"], queryFn: api.brain.datasets, enabled: !!st?.connected });
  const { data: jobsData } = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 10_000 });
  const liveJobs = useJobs((s) => s.jobs);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [busy, setBusy] = useState(false);
  const [pick, setPick] = useState<Set<string>>(new Set());
  const [mine, setMine] = useState({ budget: 40, time_limit_min: 60, lib: true, fields: true, min_grade: "B", refine_rounds: 2 });
  const [selected, setSelected] = useState<number | null>(params.get("job") ? Number(params.get("job")) : null);

  const jobs = useMemo(() => {
    const rows = (jobsData?.jobs ?? []).filter((j) => BRAIN_KINDS.has(j.kind)).map((j) => ({ ...j, ...(liveJobs[j.id] || {}) }));
    for (const lj of Object.values(liveJobs)) if (BRAIN_KINDS.has(lj.kind) && !rows.some((r) => r.id === lj.id)) rows.unshift(lj);
    return rows.sort((a, b) => b.id - a.id);
  }, [jobsData, liveJobs]);
  const sel = jobs.find((j) => j.id === selected) || jobs[0];

  const act = async (fn: () => Promise<any>, ok?: string) => {
    setBusy(true);
    try {
      const r = await fn();
      if (ok) toast.success(ok);
      qc.invalidateQueries({ queryKey: ["brain-status"] });
      if (r?.id) {
        setSelected(r.id);
        setParams({ job: String(r.id) });
        qc.invalidateQueries({ queryKey: ["jobs"] });
      }
      return r;
    } catch (e) {
      toast.error(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  const login = () =>
    act(async () => {
      const r = await api.brain.login(email, password, remember);
      setPassword("");
      if (r.persona_url) toast.info("Biometric check required: open the link, finish it, then press Complete sign-in.");
      else if (r.connected) toast.success("Connected to BRAIN");
      return r;
    });

  const saveSettings = (u: Record<string, unknown>) => act(() => api.brain.settings(u), "Saved");
  const datasets = (ds?.datasets ?? []).slice().sort((a: any, b: any) => (b.value_score ?? 0) - (a.value_score ?? 0));

  return (
    <div className="grid grid-cols-1 gap-3 p-3 xl:grid-cols-12">
      <div className="flex flex-col gap-3 xl:col-span-5">
        <Card
          title={
            <span className="flex items-center gap-1.5">
              <Link2 size={13} /> BRAIN connection
            </span>
          }
          actions={
            <button className="btn btn-ghost !p-1" onClick={() => refetch()} aria-label="Refresh status">
              {isFetching ? <Spinner /> : <RefreshCw size={13} />}
            </button>
          }
        >
          {st?.connected ? (
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <StatusBadge status="PASS" label="Connected" />
                <span className="text-[12px] text-muted">user {st.user_id}</span>
              </div>
              <Kv k="Session expires in" v={st.expiry_s != null ? `${Math.round(st.expiry_s / 60)} min` : "—"} />
              <Kv k="Multi-simulation" v={st.multi_allowed ? "allowed (10 alphas per request)" : "not in your permissions"} />
              <Kv k="Simulations today (this tool)" v={`${st.usage_today} of ${st.budget}`} />
              <Kv k="Saved sign-in" v={st.credentials_saved ? `yes (${st.saved_email ?? "env"})` : "no"} />
              <div className="mt-2 flex gap-2">
                <button className="btn" onClick={() => act(() => api.brain.logout(false), "Signed out")}>
                  <LogOut size={13} /> Sign out
                </button>
                {st.credentials_saved && (
                  <button className="btn btn-ghost" onClick={() => act(() => api.brain.logout(true), "Signed out and saved password removed")}>
                    Forget saved password
                  </button>
                )}
              </div>
            </div>
          ) : st?.persona_url ? (
            <div className="flex flex-col gap-2 text-[12px]">
              <div className="text-ink2">BRAIN asks for a biometric check on this sign-in.</div>
              <a className="btn btn-primary w-fit" href={st.persona_url} target="_blank" rel="noreferrer">
                <ShieldCheck size={13} /> Open the verification page <ExternalLink size={12} />
              </a>
              <button className="btn w-fit" disabled={busy} onClick={() => act(() => api.brain.completePersona(), "Connected to BRAIN")}>
                {busy ? <Spinner /> : null} Complete sign-in
              </button>
            </div>
          ) : (
            <form
              className="flex flex-col gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                login();
              }}
            >
              <Field label="BRAIN email">
                <input className="input" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} placeholder={st?.saved_email ?? "you@example.com"} />
              </Field>
              <Field label="BRAIN password">
                <input className="input" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />
              </Field>
              <Toggle
                checked={remember && !!st?.keyring_available}
                onChange={setRemember}
                label={st?.keyring_available ? "Remember in Windows Credential Manager (re-signs in when the session expires)" : "Remember me (unavailable: no OS keychain found)"}
              />
              <div className="flex items-center gap-2">
                <button className="btn btn-primary" type="submit" disabled={busy || (!password && !st?.credentials_saved)}>
                  {busy ? <Spinner /> : <KeyRound size={13} />} Sign in
                </button>
                {st?.credentials_saved && !password && <span className="text-[11px] text-muted">empty fields use the saved sign-in</span>}
              </div>
            </form>
          )}
        </Card>

        <HowTo />

        {st?.connected && (
          <Card title="Limits and defaults">
            <div className="flex flex-wrap items-end gap-3">
              <Num label="Parallel simulations" value={st.settings.concurrency} min={1} max={10} onChange={(v) => saveSettings({ concurrency: v })} hint="BRAIN's concurrent-simulation limit applies (3 for most accounts)" />
              <Num label="Daily budget" value={st.settings.daily_budget} min={0} onChange={(v) => saveSettings({ daily_budget: v })} hint="Most simulations this tool may run per day" />
              <Field label="Multi-simulation">
                <Select value={st.settings.multi} onChange={(v) => saveSettings({ multi: v })} options={[{ value: "auto", label: "Auto" }, { value: "on", label: "On" }, { value: "off", label: "Off" }]} />
              </Field>
              <Field label="Region">
                <Select value={st.settings.region} onChange={(v) => saveSettings({ region: v })} options={["USA", "GLB", "EUR", "ASI", "CHN", "JPN", "KOR", "TWN", "HKG", "AMR"]} />
              </Field>
              <Field label="Universe">
                <Select value={st.settings.universe} onChange={(v) => saveSettings({ universe: v })} options={["TOP3000", "TOP1000", "TOP500", "TOP200", "TOPSP500"]} />
              </Field>
              <Field label="Delay">
                <Select value={String(st.settings.delay)} onChange={(v) => saveSettings({ delay: Number(v) })} options={["1", "0"]} />
              </Field>
            </div>
            <div className="mt-2">
              <Toggle checked={st.settings.check_passing} onChange={(v) => saveSettings({ check_passing: v })} label="Run BRAIN's pre-submission check (self- and production-correlation) on alphas that pass" />
            </div>
          </Card>
        )}
      </div>

      <div className="flex flex-col gap-3 xl:col-span-7">
        {!st?.connected ? (
          <Card>
            <Empty title="Sign in to BRAIN to use these tools">
              Once connected you can import BRAIN's data catalog, pull in your own alphas and results, simulate any alpha on BRAIN from Studio or the Library, and run a closed mining loop where BRAIN judges every candidate.
            </Empty>
          </Card>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <Card title={<span className="flex items-center gap-1.5"><CloudDownload size={13} /> Data catalog</span>}>
                <div className="text-[12px] text-ink2">
                  Imports field ids, descriptions, coverage and how many alphas already use each field. The Idea Forge then finds BRAIN fields your idea describes, and BRAIN mining prefers well-covered, less-crowded fields (lower self-correlation).
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <button className="btn btn-primary" disabled={busy} onClick={() => act(() => api.brain.syncFields({ auto: pick.size === 0, datasets: [...pick] }), "Catalog sync started")}>
                    <CloudDownload size={13} /> {pick.size ? `Import ${pick.size} dataset(s)` : "Import top datasets"}
                  </button>
                  {ds?.synced && <span className="self-center text-[11px] text-muted">last sync {ds.synced}</span>}
                </div>
                {datasets.length > 0 && (
                  <div className="mt-2 max-h-56 overflow-auto rounded border border-line">
                    {datasets.map((d: any) => (
                      <label key={d.id} className="flex cursor-pointer items-center gap-2 border-b border-line px-2 py-1 text-[11.5px] last:border-0 hover:bg-[var(--surface-2)]">
                        <input
                          type="checkbox"
                          checked={pick.has(d.id)}
                          onChange={() => setPick((p) => { const n = new Set(p); n.has(d.id) ? n.delete(d.id) : n.add(d.id); return n; })}
                        />
                        <span className="mono w-28 truncate">{d.id}</span>
                        <span className="min-w-0 flex-1 truncate text-ink2" title={d.description}>{d.name}</span>
                        <span className="tnum text-muted">{d.field_count ?? "?"} f</span>
                        {ds?.imported_fields?.[d.id] ? <span className="chip">{ds.imported_fields[d.id]} in</span> : null}
                      </label>
                    ))}
                  </div>
                )}
              </Card>
              <Card title="Your BRAIN alphas">
                <div className="text-[12px] text-ink2">
                  Imports your recent BRAIN alphas with their real results. Submitted ones are marked, and their PnL feeds the local SELF_CORRELATION check. Every result recalibrates the local pass model and steers the miners toward what actually passes.
                </div>
                <button className="btn btn-primary mt-2" disabled={busy} onClick={() => act(() => api.brain.syncAlphas({ limit: 300 }), "Alpha sync started")}>
                  <RefreshCw size={13} /> Sync my alphas
                </button>
              </Card>
            </div>
            <Card title={<span className="flex items-center gap-1.5"><Rocket size={13} /> BRAIN mining: BRAIN judges every candidate</span>}>
              <div className="text-[12px] text-ink2">
                Sends your best untested local alphas (by quality grade) and fresh ideas on imported catalog fields to BRAIN. Near misses are repaired from BRAIN's own failing checks and re-simulated. Nothing is submitted.
              </div>
              <div className="mt-2 flex flex-wrap items-end gap-3">
                <Num label="Simulation budget" value={mine.budget} min={1} onChange={(v) => setMine((m) => ({ ...m, budget: v }))} />
                <Num label="Time limit (min)" value={mine.time_limit_min} min={1} onChange={(v) => setMine((m) => ({ ...m, time_limit_min: v }))} />
                <Num label="Repair rounds" value={mine.refine_rounds} min={0} max={4} onChange={(v) => setMine((m) => ({ ...m, refine_rounds: v }))} />
                <Field label="Library alphas of grade">
                  <Select value={mine.min_grade} onChange={(v) => setMine((m) => ({ ...m, min_grade: v }))} options={[{ value: "A", label: "A only" }, { value: "B", label: "A or B" }, { value: "C", label: "A to C" }]} />
                </Field>
              </div>
              <div className="mt-2 flex flex-wrap gap-4">
                <Toggle checked={mine.lib} onChange={(v) => setMine((m) => ({ ...m, lib: v }))} label="Best untested library alphas" />
                <Toggle checked={mine.fields} onChange={(v) => setMine((m) => ({ ...m, fields: v }))} label="Ideas on imported catalog fields" />
              </div>
              <div className="mt-2 flex items-center gap-2">
                <button
                  className="btn btn-primary"
                  disabled={busy || (!mine.lib && !mine.fields)}
                  onClick={() =>
                    act(
                      () =>
                        api.brain.mine({
                          budget: mine.budget,
                          time_limit_min: mine.time_limit_min,
                          refine_rounds: mine.refine_rounds,
                          min_grade: mine.min_grade,
                          sources: [...(mine.lib ? ["library"] : []), ...(mine.fields ? ["brain_fields"] : [])],
                        }),
                      "BRAIN mining started",
                    )
                  }
                >
                  <Rocket size={13} /> Start BRAIN mining
                </button>
                <span className="text-[11px] text-muted">{st.budget_left} simulations left in today's budget</span>
              </div>
            </Card>
          </>
        )}

        <Card title="BRAIN jobs" bodyClass="px-0 pb-1">
          {jobs.length === 0 ? (
            <Empty title="No BRAIN jobs yet" />
          ) : (
            <div className="max-h-[200px] overflow-auto">
              {jobs.map((j) => (
                <button
                  key={j.id}
                  onClick={() => {
                    setSelected(j.id);
                    setParams({ job: String(j.id) });
                  }}
                  className={clsx("flex w-full items-center gap-2 border-b border-line px-3 py-1.5 text-left text-[12px] hover:bg-[var(--surface-2)]", sel?.id === j.id && "bg-[var(--surface-2)]")}
                >
                  <span className="w-10 text-muted">#{j.id}</span>
                  <span className="w-28 font-medium">{j.kind.replace("brain_", "")}</span>
                  {jobStatusBadge(j.status)}
                  <span className="tnum ml-auto text-muted">
                    {j.stats?.brain_simulated ?? 0} simulated · {j.stats?.brain_passed ?? 0} pass · {j.stats?.brain_ready ?? 0} ready · {fmt.ago(j.created_at)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </Card>
        {sel && <JobPanel job={sel} />}
      </div>
    </div>
  );
}
