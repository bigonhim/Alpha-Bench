import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, DownloadCloud, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Card, Empty, Kv, Progress, Select, Spinner } from "../components/ui";
import { api } from "../lib/api";
import { useStatus } from "../lib/hooks";
import { useJobs } from "../lib/store";

export default function DataPage() {
  const { data: status } = useStatus();
  const { data: cov, isFetching } = useQuery({ queryKey: ["coverage", status?.data?.version], queryFn: api.coverage });
  const [email, setEmail] = useState("");
  const [maxTickers, setMaxTickers] = useState<string>("");
  const [pool, setPool] = useState<string>("broad");
  const qc = useQueryClient();
  const jobs = useJobs((s) => s.jobs);
  const buildJob = useMemo(() => Object.values(jobs).filter((j) => j.kind === "data_build" || j.kind === "demo_build").sort((a, b) => b.id - a.id)[0], [jobs]);
  const running = buildJob && ["running", "paused", "queued"].includes(buildJob.status);

  useEffect(() => {
    if (status?.settings?.sec_contact_email !== undefined) setEmail(status.settings.sec_contact_email || "");
  }, [status?.settings?.sec_contact_email]);
  useEffect(() => {
    if (status?.settings?.universe_pool) setPool(status.settings.universe_pool);
  }, [status?.settings?.universe_pool]);

  const saveSettings = async (u: Record<string, unknown>) => {
    try {
      await api.saveSettings(u);
      qc.invalidateQueries();
      toast.success("Saved");
    } catch (e) {
      toast.error(String((e as Error).message));
    }
  };

  const build = async () => {
    if (!email.includes("@")) {
      toast.error("Enter a contact email first (SEC EDGAR and Wikipedia require one for automated downloads).");
      return;
    }
    await api.saveSettings({ sec_contact_email: email, universe_pool: pool });
    const r = await api.buildData({ pool, ...(maxTickers ? { max_tickers: Number(maxTickers) } : {}) });
    toast.success(`Data build started (job #${r.id})`);
  };

  const d = status?.data;
  const covRows = Object.entries(cov?.coverage ?? {}).sort((a, b) => a[0].localeCompare(b[0]));
  return (
    <div className="grid grid-cols-1 gap-3 p-3 xl:grid-cols-12">
      <div className="flex flex-col gap-3 xl:col-span-5">
        <Card title={<span className="flex items-center gap-1.5"><Database size={13} /> Active dataset</span>}>
          {d ? (
            <>
              <Kv k="Source" v={d.source === "real" ? "Real (Yahoo + SEC EDGAR)" : "Synthetic demo (mining disabled)"} />
              {d.source === "real" && <Kv k="Pool" v={d.pool === "broad" ? `Broad US (TOP3000-like)` : "S&P 1500"} />}
              {d.classification && <Kv k="Classification" v={d.classification} />}
              <Kv k="Dates" v={`${d.start} → ${d.end}`} />
              <Kv k="Instruments × days" v={`${d.N.toLocaleString()} × ${d.T.toLocaleString()}`} />
              <Kv k="Fields" v={d.fields.length} />
              <Kv k="Universes" v={d.universes.join(", ")} />
              <Kv k="In-sample" v={`${status?.periods.is_start} → ${status?.periods.os_start}`} />
              <Kv k="Out-of-sample holdout" v={`${status?.periods.os_start} → ${status?.periods.end}`} />
              <Kv k="BRAIN-like window" v={`${status?.periods.brain_start} → ${status?.periods.os_start}`} />
              <Kv k="Built" v={d.built ?? "—"} />
              <div className="mt-2 flex items-center gap-2">
                <span className="text-[12px] text-muted">Use</span>
                <Select
                  value={status?.settings?.active_dataset ?? "auto"}
                  onChange={(v) => saveSettings({ active_dataset: v })}
                  options={[
                    { value: "auto", label: "Auto (real if built, else demo)" },
                    { value: "real", label: "Real data" },
                    { value: "demo", label: "Synthetic demo" },
                  ]}
                />
              </div>
            </>
          ) : (
            <Spinner />
          )}
        </Card>
        <Card title={<span className="flex items-center gap-1.5"><DownloadCloud size={13} /> Build / update real data</span>}>
          <div className="text-[12px] text-ink2">
            Downloads daily prices (Yahoo Finance) and point-in-time fundamentals (SEC EDGAR XBRL: first-filed values, usable the day after filing) for the chosen pool. Updates are incremental.
          </div>
          <div className="mt-2 flex flex-col gap-1.5">
            <label className="lbl">Stock pool</label>
            <Select
              value={pool}
              onChange={setPool}
              options={[
                { value: "broad", label: "Broad US: ~3,400 liquid stocks, universes up to TOP3000 (closest to BRAIN)" },
                { value: "sp1500", label: "S&P 1500 only: faster, universes up to TOP1500" },
              ]}
            />
            <div className="text-[11.5px] text-muted">
              {pool === "broad"
                ? "Every listed US common stock from SEC's exchange list, kept when it ranks among the ~3,450 most liquid at any month end. Names outside the S&P 1500 get GICS-like groups from a SIC crosswalk learned on the S&P names. First build: about 40-70 minutes and 2-3 GB of disk (pause OneDrive sync while it runs). Simulations are about twice as slow as with the S&P 1500."
                : "Current S&P 500/400/600 constituents with GICS classes (Wikipedia). First build: about 10-20 minutes, about 1 GB of disk. BRAIN TOP3000 is simulated on local TOP1500, which misses BRAIN's smaller names."}
            </div>
          </div>
          <div className="mt-2 flex flex-col gap-2">
            <label className="lbl">Contact email: included in the User-Agent of SEC EDGAR and Wikipedia requests only, as their access policies require</label>
            <div className="flex gap-2">
              <input className="input flex-1" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
              <button className="btn" onClick={() => saveSettings({ sec_contact_email: email })}>
                Save
              </button>
            </div>
            <div className="flex items-center gap-2">
              <input className="input w-40 tnum" type="number" placeholder="max tickers (all)" value={maxTickers} onChange={(e) => setMaxTickers(e.target.value)} title="Limit for a quick trial build" />
              <button className="btn btn-primary ml-auto" onClick={build} disabled={!!running}>
                {running ? <Spinner /> : <RefreshCw size={13} />} {status?.real_data_available ? "Update data" : "Build real data"}
              </button>
            </div>
          </div>
          {buildJob && (
            <div className="mt-3 rounded-md border border-line p-2">
              <div className="mb-1 flex justify-between text-[12px]">
                <span className="text-ink2">{buildJob.progress?.phase ?? buildJob.status}</span>
                <span className="tnum text-muted">{buildJob.progress?.total ? `${buildJob.progress.done}/${buildJob.progress.total}` : buildJob.status}</span>
              </div>
              <Progress value={running ? (buildJob.progress?.done ?? 0) / Math.max(1, buildJob.progress?.total ?? 1) : 1} />
              {buildJob.status === "error" && <div className="mt-1 text-[12px] text-[var(--bad-text)]">{buildJob.error ?? "failed — see job details in the Miner"}</div>}
            </div>
          )}
          <div className="mt-3 text-[11.5px] text-muted">
            Limitations: current constituents only (survivorship bias), typical price as VWAP proxy, TTM fundamentals mapped to BRAIN-style names. The proxy ranks and filters; BRAIN remains the ground truth. Import BRAIN results to calibrate.
          </div>
        </Card>
      </div>
      <div className="xl:col-span-7">
        <Card title="Field coverage (last 252 days)" actions={isFetching ? <Spinner /> : null}>
          {covRows.length === 0 ? (
            <Empty title="No coverage data" />
          ) : (
            <div className="grid grid-cols-1 gap-x-6 md:grid-cols-2">
              {covRows.map(([k, v]) => (
                <div key={k} className="flex items-center gap-2 py-0.5 text-[12px]">
                  <span className="mono w-36 truncate">{k}</span>
                  <div className="h-1.5 flex-1 rounded bg-[var(--surface-3)]">
                    <div className="h-full rounded bg-[var(--series-1)]" style={{ width: `${v * 100}%` }} />
                  </div>
                  <span className="tnum w-12 text-right text-ink2">{(v * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
