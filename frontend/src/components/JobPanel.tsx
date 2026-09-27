import { useQuery } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Pause, Play, Square } from "lucide-react";
import { useMemo, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { fmt } from "../lib/format";
import { useJobs, useUI } from "../lib/store";
import type { JobResultRow, JobSnapshot } from "../lib/types";
import { GpProgressChart, ParetoScatter } from "./charts";
import { ReengineerView } from "./Reengineer";
import { Card, Empty, GradeBadge, Kv, Progress, StatusBadge } from "./ui";

const LIVE = new Set(["running", "paused", "queued"]);

export function jobStatusBadge(status: string) {
  const map: Record<string, string> = { done: "PASS", error: "FAIL", cancelled: "WARNING", running: "PENDING", paused: "WARNING", queued: "PENDING" };
  return <StatusBadge status={map[status] || "PENDING"} label={status} compact />;
}

export function ResultsTable({ rows, height = 360 }: { rows: JobResultRow[]; height?: number }) {
  const parent = useRef<HTMLDivElement>(null);
  const nav = useNavigate();
  const setStudio = useUI((s) => s.setStudio);
  const v = useVirtualizer({ count: rows.length, getScrollElement: () => parent.current, estimateSize: () => 28, overscan: 12 });
  if (!rows.length) return <Empty title="No saved results yet">Alphas that clear the save threshold stream in here as they are evaluated.</Empty>;
  return (
    <div>
      <div className="grid grid-cols-[92px_30px_60px_60px_56px_64px_52px_1fr] gap-2 border-b border-line px-2 py-1.5 text-[10.5px] font-semibold uppercase tracking-wide text-muted">
        <span>Status</span>
        <span>Gr.</span>
        <span className="text-right">Sharpe</span>
        <span className="text-right">Fitness</span>
        <span className="text-right">TO</span>
        <span className="text-right">OS Sh.</span>
        <span className="text-right">P(pass)</span>
        <span>Expression</span>
      </div>
      <div ref={parent} style={{ height, overflow: "auto" }}>
        <div style={{ height: v.getTotalSize(), position: "relative" }}>
          {v.getVirtualItems().map((it) => {
            const r = rows[it.index];
            return (
              <button
                key={it.key}
                onClick={() => {
                  if (r.settings) setStudio(r.expr, r.settings);
                  else setStudio(r.expr);
                  nav("/studio");
                }}
                className="tnum absolute left-0 grid w-full grid-cols-[92px_30px_60px_60px_56px_64px_52px_1fr] items-center gap-2 border-b border-line px-2 text-left text-[12px] hover:bg-[var(--surface-2)]"
                style={{ top: it.start, height: it.size }}
                title="Open in Studio"
              >
                <StatusBadge status={r.status} compact label={r.status_brain ? `BRAIN ${r.status_brain}` : undefined} />
                <GradeBadge grade={r.grade} />
                <span className="text-right">{fmt.num(r.sharpe)}</span>
                <span className="text-right">{fmt.num(r.fitness)}</span>
                <span className="text-right">{fmt.pct(r.turnover, 0)}</span>
                <span className="text-right">{fmt.num(r.os_sharpe)}</span>
                <span className="text-right">{r.pass_prob != null ? `${Math.round(r.pass_prob * 100)}%` : "—"}</span>
                <span className="mono truncate text-ink2">{r.expr}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function JobPanel({ job }: { job: JobSnapshot }) {
  const live = useJobs((s) => s.jobs[job.id]);
  const liveRows = useJobs((s) => s.results[job.id]);
  const gp = useJobs((s) => s.gp[job.id]);
  const j = { ...job, ...(live || {}) };
  const isLive = LIVE.has(j.status);
  const { data: saved } = useQuery({
    queryKey: ["alphas", "job", job.id, j.status],
    queryFn: () => api.alphas({ job_id: job.id, sort: "fitness", limit: 500 }),
    enabled: !isLive || !liveRows?.length,
  });
  const rows: JobResultRow[] = useMemo(() => {
    if (liveRows?.length && isLive) return [...liveRows].sort((a, b) => (b.fitness ?? -9) - (a.fitness ?? -9));
    return (saved?.rows ?? []).map((a) => ({
      id: a.id,
      expr: a.expr,
      sharpe: a.sharpe,
      fitness: a.fitness,
      turnover: a.turnover,
      status: a.status_local,
      pass_prob: a.pass_prob,
      os_sharpe: a.os_sharpe,
      idea: a.idea,
      origin: a.origin,
      settings: a.settings,
    }));
  }, [liveRows, saved, isLive]);
  const p = j.progress || {};
  const s = j.stats || {};
  const frac = p.total ? (p.done || 0) / p.total : 0;
  const last = gp?.[gp.length - 1];
  return (
    <div className="flex flex-col gap-3">
      <Card
        title={
          <span className="flex items-center gap-2">
            Job #{j.id} · {j.kind} {jobStatusBadge(j.status)}
          </span>
        }
        actions={
          isLive && (
            <>
              {j.status === "paused" ? (
                <button className="btn" onClick={() => api.jobAction(j.id, "resume")}>
                  <Play size={12} /> Resume
                </button>
              ) : (
                <button className="btn" onClick={() => api.jobAction(j.id, "pause")}>
                  <Pause size={12} /> Pause
                </button>
              )}
              <button className="btn btn-danger" onClick={() => api.jobAction(j.id, "cancel")}>
                <Square size={12} /> Stop
              </button>
            </>
          )
        }
      >
        <div className="mb-1 flex items-center justify-between text-[12px]">
          <span className="text-ink2">{p.phase || j.status}</span>
          <span className="tnum text-muted">{p.total ? `${p.done ?? 0} / ${p.total}` : ""}</span>
        </div>
        <Progress value={isLive ? frac : 1} />
        <div className="mt-2 grid grid-cols-2 gap-x-6 md:grid-cols-4">
          <Kv k="Evaluated" v={s.evaluated ?? 0} />
          <Kv k="Saved" v={s.saved ?? 0} />
          <Kv k="Passing checks" v={s.passed ?? 0} />
          <Kv k="Throughput" v={`${fmt.num(s.rate_per_s, 1)}/s`} />
          <Kv k="Errors" v={s.errors ?? 0} />
          <Kv k="BRAIN-only" v={s.brain_only ?? 0} />
          <Kv k="Rejected (correlated)" v={s.rejected_corr ?? 0} />
          <Kv k="Elapsed" v={`${Math.round(s.elapsed_s ?? 0)}s`} />
        </div>
        {j.error && <div className="mt-2 text-[12px] text-[var(--bad-text)]">{j.error}</div>}
      </Card>
      {gp && gp.length > 0 && (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <Card title="GP fitness by generation">
            <GpProgressChart stats={gp} />
          </Card>
          <Card title={`Pareto front · generation ${last?.generation}`}>
            <ParetoScatter front={last?.front ?? []} />
          </Card>
        </div>
      )}
      {j.kind === "reengineer" ? (
        <ReengineerView jobId={j.id} />
      ) : (
        <Card title={`Results (${rows.length})`} bodyClass="px-0 pb-1">
          <ResultsTable rows={rows} />
        </Card>
      )}
    </div>
  );
}
