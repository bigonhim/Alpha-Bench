import { useQuery } from "@tanstack/react-query";
import { Database, FlaskConical, Import, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { jobStatusBadge } from "../components/JobPanel";
import { Card, Empty, Kv, Spinner, StatusBadge } from "../components/ui";
import { api } from "../lib/api";
import { fmt } from "../lib/format";

function Tile({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="card flex flex-1 flex-col gap-0.5 px-4 py-3">
      <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">{label}</span>
      <span className="text-[26px] font-semibold leading-tight">{value}</span>
      {sub && <span className="text-[11.5px] text-muted">{sub}</span>}
    </div>
  );
}

export default function Dashboard() {
  const { data, isFetching } = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard, refetchInterval: 20_000 });
  const nav = useNavigate();
  if (!data) return <div className="p-4"><Spinner /></div>;
  const s = data.stats;
  const ideas: any[] = [...(data.ideas ?? [])].sort((a, b) => (b.n ?? 0) - (a.n ?? 0));
  const maxN = Math.max(1, ...ideas.map((i) => i.n ?? 0));
  const pyr = Object.entries<number>(data.pyramid ?? {});
  const automine = async () => {
    const r = await api.startJob("automine", { time_limit_min: 20, target_candidates: 20 });
    toast.success(`Auto-Mine started (job #${r.id})`);
    nav(`/miner?job=${r.id}`);
  };
  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex flex-wrap gap-3">
        <Tile label="Alphas in library" value={s.alphas.toLocaleString()} sub={`${s.brain_only} BRAIN-only`} />
        <Tile label="Passing local checks" value={s.local_pass.toLocaleString()} sub={s.alphas ? `${Math.round((s.local_pass / s.alphas) * 100)}% of library` : undefined} />
        <Tile label="Submitted on BRAIN" value={s.submitted} sub="used for self-correlation" />
        <Tile label="BRAIN results imported" value={s.brain_imported} sub={s.brain_imported ? `${Math.round(((s.brain_passed || 0) / s.brain_imported) * 100)}% passed` : "import to calibrate"} />
        <Tile label="Best local fitness" value={fmt.num(s.best_fitness)} sub={`best Sharpe ${fmt.num(s.best_sharpe)}`} />
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="btn btn-primary" onClick={() => nav("/studio")}><FlaskConical size={13} /> New alpha</button>
        <button className="btn" onClick={automine}><Sparkles size={13} /> Auto-Mine 20 min</button>
        <button className="btn" onClick={() => nav("/import")}><Import size={13} /> Import BRAIN results</button>
        {data.data?.source !== "real" && <button className="btn" onClick={() => nav("/data")}><Database size={13} /> Build real data</button>}
        {isFetching && <Spinner />}
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
        <Card title="Top submission candidates (by BRAIN pass likelihood)" className="xl:col-span-8" bodyClass="px-0 pb-1">
          {data.top_candidates.length === 0 ? (
            <Empty title="No passing candidates yet">Start an Auto-Mine run or write alphas in the Studio.</Empty>
          ) : (
            <table className="tbl tnum">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Status</th>
                  <th>Robustness</th>
                  <th className="text-right">P(pass)</th>
                  <th className="text-right">Sharpe</th>
                  <th className="text-right">Fitness</th>
                  <th className="text-right">TO</th>
                  <th className="text-right">OS Sh.</th>
                  <th>Idea</th>
                  <th>Expression</th>
                </tr>
              </thead>
              <tbody>
                {data.top_candidates.map((a: any) => (
                  <tr key={a.id} className="cursor-pointer" onClick={() => nav(`/library?open=${a.id}`)}>
                    <td className="text-muted">{a.id}</td>
                    <td><StatusBadge status={a.status_local} compact /></td>
                    <td>{a.robust ? <StatusBadge status="PASS" label="robust" compact /> : <StatusBadge status="WARNING" label="review OS" compact />}</td>
                    <td className="text-right">{a.pass_prob != null ? `${Math.round(a.pass_prob * 100)}%` : "—"}</td>
                    <td className="text-right">{fmt.num(a.sharpe)}</td>
                    <td className="text-right">{fmt.num(a.fitness)}</td>
                    <td className="text-right">{fmt.pct(a.turnover, 0)}</td>
                    <td className="text-right">{fmt.num(a.os_sharpe)}</td>
                    <td className="text-ink2">{a.idea}</td>
                    <td className="mono max-w-[380px] truncate text-ink2">{a.expr}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <div className="flex flex-col gap-3 xl:col-span-4">
          <Card title="Dataset">
            <Kv k="Source" v={data.data.source === "real" ? "Real (Yahoo + SEC)" : "Synthetic demo"} />
            <Kv k="Coverage" v={`${data.data.N} stocks · ${data.data.start} → ${data.data.end}`} />
            <Kv k="IS / OS split" v={`${data.periods.is_start} | ${data.periods.os_start}`} />
            <Kv k="Cache" v={`${data.cache.mb} MB · hit rate ${fmt.pct(data.cache.hit_rate, 0)}`} />
          </Card>
          <Card title="Recent jobs" bodyClass="px-0 pb-1">
            {data.jobs.length === 0 ? (
              <div className="px-3 text-[12px] text-muted">No jobs yet.</div>
            ) : (
              data.jobs.map((j: any) => (
                <button key={j.id} className="flex w-full items-center gap-2 border-b border-line px-3 py-1.5 text-left text-[12px] hover:bg-[var(--surface-2)]" onClick={() => nav(`/miner?job=${j.id}`)}>
                  <span className="w-8 text-muted">#{j.id}</span>
                  <span className="w-24">{j.kind}</span>
                  {jobStatusBadge(j.status)}
                  <span className="tnum ml-auto text-muted">{j.stats?.saved ?? 0} saved</span>
                </button>
              ))
            )}
          </Card>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <Card title="Library by idea family">
          <table className="tbl tnum">
            <thead>
              <tr>
                <th>Idea</th>
                <th>Alphas</th>
                <th className="text-right">Passing</th>
                <th className="text-right">Avg Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {ideas.map((i) => (
                <tr key={i.idea ?? "none"}>
                  <td>{i.idea ?? "—"}</td>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-32 rounded bg-[var(--surface-3)]">
                        <div className="h-full rounded bg-[var(--series-1)]" style={{ width: `${((i.n ?? 0) / maxN) * 100}%` }} />
                      </div>
                      {i.n}
                    </div>
                  </td>
                  <td className="text-right">{i.p ?? 0}</td>
                  <td className="text-right">{fmt.num(i.s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Pyramid coverage of submitted alphas (region · delay · data category)">
          {pyr.length === 0 ? (
            <Empty title="No submitted alphas yet">Mark alphas as submitted (or import BRAIN results) to track diversity across regions, delays and data categories.</Empty>
          ) : (
            <table className="tbl tnum">
              <thead>
                <tr>
                  <th>Region</th>
                  <th>Delay</th>
                  <th>Data category</th>
                  <th className="text-right">Alphas</th>
                </tr>
              </thead>
              <tbody>
                {pyr.map(([k, n]) => {
                  const [r, d, c] = k.split("|");
                  return (
                    <tr key={k}>
                      <td>{r}</td>
                      <td>{d}</td>
                      <td>{c}</td>
                      <td className="text-right">{n}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </Card>
      </div>
    </div>
  );
}
