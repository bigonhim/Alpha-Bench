import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FileUp, Import } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { CalibrationScatter } from "../components/charts";
import { Card, Empty, Kv, Select, Spinner, StatusBadge, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { fmt } from "../lib/format";

const SAMPLE = `code,region,universe,delay,decay,neutralization,truncation,sharpe,fitness,turnover,returns,drawdown,margin,status
"rank(-ts_delta(close, 5))",USA,TOP3000,1,4,SUBINDUSTRY,0.08,1.41,1.02,38.2%,9.1%,6.2%,4.8,PASS`;

export default function ImportPage() {
  const [text, setText] = useState("");
  const [format, setFormat] = useState("auto");
  const [markSub, setMarkSub] = useState(false);
  const [busy, setBusy] = useState(false);
  const [summary, setSummary] = useState<any>(null);
  const [fieldsText, setFieldsText] = useState("");
  const qc = useQueryClient();
  const { data: cal, isFetching } = useQuery({ queryKey: ["calibration"], queryFn: api.calibration });

  const run = async () => {
    setBusy(true);
    try {
      const s = await api.importResults(text, format, markSub);
      setSummary(s);
      toast.success(`Imported ${s.parsed} rows: ${s.matched} matched, ${s.created} new`);
      qc.invalidateQueries({ queryKey: ["calibration"] });
      qc.invalidateQueries({ queryKey: ["alphas"] });
    } catch (e) {
      toast.error(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setText(await f.text());
    if (f.name.endsWith(".json")) setFormat("json");
    else if (f.name.endsWith(".csv") || f.name.endsWith(".tsv")) setFormat("csv");
  };

  return (
    <div className="grid grid-cols-1 gap-3 p-3 xl:grid-cols-12">
      <div className="flex flex-col gap-3 xl:col-span-6">
        <Card
          title="Import BRAIN results"
          actions={
            <label className="btn cursor-pointer">
              <FileUp size={13} /> Open file
              <input type="file" accept=".csv,.tsv,.json,.txt" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
            </label>
          }
        >
          <div className="mb-2 text-[12px] text-muted">
            Paste IS stats from BRAIN: a CSV (columns like code/expression, sharpe, fitness, turnover, returns, drawdown, margin, status), JSON alpha objects from your own scripts, or text copied from the BRAIN results panel (expression line followed by the metrics). Rows are matched to library alphas by canonical expression. They calibrate the pass-likelihood model and steer the miners.
          </div>
          <textarea className="input mono h-56 w-full text-[11.5px]" value={text} onChange={(e) => setText(e.target.value)} placeholder={SAMPLE} />
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <Select value={format} onChange={setFormat} options={[{ value: "auto", label: "Auto-detect" }, { value: "csv", label: "CSV" }, { value: "json", label: "JSON" }, { value: "text", label: "Copied text" }]} />
            <Toggle checked={markSub} onChange={setMarkSub} label="Mark all as submitted" />
            <button className="btn btn-primary ml-auto" onClick={run} disabled={busy || !text.trim()}>
              {busy ? <Spinner /> : <Import size={13} />} Import
            </button>
          </div>
        </Card>
        {summary && (
          <Card title={`Import summary · ${summary.parsed} parsed · ${summary.matched} matched · ${summary.created} created`}>
            {summary.errors.length > 0 && (
              <div className="mb-2 text-[12px] text-[var(--bad-text)]">
                {summary.errors.map((e: any, i: number) => (
                  <div key={i}>
                    <span className="mono">{e.expr}</span>: {e.error}
                  </div>
                ))}
              </div>
            )}
            <div className="max-h-72 overflow-auto">
              <table className="tbl tnum">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>BRAIN</th>
                    <th className="text-right">BRAIN Sharpe</th>
                    <th className="text-right">Local Sharpe</th>
                    <th>Expression</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.items.map((it: any) => (
                    <tr key={it.id}>
                      <td>{it.id}</td>
                      <td>
                        <StatusBadge status={it.passed ? "PASS" : it.passed === false ? "FAIL" : "PENDING"} compact />
                      </td>
                      <td className="text-right">{fmt.num(it.brain_sharpe)}</td>
                      <td className="text-right">{fmt.num(it.local_sharpe)}</td>
                      <td className="mono max-w-[340px] truncate">{it.expr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
        <Card title="Import BRAIN data-field catalog">
          <div className="mb-2 text-[12px] text-muted">
            Paste the fields you see in BRAIN's Data Explorer (CSV with columns id/field, description, dataset, type, or JSON). The generators use them to build BRAIN-only candidates with real field ids for your account.
          </div>
          <textarea className="input mono h-28 w-full text-[11.5px]" value={fieldsText} onChange={(e) => setFieldsText(e.target.value)} placeholder={"id,description,dataset,type\nanl4_eps_mean,EPS consensus mean,analyst4,MATRIX"} />
          <div className="mt-2 flex justify-end">
            <button
              className="btn btn-primary"
              disabled={!fieldsText.trim()}
              onClick={async () => {
                try {
                  const r = await api.importFields(fieldsText);
                  toast.success(`Imported ${r.imported} fields`);
                  qc.invalidateQueries({ queryKey: ["catalog"] });
                } catch (e) {
                  toast.error(String((e as Error).message));
                }
              }}
            >
              Import fields
            </button>
          </div>
        </Card>
      </div>
      <div className="flex flex-col gap-3 xl:col-span-6">
        <Card title="Local vs BRAIN calibration" actions={isFetching ? <Spinner /> : null}>
          {!cal || cal.points.length === 0 ? (
            <Empty title="No BRAIN results imported yet">Once imported, this chart shows how local proxy Sharpe maps to BRAIN Sharpe, and the pass-likelihood model refits (after {cal?.min_rows_for_fit ?? 30} labeled rows).</Empty>
          ) : (
            <>
              <CalibrationScatter points={cal.points} map={cal.sharpe_map} />
              <div className="mt-2 grid grid-cols-2 gap-x-6">
                <Kv k="Points" v={cal.points.length} />
                <Kv k="Model rows fitted" v={cal.n_fit || `prior (needs ${cal.min_rows_for_fit})`} />
                {cal.sharpe_map && (
                  <>
                    <Kv k="BRAIN ≈ a·local + b" v={`${fmt.num(cal.sharpe_map.slope)} · x ${cal.sharpe_map.intercept >= 0 ? "+" : "−"} ${fmt.num(Math.abs(cal.sharpe_map.intercept))}`} />
                    <Kv k="R²" v={fmt.num(cal.sharpe_map.r2)} />
                  </>
                )}
                {Object.entries(cal.ratios || {}).map(([k, v]) => (
                  <Kv key={k} k={`median BRAIN/local ${k}`} v={fmt.num(v as number)} />
                ))}
              </div>
            </>
          )}
        </Card>
        <Card title="Pass rate by idea family (BRAIN)">
          {cal && Object.keys(cal.families || {}).length ? (
            <table className="tbl tnum">
              <thead>
                <tr>
                  <th>Idea</th>
                  <th className="text-right">Imported</th>
                  <th className="text-right">Pass rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(cal.families).map(([k, v]: any) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="text-right">{v.n}</td>
                    <td className="text-right">{fmt.pct(v.pass_rate, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-[12px] text-muted">No labeled BRAIN results yet.</div>
          )}
        </Card>
        <Card title="Search bandit (what the miners favour)">
          <div className="max-h-72 overflow-auto">
            <table className="tbl tnum">
              <thead>
                <tr>
                  <th>Arm</th>
                  <th className="text-right">Trials</th>
                  <th className="text-right">Success mean</th>
                </tr>
              </thead>
              <tbody>
                {(cal?.bandit ?? []).map((b: any) => (
                  <tr key={b.arm}>
                    <td className="mono">{b.arm}</td>
                    <td className="text-right">{b.n}</td>
                    <td className="text-right">{fmt.pct(b.mean, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
