import * as Dialog from "@radix-ui/react-dialog";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Braces, Copy, Dna, Download, FlaskConical, Grid3x3, Layers, SlidersHorizontal, Star, Trash2, Upload, Wand2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { ComboChart, Heatmap } from "../components/charts";
import { ChartsCard, ChecksList, DescriptionCard, MetricStrip, RobustnessCard } from "../components/ResultView";
import { Card, Empty, Kv, Modal, Select, Spinner, StatusBadge, Toggle } from "../components/ui";
import { api, copyText, downloadText, type ListParams } from "../lib/api";
import { useDebounced } from "../lib/hooks";
import { clsx, fmt, settingsLabel } from "../lib/format";
import { useUI } from "../lib/store";
import type { Alpha } from "../lib/types";
import { brainPayload } from "./Studio";
import { startReengineer } from "../components/Reengineer";

const COLS = "28px 22px 52px 92px 58px 58px 52px 60px 58px 58px 52px 48px 92px 80px 1fr";

function ExportModal({ ids, open, onOpenChange }: { ids: number[]; open: boolean; onOpenChange: (o: boolean) => void }) {
  const [format, setFormat] = useState("json");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!open || !ids.length) return;
    setLoading(true);
    api
      .exportText(ids, format, 10)
      .then(setText)
      .catch((e) => toast.error(String(e.message)))
      .finally(() => setLoading(false));
  }, [open, ids, format]);
  const ext = { json: "json", csv: "csv", text: "txt", markdown: "md" }[format] || "txt";
  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title={`Export ${ids.length} alpha${ids.length === 1 ? "" : "s"}`}
      wide
      footer={
        <>
          <button className="btn" onClick={() => { copyText(text); toast.success("Copied"); }}>
            <Copy size={13} /> Copy
          </button>
          <button className="btn btn-primary" onClick={() => downloadText(`alphas.${ext}`, text)}>
            <Download size={13} /> Download .{ext}
          </button>
        </>
      }
    >
      <div className="mb-2 flex items-center gap-2 text-[12px]">
        <span className="text-muted">Format</span>
        <Select
          value={format}
          onChange={setFormat}
          options={[
            { value: "json", label: "BRAIN simulation payloads (JSON, batches of 10)" },
            { value: "text", label: "Plain expressions with settings comments" },
            { value: "csv", label: "CSV with local metrics" },
            { value: "markdown", label: "Markdown reports" },
          ]}
        />
        {loading && <Spinner />}
      </div>
      <textarea className="input mono h-[52vh] w-full text-[11.5px]" readOnly value={text} />
      <div className="mt-1 text-[11px] text-muted">Alpha Foundry never contacts BRAIN. Paste these into BRAIN, or feed the JSON to your own multi-simulation workflow.</div>
    </Modal>
  );
}

function CorrelationModal({ ids, open, onOpenChange }: { ids: number[]; open: boolean; onOpenChange: (o: boolean) => void }) {
  const { data, isFetching } = useQuery({ queryKey: ["corr", ids], queryFn: () => api.correlation(ids), enabled: open && ids.length > 0 });
  const cells: [number, number, number | null][] = [];
  data?.matrix.forEach((row, i) => row.forEach((v, j) => cells.push([j, i, v])));
  const labels = (data?.ids ?? []).map((i) => `#${i}`);
  return (
    <Modal open={open} onOpenChange={onOpenChange} title="PnL correlation (last IS years, clustered)" wide>
      {isFetching && <Spinner />}
      {data && (
        <>
          <Heatmap xLabels={labels} yLabels={labels} cells={cells} height={Math.min(640, 140 + labels.length * 26)} invert valueLabel="correlation" showValues={labels.length <= 14} />
          <div className="mt-1 text-[11px] text-muted">Red = positively correlated (redundant; BRAIN's self-correlation limit is 0.7). Blue = negatively correlated. Rows are ordered by hierarchical clustering.</div>
          {data.missing.length > 0 && <div className="text-[11px] text-muted">Not simulated locally: {data.missing.map((i) => `#${i}`).join(", ")}</div>}
        </>
      )}
    </Modal>
  );
}

function CombineModal({ ids, open, onOpenChange }: { ids: number[]; open: boolean; onOpenChange: (o: boolean) => void }) {
  const [method, setMethod] = useState("equal");
  const { data, isFetching } = useQuery({ queryKey: ["combine", ids, method], queryFn: () => api.combine(ids, method), enabled: open && ids.length > 0 });
  return (
    <Modal open={open} onOpenChange={onOpenChange} title={`Combine ${ids.length} alphas (SuperAlpha preview)`} wide>
      <div className="mb-2 flex items-center gap-2 text-[12px]">
        <span className="text-muted">Weighting</span>
        <Select value={method} onChange={setMethod} options={[{ value: "equal", label: "Equal" }, { value: "inverse_vol", label: "Inverse volatility" }, { value: "sharpe", label: "IS Sharpe" }]} />
        {isFetching && <Spinner />}
      </div>
      {data?.ok ? (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <ComboChart dates={data.dates} cum={data.cum_pnl} height={280} />
          </div>
          <div className="flex flex-col gap-1">
            <Kv k="Combined IS Sharpe" v={fmt.num(data.is?.sharpe)} />
            <Kv k="Combined OS Sharpe" v={fmt.num(data.os?.sharpe)} />
            <Kv k="Avg pairwise correlation" v={fmt.num(data.avg_pairwise_corr)} />
            <div className="lbl mt-2">Components</div>
            {data.components.map((c: any, i: number) => (
              <Kv key={c.id} k={`#${c.id} (IS Sharpe ${fmt.num(c.is_sharpe)})`} v={`${(data.weights[i] * 100).toFixed(1)}%`} />
            ))}
            <div className="mt-1 text-[11px] text-muted">PnL-level blend. Combined turnover requires position-level re-simulation on BRAIN.</div>
          </div>
        </div>
      ) : (
        data && <div className="text-[12px] text-muted">{data.error}</div>
      )}
    </Modal>
  );
}

function AlphaDrawer({ id, onClose }: { id: number | null; onClose: () => void }) {
  const { data, isFetching } = useQuery({ queryKey: ["alpha", id], queryFn: () => api.alpha(id!), enabled: id !== null });
  const [notes, setNotes] = useState("");
  const [tags, setTags] = useState("");
  const qc = useQueryClient();
  const nav = useNavigate();
  const setStudio = useUI((s) => s.setStudio);
  const a = data?.alpha;
  useEffect(() => {
    setNotes(a?.notes ?? "");
    setTags((a?.tags ?? []).join(", "));
  }, [a?.id, a?.notes, a?.tags]);
  const patch = async (fields: Record<string, unknown>) => {
    if (!a) return;
    await api.patch(a.id, fields);
    qc.invalidateQueries({ queryKey: ["alpha", a.id] });
    qc.invalidateQueries({ queryKey: ["alphas"] });
  };
  return (
    <Dialog.Root open={id !== null} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <Dialog.Content className="fixed inset-y-0 right-0 z-50 flex w-[min(1080px,96vw)] flex-col border-l border-[var(--border)] bg-[var(--page)] shadow-2xl outline-none fade-in">
          <div className="flex items-center gap-2 border-b border-line bg-[var(--surface-1)] px-4 py-2.5">
            <Dialog.Title className="text-[14px] font-semibold">Alpha #{id}</Dialog.Title>
            {a && <StatusBadge status={a.status_local} />}
            {a?.status_brain && a.status_brain !== "untested" && <span className="chip">BRAIN: {a.status_brain}</span>}
            {isFetching && <Spinner />}
            <Dialog.Description className="sr-only">Alpha details</Dialog.Description>
            <Dialog.Close className="btn btn-ghost ml-auto !p-1" aria-label="Close">
              <X size={16} />
            </Dialog.Close>
          </div>
          {a && (
            <div className="flex-1 overflow-auto p-3">
              <div className="card mb-3 p-3">
                <div className="mono whitespace-pre-wrap break-all text-[13px]">{a.expr}</div>
                <div className="mt-1 text-[11.5px] text-muted">
                  {settingsLabel(a.settings)} · origin {a.origin}
                  {a.template_id ? ` · template ${a.template_id}` : ""} · {a.idea}/{a.category}/{a.horizon}
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <button className="btn btn-primary" onClick={() => { setStudio(a.expr, a.settings); nav("/studio"); }}>
                    <FlaskConical size={13} /> Open in Studio
                  </button>
                  <button className="btn" onClick={() => { copyText(a.expr); toast.success("Expression copied"); }}>
                    <Copy size={13} /> Copy
                  </button>
                  <button className="btn" onClick={() => { copyText(JSON.stringify(brainPayload(a.expr, a.settings), null, 2)); toast.success("BRAIN payload copied"); }}>
                    <Braces size={13} /> BRAIN JSON
                  </button>
                  <button className={clsx("btn", !!a.starred && "!text-[var(--warn-text)]")} onClick={() => patch({ starred: !a.starred })}>
                    <Star size={13} /> {a.starred ? "Starred" : "Star"}
                  </button>
                  {!!a.local && (
                    <button
                      className="btn"
                      title="Diagnose this alpha and search for a much stronger version (runs as a background job)"
                      onClick={async () => {
                        try {
                          const id = await startReengineer(a.expr, a.settings, {}, a.id);
                          toast.success(`Re-engineering alpha #${a.id} (job #${id})`);
                          nav(`/miner?job=${id}`);
                        } catch (e) {
                          toast.error(String((e as Error).message));
                        }
                      }}
                    >
                      <Wand2 size={13} /> Re-engineer
                    </button>
                  )}
                  <Toggle checked={!!a.submitted} onChange={(v) => patch({ submitted: v })} label="Submitted on BRAIN (used for self-correlation)" />
                </div>
                <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
                  <textarea className="input h-16" placeholder="Notes" value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => notes !== (a.notes ?? "") && patch({ notes })} />
                  <input className="input h-8" placeholder="tags, comma separated" value={tags} onChange={(e) => setTags(e.target.value)} onBlur={() => patch({ tags: tags.split(",").map((t) => t.trim()).filter(Boolean) })} />
                </div>
              </div>
              {data?.result && data.result.ok && !data.result.brain_only ? (
                <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
                  <div className="flex flex-col gap-3 xl:col-span-8">
                    <MetricStrip r={data.result} />
                    <ChartsCard r={data.result} />
                    <DescriptionCard r={data.result} />
                  </div>
                  <div className="flex flex-col gap-3 xl:col-span-4">
                    <ChecksList r={data.result} />
                    <RobustnessCard r={data.result} />
                    {data.brain.length > 0 && (
                      <Card title="BRAIN results (imported)">
                        {data.brain.map((b: any) => (
                          <div key={b.id} className="border-b border-line py-1 text-[12px] last:border-0">
                            <div className="flex justify-between">
                              <StatusBadge status={b.passed ? "PASS" : b.passed === 0 ? "FAIL" : "PENDING"} compact />
                              <span className="text-muted">{fmt.ago(b.imported_at)}</span>
                            </div>
                            <div className="tnum text-ink2">
                              Sharpe {fmt.num(b.sharpe)} · Fitness {fmt.num(b.fitness)} · TO {fmt.pct(b.turnover)} · Returns {fmt.pct(b.returns, 2)}
                            </div>
                          </div>
                        ))}
                      </Card>
                    )}
                  </div>
                </div>
              ) : (
                <Card title="BRAIN-only alpha">
                  <div className="text-[12px] text-muted">Not simulated locally: {(a.brain_only_reasons ?? []).join("; ") || "uses BRAIN-only data"}. Export it and test on BRAIN, then import the results.</div>
                </Card>
              )}
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export default function Library() {
  const [params, setParams] = useSearchParams();
  const [search, setSearch] = useState("");
  const [f, setF] = useState<ListParams>({ sort: "fitness", desc: true });
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [modal, setModal] = useState<"" | "export" | "corr" | "combine">("");
  const openId = params.get("open") ? Number(params.get("open")) : null;
  const dSearch = useDebounced(search, 250);
  const qc = useQueryClient();
  const nav = useNavigate();
  const q: ListParams = { ...f, search: dSearch, limit: 2000 };
  const { data, isFetching } = useQuery({ queryKey: ["alphas", q], queryFn: () => api.alphas(q) });
  const { data: facets } = useQuery({ queryKey: ["alphas", "facets"], queryFn: api.facets });
  const rows = data?.rows ?? [];
  const parent = useRef<HTMLDivElement>(null);
  const v = useVirtualizer({ count: rows.length, getScrollElement: () => parent.current, estimateSize: () => 30, overscan: 16 });
  const ids = useMemo(() => [...sel], [sel]);

  const setFilter = (k: keyof ListParams, val: unknown) => setF((p) => ({ ...p, [k]: val === "" ? undefined : val }));
  const toggle = (id: number) => setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const allOn = rows.length > 0 && rows.every((r) => sel.has(r.id));
  const refresh = () => qc.invalidateQueries({ queryKey: ["alphas"] });

  const bulk = async (what: string) => {
    if (!ids.length) return;
    try {
      if (what === "submit") await api.submitted(ids, true);
      if (what === "unsubmit") await api.submitted(ids, false);
      if (what === "star") await Promise.all(ids.map((i) => api.patch(i, { starred: true })));
      if (what === "delete") {
        if (!confirm(`Delete ${ids.length} alpha(s)? This cannot be undone.`)) return;
        await api.remove(ids);
        setSel(new Set());
      }
      if (what === "optimize") {
        const r = await api.startJob("settings_opt", { alpha_ids: ids });
        nav(`/miner?job=${r.id}`);
        return;
      }
      if (what === "gp") {
        const exprs = rows.filter((r) => sel.has(r.id)).map((r) => r.expr);
        const r = await api.startJob("gp", { seed_exprs: exprs, seed_from_library: false, population: 40, generations: 10 });
        nav(`/miner?job=${r.id}`);
        return;
      }
      toast.success("Done");
      refresh();
    } catch (e) {
      toast.error(String((e as Error).message));
    }
  };

  const sortHeader = (key: string, label: string, right = true) => (
    <button className={clsx("truncate uppercase", right && "text-right", f.sort === key && "text-ink")} onClick={() => setF((p) => ({ ...p, sort: key, desc: p.sort === key ? !p.desc : true }))}>
      {label}
      {f.sort === key ? (f.desc ? " ↓" : " ↑") : ""}
    </button>
  );

  return (
    <div className="flex h-full flex-col gap-2 p-3">
      <div className="card flex flex-wrap items-center gap-2 p-2">
        <input className="input w-64" placeholder="Search expression, notes, tags…" value={search} onChange={(e) => setSearch(e.target.value)} />
        <Select value={f.status ?? ""} onChange={(x) => setFilter("status", x)} options={[{ value: "", label: "Any status" }, "PASS", "FAIL", "PENDING", "UNSCORED"]} />
        <Select value={f.origin ?? ""} onChange={(x) => setFilter("origin", x)} options={[{ value: "", label: "Any origin" }, ...Object.keys(facets?.origin ?? {})]} />
        <Select value={f.family ?? ""} onChange={(x) => setFilter("family", x)} options={[{ value: "", label: "Any idea" }, ...Object.keys(facets?.idea ?? {})]} />
        <Select value={f.category ?? ""} onChange={(x) => setFilter("category", x)} options={[{ value: "", label: "Any data" }, ...Object.keys(facets?.category ?? {})]} />
        <input className="input w-24 tnum" type="number" step={0.1} placeholder="min Sharpe" value={f.min_sharpe ?? ""} onChange={(e) => setFilter("min_sharpe", e.target.value === "" ? undefined : Number(e.target.value))} />
        <Toggle checked={!!f.starred} onChange={(x) => setFilter("starred", x || undefined)} label="Starred" />
        <Toggle checked={!!f.submitted} onChange={(x) => setFilter("submitted", x || undefined)} label="Submitted" />
        <Toggle checked={f.local === true} onChange={(x) => setFilter("local", x || undefined)} label="Local only" />
        <span className="ml-auto text-[12px] text-muted">
          {isFetching && <Spinner />} {data?.total ?? 0} alphas{data && data.total > rows.length ? ` (showing ${rows.length})` : ""}
        </span>
      </div>
      <div className="card flex flex-wrap items-center gap-1.5 p-2">
        <span className="mr-1 text-[12px] text-ink2">{ids.length} selected</span>
        <button className="btn" disabled={!ids.length} onClick={() => setModal("export")}><Upload size={13} /> Export</button>
        <button className="btn" disabled={ids.length < 2} onClick={() => setModal("corr")}><Grid3x3 size={13} /> Correlation</button>
        <button className="btn" disabled={ids.length < 2} onClick={() => setModal("combine")}><Layers size={13} /> Combine</button>
        <button className="btn" disabled={!ids.length} onClick={() => bulk("optimize")}><SlidersHorizontal size={13} /> Optimize settings</button>
        <button className="btn" disabled={!ids.length} onClick={() => bulk("gp")}><Dna size={13} /> Refine with GP</button>
        <button className="btn" disabled={!ids.length} onClick={() => bulk("star")}><Star size={13} /> Star</button>
        <button className="btn" disabled={!ids.length} onClick={() => bulk("submit")}>Mark submitted</button>
        <button className="btn" disabled={!ids.length} onClick={() => bulk("unsubmit")}>Unmark</button>
        <button className="btn btn-danger" disabled={!ids.length} onClick={() => bulk("delete")}><Trash2 size={13} /> Delete</button>
        {ids.length > 0 && <button className="btn btn-ghost" onClick={() => setSel(new Set())}>Clear</button>}
      </div>
      <div className="card flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="grid items-center gap-2 border-b border-line bg-[var(--surface-2)] px-2 py-1.5 text-[10.5px] font-semibold tracking-wide text-muted" style={{ gridTemplateColumns: COLS }}>
          <input type="checkbox" checked={allOn} onChange={() => setSel(allOn ? new Set() : new Set(rows.map((r) => r.id)))} aria-label="Select all" />
          <span>★</span>
          {sortHeader("id", "#", false)}
          <span>STATUS</span>
          {sortHeader("sharpe", "Sharpe")}
          {sortHeader("fitness", "Fitness")}
          {sortHeader("turnover", "TO")}
          {sortHeader("returns", "Returns")}
          {sortHeader("os_sharpe", "OS Sh.")}
          {sortHeader("sub_sharpe", "Sub Sh.")}
          {sortHeader("pass_prob", "P(pass)")}
          {sortHeader("max_corr", "Corr")}
          <span>IDEA</span>
          <span>BRAIN</span>
          <span>EXPRESSION</span>
        </div>
        {rows.length === 0 && !isFetching ? (
          <Empty title="No alphas match">Save alphas from the Studio, or start a mining job from the Miner.</Empty>
        ) : (
          <div ref={parent} className="min-h-0 flex-1 overflow-auto">
            <div style={{ height: v.getTotalSize(), position: "relative" }}>
              {v.getVirtualItems().map((it) => {
                const a: Alpha = rows[it.index];
                return (
                  <div
                    key={a.id}
                    className={clsx("tnum absolute left-0 grid w-full cursor-pointer items-center gap-2 border-b border-line px-2 text-[12px] hover:bg-[var(--surface-2)]", sel.has(a.id) && "bg-[var(--surface-2)]")}
                    style={{ top: it.start, height: it.size, gridTemplateColumns: COLS }}
                    onClick={() => setParams({ open: String(a.id) })}
                  >
                    <input type="checkbox" checked={sel.has(a.id)} onClick={(e) => e.stopPropagation()} onChange={() => toggle(a.id)} aria-label={`Select ${a.id}`} />
                    <span className={a.starred ? "text-[var(--warn-text)]" : "text-muted"}>{a.starred ? "★" : "☆"}</span>
                    <span className="text-muted">{a.id}</span>
                    <StatusBadge status={a.status_local} compact />
                    <span className="text-right">{fmt.num(a.sharpe)}</span>
                    <span className="text-right">{fmt.num(a.fitness)}</span>
                    <span className="text-right">{fmt.pct(a.turnover, 0)}</span>
                    <span className="text-right">{fmt.pct(a.returns, 1)}</span>
                    <span className="text-right">{fmt.num(a.os_sharpe)}</span>
                    <span className="text-right">{fmt.num(a.sub_sharpe)}</span>
                    <span className="text-right">{a.pass_prob != null ? `${Math.round(a.pass_prob * 100)}%` : "—"}</span>
                    <span className="text-right">{fmt.num(a.max_corr)}</span>
                    <span className="truncate text-ink2">{a.idea}</span>
                    <span className="truncate text-ink2">{a.submitted ? "submitted" : a.status_brain}</span>
                    <span className="mono truncate text-ink2" title={a.expr}>{a.expr}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
      <ExportModal ids={ids} open={modal === "export"} onOpenChange={(o) => setModal(o ? "export" : "")} />
      <CorrelationModal ids={ids} open={modal === "corr"} onOpenChange={(o) => setModal(o ? "corr" : "")} />
      <CombineModal ids={ids} open={modal === "combine"} onOpenChange={(o) => setModal(o ? "combine" : "")} />
      <AlphaDrawer id={openId} onClose={() => { params.delete("open"); setParams(params); }} />
    </div>
  );
}
