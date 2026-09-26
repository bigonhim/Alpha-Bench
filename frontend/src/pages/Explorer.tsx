import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, Pickaxe, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Card, Select, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { useCatalog } from "../lib/hooks";
import { useUI } from "../lib/store";

function firstExpansion(t: any): string {
  let e: string = t.expr;
  for (const [name, spec] of Object.entries<any>(t.slots || {})) {
    let v = "";
    if (spec?.fields) v = spec.fields[0];
    else if (spec?.windows) v = String(spec.windows[0]);
    else if (spec?.groups) v = spec.groups[0] === "capbucket" ? 'bucket(rank(cap), range="0.1,1,0.1")' : spec.groups[0];
    else if (spec?.values) v = spec.values[0];
    else if (spec?.select) v = "close";
    e = e.split(`{${name}}`).join(v);
  }
  return e;
}

function useTry() {
  const setStudio = useUI((s) => s.setStudio);
  const nav = useNavigate();
  return (expr: string) => {
    setStudio(expr);
    nav("/studio");
  };
}

function Operators() {
  const { data } = useCatalog();
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const tryIt = useTry();
  const ops = useMemo(() => (data?.operators ?? []).filter((o) => (!cat || o.category === cat) && (o.name.includes(q) || o.doc.toLowerCase().includes(q.toLowerCase()))), [data, q, cat]);
  const cats = [...new Set((data?.operators ?? []).map((o) => o.category))];
  return (
    <div className="p-2">
      <div className="mb-2 flex gap-2">
        <input className="input w-72" placeholder="Search operators…" value={q} onChange={(e) => setQ(e.target.value)} />
        <Select value={cat} onChange={setCat} options={[{ value: "", label: "All categories" }, ...cats]} />
        <span className="ml-auto self-center text-[12px] text-muted">
          {ops.length} operators · {ops.filter((o) => o.local).length} simulate locally
        </span>
      </div>
      <div className="max-h-[calc(100vh-190px)] overflow-auto">
        <table className="tbl">
          <thead>
            <tr>
              <th>Signature</th>
              <th>Category</th>
              <th>Local</th>
              <th>Description</th>
              <th>Example</th>
            </tr>
          </thead>
          <tbody>
            {ops.map((o) => (
              <tr key={o.name}>
                <td className="mono whitespace-nowrap text-[11.5px]">{o.signature}</td>
                <td className="whitespace-nowrap text-ink2">{o.category}{o.level === "consultant" ? " · C" : ""}</td>
                <td>{o.local ? "yes" : <span className="text-muted">BRAIN-only</span>}</td>
                <td className="text-ink2">{o.doc}</td>
                <td>
                  <button className="mono text-left text-[11.5px] text-[var(--accent)] hover:underline" onClick={() => tryIt(o.example)} title="Try in Studio">
                    {o.example}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Fields() {
  const { data } = useCatalog();
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [avail, setAvail] = useState("");
  const fields = useMemo(
    () =>
      (data?.fields ?? []).filter(
        (f) => (!cat || f.category === cat) && (!avail || (avail === "local" ? f.available : !f.available)) && (f.id.includes(q) || (f.description || "").toLowerCase().includes(q.toLowerCase())),
      ),
    [data, q, cat, avail],
  );
  const cats = [...new Set((data?.fields ?? []).map((f) => f.category))];
  return (
    <div className="p-2">
      <div className="mb-2 flex gap-2">
        <input className="input w-72" placeholder="Search fields…" value={q} onChange={(e) => setQ(e.target.value)} />
        <Select value={cat} onChange={setCat} options={[{ value: "", label: "All categories" }, ...cats]} />
        <Select value={avail} onChange={setAvail} options={[{ value: "", label: "Local + BRAIN-only" }, { value: "local", label: "Available locally" }, { value: "brain", label: "BRAIN-only" }]} />
        <span className="ml-auto self-center text-[12px] text-muted">{fields.length} fields</span>
      </div>
      <div className="max-h-[calc(100vh-190px)] overflow-auto">
        <table className="tbl">
          <thead>
            <tr>
              <th>Field</th>
              <th>Dataset</th>
              <th>Category</th>
              <th>Type</th>
              <th>Unit</th>
              <th>Local</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => (
              <tr key={f.id}>
                <td className="mono text-[11.5px]">{f.id}</td>
                <td className="text-ink2">{f.dataset}</td>
                <td className="text-ink2">{f.category}</td>
                <td className="text-ink2">{f.type}</td>
                <td className="text-ink2">{f.unit}</td>
                <td>{f.available ? "yes" : <span className="text-muted">no</span>}</td>
                <td className="text-ink2">
                  {f.description}
                  {!f.verified && <span className="text-muted"> (id unverified)</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const TEMPLATE_EXAMPLE = `{
  "id": "my_value_idea",
  "idea": "value",
  "category": "fundamental",
  "horizon": "long",
  "expr": "group_rank(ts_backfill({num}, 120) / cap, {g})",
  "slots": {"num": {"fields": ["income", "sales"]}, "g": {"groups": ["industry", "subindustry"]}},
  "settings": {"decay": [0, 4], "neutralization": ["SUBINDUSTRY"]},
  "rationale": "Cheap relative to fundamentals within peers."
}`;

function Templates() {
  const { data } = useQuery({ queryKey: ["templates"], queryFn: api.templates });
  const [q, setQ] = useState("");
  const [draft, setDraft] = useState(TEMPLATE_EXAMPLE);
  const tryIt = useTry();
  const nav = useNavigate();
  const qc = useQueryClient();
  const rows = (data ?? []).filter((t: any) => t.id.includes(q) || t.idea.includes(q) || t.expr.includes(q));
  return (
    <div className="grid grid-cols-1 gap-3 p-2 xl:grid-cols-3">
      <div className="xl:col-span-2">
        <div className="mb-2 flex gap-2">
          <input className="input w-72" placeholder="Search templates…" value={q} onChange={(e) => setQ(e.target.value)} />
          <span className="ml-auto self-center text-[12px] text-muted">{rows.length} templates</span>
        </div>
        <div className="max-h-[calc(100vh-190px)] overflow-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th />
                <th>Template</th>
                <th>Idea / data / horizon</th>
                <th className="text-right">Expansions</th>
                <th>Expression</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((t: any) => (
                <tr key={t.id}>
                  <td className="whitespace-nowrap">
                    <button className="btn btn-ghost !px-1.5" onClick={() => tryIt(firstExpansion(t))} title="Open first expansion in Studio">
                      <FlaskConical size={13} />
                    </button>
                    <button className="btn btn-ghost !px-1.5" onClick={() => nav(`/miner?method=templates&family=${t.idea}`)} title="Mine this family">
                      <Pickaxe size={13} />
                    </button>
                    {t.source === "user" && (
                      <button
                        className="btn btn-ghost btn-danger !px-1.5"
                        onClick={async () => {
                          await api.deleteTemplate(t.id);
                          qc.invalidateQueries({ queryKey: ["templates"] });
                        }}
                        title="Delete user template"
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </td>
                  <td className="mono text-[11.5px]">
                    {t.id}
                    {t.source === "user" && <span className="chip ml-1">user</span>}
                  </td>
                  <td className="whitespace-nowrap text-ink2">
                    {t.idea} · {t.category} · {t.horizon}
                  </td>
                  <td className="tnum text-right">{t.brain_only ? <span className="text-muted">BRAIN-only</span> : `${t.n_local}/${t.n_expansions}`}</td>
                  <td>
                    <div className="mono max-w-[440px] truncate text-[11.5px]" title={t.expr}>
                      {t.expr}
                    </div>
                    <div className="max-w-[440px] truncate text-[11px] text-muted" title={t.rationale}>
                      {t.rationale}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <Card title="Add / update a user template">
        <div className="mb-2 text-[12px] text-muted">
          Slots: <code>fields</code>, <code>windows</code>, <code>groups</code> (incl. <code>capbucket</code>), <code>values</code>, or <code>select</code> (catalog filter). Saved to runtime/templates_user.yaml.
        </div>
        <textarea className="input mono h-80 w-full text-[11.5px]" value={draft} onChange={(e) => setDraft(e.target.value)} />
        <div className="mt-2 flex justify-end">
          <button
            className="btn btn-primary"
            onClick={async () => {
              try {
                await api.saveTemplate(JSON.parse(draft));
                toast.success("Template saved");
                qc.invalidateQueries({ queryKey: ["templates"] });
              } catch (e) {
                toast.error(String((e as Error).message));
              }
            }}
          >
            Save template
          </button>
        </div>
      </Card>
    </div>
  );
}

function Alpha101() {
  const { data } = useQuery({ queryKey: ["alpha101"], queryFn: api.alpha101 });
  const tryIt = useTry();
  return (
    <div className="p-2">
      <div className="mb-2 text-[12px] text-muted">Translations of Kakushadze's "101 Formulaic Alphas" (2015) into BRAIN syntax. Verify against the paper before relying on them; widely known alphas are likely to be correlated with existing BRAIN submissions.</div>
      <table className="tbl">
        <thead>
          <tr>
            <th>#</th>
            <th>Expression</th>
          </tr>
        </thead>
        <tbody>
          {(data ?? []).map((a) => (
            <tr key={a.id}>
              <td className="mono">{a.id}</td>
              <td>
                <button className="mono text-left text-[11.5px] text-[var(--accent)] hover:underline" onClick={() => tryIt(a.expr)}>
                  {a.expr}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Explorer() {
  const [tab, setTab] = useState("ops");
  return (
    <div className="p-3">
      <div className="card">
        <Tabs
          value={tab}
          onChange={setTab}
          tabs={[
            { id: "ops", label: "Operators", content: <Operators /> },
            { id: "fields", label: "Data fields", content: <Fields /> },
            { id: "tmpl", label: "Templates", content: <Templates /> },
            { id: "a101", label: "101 Alphas", content: <Alpha101 /> },
          ]}
        />
      </div>
    </div>
  );
}
