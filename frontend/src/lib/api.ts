import type { Alpha, Analysis, Catalog, IdeaSpec, JobSnapshot, ReReport, Settings, SimResult, Status } from "./types";

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      msg = j.detail || msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  const ct = res.headers.get("content-type") || "";
  return (ct.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

const get = <T>(u: string) => req<T>("GET", u);
const post = <T>(u: string, b?: unknown) => req<T>("POST", u, b ?? {});

export interface ListParams {
  search?: string;
  status?: string;
  origin?: string;
  family?: string;
  category?: string;
  submitted?: boolean;
  starred?: boolean;
  local?: boolean;
  min_sharpe?: number;
  min_fitness?: number;
  tag?: string;
  job_id?: number;
  sort?: string;
  desc?: boolean;
  limit?: number;
  offset?: number;
}

function qs(p: Record<string, unknown>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(p)) if (v !== undefined && v !== "" && v !== null) u.set(k, String(v));
  return u.toString();
}

export interface DoctorFix {
  label: string;
  reason: string;
  expr: string;
  settings: Settings;
  kind: string;
  metrics: any;
  os: any;
  failed: string[];
  n_failed: number;
  delta: Record<string, number>;
}

export const api = {
  status: () => get<Status>("/api/status"),
  catalog: () => get<Catalog>("/api/catalog"),
  parse: (text: string) => post<Analysis>("/api/parse", { text }),
  simulate: (text: string, settings: Partial<Settings>, extras = false) =>
    post<SimResult>("/api/simulate", { text, settings, extras }),
  extras: (text: string, settings: Partial<Settings>) =>
    post<{ ok: boolean; extras: any; checks: any; pass_prob: number }>("/api/extras", { text, settings }),
  doctor: (text: string, settings: Partial<Settings>) =>
    post<{ ok: boolean; fixes: DoctorFix[]; base_failed: string[]; base_warnings: string[]; base_metrics: any; base?: any }>(
      "/api/doctor",
      { text, settings },
    ),
  sweep: async (
    text: string,
    settings: Partial<Settings>,
    grid: Record<string, unknown> | undefined,
    onEvent: (e: any) => void,
    signal?: AbortSignal,
  ) => {
    const res = await fetch("/api/sweep", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, settings, grid }),
      signal,
    });
    if (!res.body) return;
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i: number;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
  },
  explain: (text: string) => post<{ description: any; tags: any }>("/api/explain", { text }),
  alphas: (p: ListParams) => get<{ total: number; rows: Alpha[] }>(`/api/alphas?${qs(p as Record<string, unknown>)}`),
  facets: () => get<Record<string, Record<string, number>>>("/api/alphas/facets"),
  alpha: (id: number) => get<{ alpha: Alpha; result: SimResult | null; brain: any[] }>(`/api/alphas/${id}`),
  save: (text: string, settings: Partial<Settings>, extra?: { tags?: string[]; notes?: string; origin?: string }) =>
    post<{ ok: boolean; id: number; status?: string }>("/api/alphas", { text, settings, ...extra }),
  patch: (id: number, fields: Record<string, unknown>) => req<{ ok: boolean; alpha: Alpha }>("PATCH", `/api/alphas/${id}`, fields),
  remove: (ids: number[]) => post<{ ok: boolean; deleted: number }>("/api/alphas/delete", { ids }),
  submitted: (ids: number[], submitted: boolean) => post("/api/alphas/submitted", { ids, submitted }),
  correlation: (ids: number[]) => post<{ ids: number[]; matrix: number[][]; missing: number[] }>("/api/correlation", { ids }),
  combine: (ids: number[], method: string) => post<any>("/api/combine", { ids, method }),
  exportText: (ids: number[], format: string, batch = 10) => post<string>("/api/export", { ids, format, batch }),
  importResults: (text: string, format: string, mark_submitted: boolean) =>
    post<any>("/api/import", { text, format, mark_submitted }),
  calibration: () => get<any>("/api/calibration"),
  templates: () => get<any[]>("/api/templates"),
  saveTemplate: (t: any) => post<any>("/api/templates", t),
  deleteTemplate: (id: string) => req<any>("DELETE", `/api/templates/${encodeURIComponent(id)}`),
  alpha101: () => get<{ id: string; expr: string }[]>("/api/alpha101"),
  importFields: (text: string) => post<{ ok: boolean; imported: number }>("/api/catalog/fields/import", { text }),
  checks: () => get<any>("/api/checks"),
  saveChecks: (cfg: any) => post<any>("/api/checks", cfg),
  settings: () => get<Record<string, any>>("/api/settings"),
  saveSettings: (u: Record<string, unknown>) => post<Record<string, any>>("/api/settings", u),
  startJob: (kind: string, config: Record<string, unknown>) => post<{ ok: boolean; id: number }>("/api/jobs", { kind, config }),
  jobs: () => get<{ jobs: JobSnapshot[] }>("/api/jobs"),
  job: (id: number) => get<JobSnapshot>(`/api/jobs/${id}`),
  jobAction: (id: number, action: string) => post<{ ok: boolean }>(`/api/jobs/${id}/${action}`),
  reengineer: (id: number) => get<ReReport>(`/api/reengineer/${id}`),
  dashboard: () => get<any>("/api/dashboard"),
  coverage: () => get<{ coverage: Record<string, number>; info: any }>("/api/data/coverage"),
  buildData: (cfg: Record<string, unknown>) => post<{ ok: boolean; id: number }>("/api/data/build", cfg),
  bandit: () => get<any[]>("/api/bandit"),
  forgeInterpret: (text: string, opts: { families?: string[]; horizon?: string; settings?: Partial<Settings> } = {}) =>
    post<IdeaSpec>("/api/forge/interpret", { text, ...opts }),
};

export async function copyText(t: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(t);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = t;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

export function downloadText(name: string, text: string, mime = "text/plain"): void {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
