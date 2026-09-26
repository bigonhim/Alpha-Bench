export type Severity = "error" | "warning" | "info";

export interface Diagnostic {
  severity: Severity;
  message: string;
  start: number;
  end: number;
}

export interface Tags {
  idea: string;
  category: string;
  horizon: string;
}

export interface Analysis {
  ok: boolean;
  diagnostics: Diagnostic[];
  fields: string[];
  operators: string[];
  categories: string[];
  lookback: number;
  depth: number;
  size: number;
  op_count: number;
  local: boolean;
  brain_only_reasons: string[];
  canonical: string;
  canon_hash: string;
  pretty: string;
  tags?: Tags;
}

export interface Settings {
  instrumentType: string;
  region: string;
  universe: string;
  delay: number;
  decay: number;
  neutralization: string;
  truncation: number;
  pasteurization: string;
  unitHandling: string;
  nanHandling: string;
  language?: string;
  visualization?: boolean;
}

export interface Metrics {
  sharpe: number;
  fitness: number;
  returns: number;
  turnover: number;
  drawdown: number;
  margin_bps: number;
  pnl: number;
  long_count: number;
  short_count: number;
  max_weight: number;
  days: number;
}

export type CheckResult = "PASS" | "FAIL" | "WARNING" | "PENDING";

export interface Check {
  name: string;
  result: CheckResult;
  value: number | null;
  limit: number | null;
  message: string;
}

export interface Checks {
  checks: Check[];
  status: "PASS" | "FAIL" | "PENDING";
  failed: string[];
  warnings: string[];
  robust: boolean;
}

export interface YearRow extends Metrics {
  year: number;
  period: "IS" | "OS";
}

export interface Series {
  dates: string[];
  cum_pnl: number[];
  drawdown: number[];
  cum_long: number[];
  cum_short: number[];
  turnover: number[];
  rolling_sharpe: (number | null)[];
  long_count: number[];
  short_count: number[];
}

export interface Extras {
  sub_universe?: string;
  sub_sharpe?: number;
  sub_size?: number;
  univ_size?: number;
  sub_metrics?: Metrics;
  variants?: { expr: string; sharpe: number; fitness: number }[];
  stability?: number | null;
}

export interface Description {
  summary: string;
  idea: string;
  data: string;
  operators: string[];
  tags: Tags;
}

export interface SimResult {
  ok: boolean;
  brain_only?: boolean;
  message?: string;
  error?: string;
  analysis: Analysis;
  settings: Settings;
  local_universe?: string;
  universe_size?: number;
  metrics?: { is: Metrics; os?: Metrics; brain?: Metrics; all?: Metrics };
  yearly?: YearRow[];
  series?: Series;
  periods?: Record<string, [number, number]>;
  checks?: Checks;
  extras?: Extras | null;
  sector_pnl?: Record<string, number>;
  top_names?: { best: [string, number][]; worst: [string, number][] };
  correlation?: { top: { alpha_id: number; corr: number }[] };
  pass_prob?: number;
  expected_brain_sharpe?: number | null;
  dsr?: { dsr: number; expected_max_sharpe: number; haircut_sharpe: number };
  description?: Description;
  elapsed_ms?: number;
  data?: { source: string; version: string };
}

export interface Alpha {
  id: number;
  expr: string;
  canon: string;
  canon_hash: string;
  settings: Settings;
  settings_key: string;
  origin: string;
  family?: string;
  idea?: string;
  category?: string;
  horizon?: string;
  template_id?: string;
  parents?: number[];
  tags: string[];
  notes?: string;
  description?: Description | null;
  local: number;
  brain_only_reasons?: string[];
  status_local: string;
  robust?: number;
  status_brain?: string;
  submitted: number;
  starred: number;
  job_id?: number;
  sharpe?: number;
  fitness?: number;
  turnover?: number;
  returns?: number;
  drawdown?: number;
  margin?: number;
  os_sharpe?: number;
  sub_sharpe?: number;
  max_corr?: number;
  pass_prob?: number;
  complexity?: number;
  failed: string[];
  created_at: string;
  updated_at: string;
}

export interface OpParam {
  name: string;
  kind: string;
  default?: unknown;
  choices?: string[];
}

export interface OpSpec {
  name: string;
  category: string;
  params: OpParam[];
  returns: string;
  doc: string;
  example: string;
  level: string;
  variadic: boolean;
  signature: string;
  local: boolean;
}

export interface FieldSpec {
  id: string;
  dataset: string;
  category: string;
  type: string;
  unit: string;
  description: string;
  local: boolean;
  verified: boolean;
  available: boolean;
  source?: string;
}

export interface Catalog {
  operators: OpSpec[];
  fields: FieldSpec[];
  settings_options: {
    regions: string[];
    universes: string[];
    delays: number[];
    neutralizations: string[];
    neutralizations_local: string[];
    pasteurization: string[];
    nanHandling: string[];
  };
}

export interface JobSnapshot {
  id: number;
  kind: string;
  status: string;
  progress: Record<string, any>;
  stats: Record<string, any>;
  config: Record<string, any>;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  error?: string | null;
}

export interface JobResultRow {
  id: number;
  expr: string;
  sharpe?: number;
  fitness?: number;
  turnover?: number;
  returns?: number;
  status: string;
  pass_prob?: number;
  os_sharpe?: number;
  idea?: string;
  origin?: string;
  settings?: Settings;
  failed?: string[];
}

export interface GPStat {
  generation: number;
  evaluations: number;
  hof: number;
  best_fitness: number | null;
  median_fitness: number | null;
  front: { sharpe: number; turnover: number; fitness: number; novelty: number; expr: string }[];
}

/** Re-engineer job report (GET /api/reengineer/{job_id} and live "reengineer" events). */
export type ReMetrics = Partial<Pick<Metrics, "sharpe" | "fitness" | "turnover" | "returns" | "drawdown" | "max_weight" | "margin_bps">>;

export interface ReStep {
  stage: string;
  stage_title: string;
  label: string;
  reason: string;
  expr: string;
  settings: Partial<Settings>;
  metrics: ReMetrics;
  score: number;
  score_delta: number;
  delta: { sharpe: number; fitness: number; turnover: number; returns: number };
}

export interface ReDiagnosis {
  code: string;
  severity: "bad" | "warn" | "info";
  title: string;
  detail: string;
  stage: string;
}

export interface ReCheck {
  name: string;
  result: CheckResult;
  value: number | null;
  limit: number | null;
  message: string;
}

export interface ReFinalRow {
  expr: string;
  settings: Partial<Settings>;
  is: ReMetrics;
  os: ReMetrics;
  sub_sharpe?: number | null;
  stability?: number | null;
  status: string;
  failed: string[];
  warnings: string[];
  checks: ReCheck[];
  pass_prob?: number;
  score: number;
  final: number;
  alpha_id?: number | null;
  recipe?: string;
  lineage?: ReStep[];
  size?: number;
}

export interface ReReport {
  job_id: number;
  status: "running" | "done" | "stopped";
  config: Record<string, any>;
  original?: ReFinalRow & { parts?: any };
  diagnosis?: ReDiagnosis[];
  exposures?: Record<string, number>;
  best?: { expr: string; settings: Partial<Settings>; metrics: ReMetrics; score: number; steps: number; lineage: ReStep[] };
  beam?: { expr: string; settings: Partial<Settings>; metrics: ReMetrics; score: number; steps: number }[];
  stages: { pass: number; stage: string; title: string; tried: number; accepted: number; best_label: string | null; gain: number; elapsed_s: number }[];
  trajectory: { label: string; stage: string; pass: number; score: number; sharpe: number | null; fitness: number | null; turnover: number | null; evals: number }[];
  n_evaluated?: number;
  elapsed_s?: number;
  improvement?: number;
  final: {
    n_evaluated: number;
    elapsed_s: number;
    original: ReFinalRow;
    champion: ReFinalRow | null;
    alternatives: ReFinalRow[];
    series: { dates: string[]; original: number[]; champion: number[]; os_start: string | null } | null;
    exposures_after?: Record<string, number>;
    kinship?: number | null;
    kinship_flipped?: boolean;
    verdict: { level: "good" | "warn" | "bad"; title: string; text: string; os_is_ratio?: number };
  } | null;
}

export interface Status {
  data: {
    version: string;
    source: string;
    start: string;
    end: string;
    T: number;
    N: number;
    fields: string[];
    groups: Record<string, number>;
    universes: string[];
    built?: string;
  };
  periods: { is_start: string; os_start: string; end: string; brain_start: string };
  cache: { entries: number; mb: number; hits: number; misses: number; hit_rate: number };
  db: Record<string, number | null>;
  real_data_available: boolean;
  settings: Record<string, any>;
  warm: boolean;
  jobs_running: JobSnapshot[];
  defaults: Settings;
  universe_map: Record<string, string>;
}

/* ---------------------------------------------------------------- Idea Forge */

export interface IdeaFamily {
  id: string;
  label: string;
  score: number;
  direction: number;
  flipped: boolean;
  stated: boolean;
  amount: string;
  brain_only: boolean;
}

export interface IdeaSpec {
  text: string;
  families: IdeaFamily[];
  fields: { id: string; description: string; local: boolean; mentioned: boolean }[];
  horizon: "short" | "medium" | "long";
  horizon_stated: boolean;
  windows: number[];
  groups: string[];
  conditions: { id: string; label: string }[];
  smooth: boolean;
  interaction: boolean;
  subset_families: string[];
  seeds: string[];
  notes: string[];
  warnings: string[];
  confidence: number;
  preview?: { expr: string; label: string; family: string; origin: string }[];
  templates?: { id: string; idea: string; rationale: string; score: number }[];
  all_families?: { id: string; label: string }[];
}

export interface ForgeAlpha {
  id: number | null;
  expr: string;
  settings: Settings;
  family: string;
  family_label: string;
  stage: string;
  lineage: string[];
  fidelity: number;
  score: number;
  sharpe?: number;
  fitness?: number;
  turnover?: number;
  returns?: number;
  drawdown?: number;
  margin?: number;
  os_sharpe?: number;
  os_fitness?: number;
  sub_sharpe?: number;
  stability?: number | null;
  status: string;
  failed: string[];
  warnings: string[];
  pass_prob: number;
  expected_brain_sharpe?: number | null;
  complexity?: number;
  sign: number;
  foreign: string[];
  missing: string[];
}

export interface ForgeResult {
  champion: ForgeAlpha | null;
  runners: ForgeAlpha[];
  faithful?: ForgeAlpha | null;
  message: string;
  hypothesis?: { holds: boolean | null; text: string; sharpe_hyp?: number; sharpe_rev?: number };
  brain_only: { id: number; expr: string; label: string; settings: Settings }[];
  evaluated: number;
  pool: number;
  seconds: number;
  idea: string;
}

export interface ForgeStage {
  status: "pending" | "running" | "done" | "skipped";
  started?: number;
  seconds?: number;
  candidates?: number;
  note?: string;
  best?: ForgeRow;
}

export interface ForgeRow {
  expr: string;
  settings: Settings;
  family: string;
  stage: string;
  sharpe?: number;
  fitness?: number;
  turnover?: number;
  returns?: number;
  fidelity: number;
  score: number;
}
