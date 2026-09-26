import { create } from "zustand";
import type { GPStat, JobResultRow, JobSnapshot, ReReport, Settings } from "./types";

export const DEFAULT_SETTINGS: Settings = {
  instrumentType: "EQUITY",
  region: "USA",
  universe: "TOP3000",
  delay: 1,
  decay: 0,
  neutralization: "SUBINDUSTRY",
  truncation: 0.08,
  pasteurization: "ON",
  unitHandling: "VERIFY",
  nanHandling: "OFF",
  language: "FASTEXPR",
  visualization: false,
};

function load<T>(k: string, d: T): T {
  try {
    const v = localStorage.getItem(k);
    return v ? (JSON.parse(v) as T) : d;
  } catch {
    return d;
  }
}
function save(k: string, v: unknown) {
  try {
    localStorage.setItem(k, JSON.stringify(v));
  } catch {
    /* storage unavailable */
  }
}

export interface HistoryItem {
  text: string;
  settings: Settings;
  sharpe?: number;
  fitness?: number;
  turnover?: number;
  status?: string;
  at: number;
}

interface UIState {
  theme: "dark" | "light";
  sidebar: boolean;
  studioText: string;
  studioSettings: Settings;
  history: HistoryItem[];
  paletteOpen: boolean;
  studioReJob: number | null;
  setTheme: (t: "dark" | "light") => void;
  toggleSidebar: () => void;
  setStudio: (text?: string, settings?: Partial<Settings>) => void;
  pushHistory: (h: HistoryItem) => void;
  setPalette: (open: boolean) => void;
  setStudioReJob: (id: number | null) => void;
}

export const useUI = create<UIState>((set, get) => ({
  theme: load<"dark" | "light">("af-theme", "dark"),
  sidebar: load("af-sidebar", true),
  studioText: load("af-studio-text", "rank(-ts_delta(close, 5))"),
  studioSettings: { ...DEFAULT_SETTINGS, ...load<Partial<Settings>>("af-studio-settings", {}) },
  history: load<HistoryItem[]>("af-history", []),
  paletteOpen: false,
  studioReJob: load<number | null>("af-studio-rejob", null),
  setTheme: (t) => {
    document.documentElement.setAttribute("data-theme", t);
    save("af-theme", t);
    try {
      localStorage.setItem("af-theme", t);
    } catch {
      /* ignore */
    }
    set({ theme: t });
  },
  toggleSidebar: () => {
    const v = !get().sidebar;
    save("af-sidebar", v);
    set({ sidebar: v });
  },
  setStudio: (text, settings) => {
    const s = settings ? { ...get().studioSettings, ...settings } : get().studioSettings;
    const t = text ?? get().studioText;
    save("af-studio-text", t);
    save("af-studio-settings", s);
    set({ studioText: t, studioSettings: s });
  },
  pushHistory: (h) => {
    const list = [h, ...get().history.filter((x) => !(x.text === h.text && JSON.stringify(x.settings) === JSON.stringify(h.settings)))].slice(0, 60);
    save("af-history", list);
    set({ history: list });
  },
  setPalette: (open) => set({ paletteOpen: open }),
  setStudioReJob: (id) => {
    save("af-studio-rejob", id);
    set({ studioReJob: id });
  },
}));

interface JobsState {
  jobs: Record<number, JobSnapshot>;
  results: Record<number, JobResultRow[]>;
  gp: Record<number, GPStat[]>;
  re: Record<number, ReReport>;
  connected: boolean;
  lastEvent: number;
  upsertJob: (j: JobSnapshot, results?: JobResultRow[]) => void;
  pushGP: (jobId: number, s: GPStat) => void;
  setReport: (jobId: number, r: ReReport) => void;
  setConnected: (c: boolean) => void;
}

export const useJobs = create<JobsState>((set) => ({
  jobs: {},
  results: {},
  gp: {},
  re: {},
  connected: false,
  lastEvent: 0,
  upsertJob: (j, results) =>
    set((st) => {
      const jobs = { ...st.jobs, [j.id]: { ...(st.jobs[j.id] || {}), ...j } };
      let res = st.results;
      if (results && results.length) {
        const prev = st.results[j.id] || [];
        res = { ...st.results, [j.id]: [...results, ...prev].slice(0, 2000) };
      }
      return { jobs, results: res, lastEvent: Date.now() };
    }),
  pushGP: (jobId, s) =>
    set((st) => ({ gp: { ...st.gp, [jobId]: s.generation === 1 ? [s] : [...(st.gp[jobId] || []), s].slice(-200) } })),
  setReport: (jobId, r) => set((st) => ({ re: { ...st.re, [jobId]: r } })),
  setConnected: (c) => set({ connected: c }),
}));
