import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, BookOpen, Cloud, Database, FlaskConical, Import, LayoutDashboard, Library, Moon, PanelLeft, Pickaxe, Search, Settings, Sun, Wand2 } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { api } from "../lib/api";
import { useStatus } from "../lib/hooks";
import { clsx } from "../lib/format";
import { useJobs, useUI } from "../lib/store";
import { Tip } from "./ui";

export const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/studio", label: "Studio", icon: FlaskConical },
  { to: "/forge", label: "Idea Forge", icon: Wand2 },
  { to: "/miner", label: "Miner", icon: Pickaxe },
  { to: "/library", label: "Library", icon: Library },
  { to: "/brain", label: "BRAIN", icon: Cloud },
  { to: "/import", label: "BRAIN import", icon: Import },
  { to: "/explorer", label: "Explorer", icon: BookOpen },
  { to: "/data", label: "Data", icon: Database },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function Layout({ children }: { children: ReactNode }) {
  const { sidebar, toggleSidebar, theme, setTheme, setPalette } = useUI();
  const { data: status } = useStatus();
  const connected = useJobs((s) => s.connected);
  const running = useJobs((s) => Object.values(s.jobs).filter((j) => j.status === "running" || j.status === "paused").length);
  const loc = useLocation();
  const title = NAV.find((n) => (n.to === "/" ? loc.pathname === "/" : loc.pathname.startsWith(n.to)))?.label ?? "";
  const { data: brain } = useQuery({ queryKey: ["brain-status"], queryFn: api.brain.status, refetchInterval: 120_000, retry: 0 });
  const demo = status?.data?.source === "demo";
  return (
    <div className="flex h-full">
      <aside className={clsx("flex shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface-1)] transition-all", sidebar ? "w-52" : "w-14")}>
        <div className="flex h-12 items-center gap-2 px-3">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="7" fill="var(--surface-3)" />
            <path d="M6 23 L12 15 L17 19 L26 8" fill="none" stroke="var(--series-1)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="26" cy="8" r="3" fill="var(--status-good)" />
          </svg>
          {sidebar && <span className="text-[14px] font-semibold tracking-tight">Alpha Foundry</span>}
        </div>
        <nav className="flex flex-1 flex-col gap-0.5 px-2 py-1">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[12.5px] font-medium transition-colors",
                  isActive ? "bg-[var(--surface-3)] text-ink" : "text-ink2 hover:bg-[var(--surface-2)] hover:text-ink",
                )
              }
              title={n.label}
            >
              <n.icon size={16} className="shrink-0" />
              {sidebar && n.label}
            </NavLink>
          ))}
        </nav>
        <button className="m-2 flex items-center gap-2 rounded-md px-2.5 py-2 text-[12px] text-muted hover:bg-[var(--surface-2)]" onClick={toggleSidebar} aria-label="Toggle sidebar">
          <PanelLeft size={16} />
          {sidebar && "Collapse"}
        </button>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-3 border-b border-[var(--border)] bg-[var(--surface-1)] px-4">
          <h1 className="text-[14px] font-semibold">{title}</h1>
          {status?.data && (
            <Tip content={`Dataset ${status.data.version} · IS ${status.periods.is_start} → ${status.periods.os_start} · OS to ${status.periods.end}`}>
              <span className="chip cursor-default">
                {status.data.source === "real" ? "Real data" : "Demo data"} · {status.data.N} stocks · {status.data.start} → {status.data.end}
              </span>
            </Tip>
          )}
          {status && !status.warm && <span className="chip">warming up engine…</span>}
          <Tip content={brain?.connected ? `Signed in to BRAIN (user ${brain.user_id}); ${brain.usage_today} simulations today` : "Not connected to BRAIN: sign in on the BRAIN page to simulate there"}>
            <NavLink to="/brain" className={clsx("chip", brain?.connected ? "!text-[var(--good-text)]" : "!text-muted")}>
              <Cloud size={11} className="mr-1 inline" /> {brain?.connected ? "BRAIN connected" : "BRAIN offline"}
            </NavLink>
          </Tip>
          <div className="ml-auto flex items-center gap-2">
            {running > 0 && (
              <NavLink to="/miner" className="chip !text-[var(--accent)]">
                {running} job{running > 1 ? "s" : ""} running
              </NavLink>
            )}
            <Tip content={connected ? "Live updates connected" : "Reconnecting to the server…"}>
              <span className="flex items-center gap-1.5 text-[11px] text-muted">
                <span className={clsx("h-2 w-2 rounded-full", connected ? "bg-[var(--status-good)]" : "bg-[var(--status-warning)]")} />
                {connected ? "live" : "offline"}
              </span>
            </Tip>
            <button className="btn btn-ghost" onClick={() => setPalette(true)}>
              <Search size={14} /> <span className="text-muted">Ctrl+K</span>
            </button>
            <button className="btn btn-ghost !p-1.5" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle theme">
              {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </header>
        {demo && (
          <div className="flex items-center gap-2 border-b border-[var(--border)] bg-[var(--surface-2)] px-4 py-1.5 text-[12px] text-[var(--warn-text)]" role="alert">
            <AlertTriangle size={14} className="shrink-0" />
            <span>
              You are on the <b>synthetic demo data</b>. Alphas found here come from effects planted in the demo generator and will not pass on BRAIN, so mining is disabled.{" "}
              <NavLink to="/data" className="underline">Build real data</NavLink> first.
            </span>
          </div>
        )}
        <main className="min-h-0 flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  );
}
