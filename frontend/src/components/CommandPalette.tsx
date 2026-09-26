import { useQuery } from "@tanstack/react-query";
import { Command } from "cmdk";
import { Moon, Rocket, Sparkles, Wand2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "../lib/api";
import { useHotkey } from "../lib/hooks";
import { fmt } from "../lib/format";
import { useUI } from "../lib/store";
import { NAV } from "./Layout";

export function CommandPalette() {
  const { paletteOpen, setPalette, theme, setTheme, setStudio } = useUI();
  const nav = useNavigate();
  useHotkey("k", () => setPalette(!useUI.getState().paletteOpen));
  const { data } = useQuery({ queryKey: ["alphas", "palette"], queryFn: () => api.alphas({ sort: "fitness", limit: 60 }), enabled: paletteOpen });
  if (!paletteOpen) return null;
  const go = (fn: () => void) => {
    setPalette(false);
    fn();
  };
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 pt-[12vh]" onClick={() => setPalette(false)}>
      <div className="w-[min(640px,92vw)]" onClick={(e) => e.stopPropagation()}>
        <Command label="Command palette" onKeyDown={(e) => e.key === "Escape" && setPalette(false)}>
          <Command.Input autoFocus placeholder="Jump to a page, run an action, or open an alpha…" />
          <Command.List>
            <Command.Empty>No results.</Command.Empty>
            <Command.Group heading="Pages">
              {NAV.map((n) => (
                <Command.Item key={n.to} value={`page ${n.label}`} onSelect={() => go(() => nav(n.to))}>
                  <n.icon size={14} /> {n.label}
                </Command.Item>
              ))}
            </Command.Group>
            <Command.Group heading="Actions">
              <Command.Item
                value="start auto-mine"
                onSelect={() =>
                  go(async () => {
                    const r = await api.startJob("automine", { time_limit_min: 20, target_candidates: 20 });
                    toast.success(`Auto-Mine started (job #${r.id})`);
                    nav(`/miner?job=${r.id}`);
                  })
                }
              >
                <Sparkles size={14} /> Start Auto-Mine (20 min)
              </Command.Item>
              <Command.Item value="forge an alpha from an idea" onSelect={() => go(() => nav("/forge"))}>
                <Wand2 size={14} /> Forge an alpha from a plain-English idea
              </Command.Item>
              <Command.Item value="start genetic programming" onSelect={() => go(() => nav("/miner?method=gp"))}>
                <Rocket size={14} /> Configure a genetic-programming run
              </Command.Item>
              <Command.Item value="toggle theme" onSelect={() => go(() => setTheme(theme === "dark" ? "light" : "dark"))}>
                <Moon size={14} /> Toggle dark / light theme
              </Command.Item>
            </Command.Group>
            {data && data.rows.length > 0 && (
              <Command.Group heading="Library (best fitness)">
                {data.rows.map((a) => (
                  <Command.Item key={a.id} value={`alpha ${a.id} ${a.expr}`} onSelect={() => go(() => { setStudio(a.expr, a.settings); nav("/studio"); })}>
                    <span className="w-10 text-[11px] text-[var(--text-muted)]">#{a.id}</span>
                    <span className="mono min-w-0 flex-1 truncate text-[11.5px]">{a.expr}</span>
                    <span className="tnum shrink-0 text-[11px]">F {fmt.num(a.fitness)}</span>
                  </Command.Item>
                ))}
              </Command.Group>
            )}
          </Command.List>
        </Command>
      </div>
    </div>
  );
}
