import { useCatalog, useStatus } from "../lib/hooks";
import type { Settings } from "../lib/types";
import { Field, Select, Tip } from "./ui";

export function SettingsBar({ settings, onChange, compact = false }: { settings: Settings; onChange: (p: Partial<Settings>) => void; compact?: boolean }) {
  const { data: cat } = useCatalog();
  const { data: status } = useStatus();
  const o = cat?.settings_options;
  const localNeut = new Set(o?.neutralizations_local ?? ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]);
  const localU = status?.universe_map?.[settings.universe];
  const exportOnly = settings.region !== "USA" || !localNeut.has(settings.neutralization);
  return (
    <div className={compact ? "flex flex-wrap items-end gap-2" : "flex flex-wrap items-end gap-x-3 gap-y-2"}>
      <Field label="Region">
        <Select value={settings.region} onChange={(v) => onChange({ region: v })} options={o?.regions ?? ["USA"]} />
      </Field>
      <Field label="Universe">
        <Select value={settings.universe} onChange={(v) => onChange({ universe: v })} options={o?.universes ?? ["TOP3000"]} />
      </Field>
      <Field label="Delay">
        <Select value={settings.delay} onChange={(v) => onChange({ delay: Number(v) })} options={[1, 0]} />
      </Field>
      <Field label="Decay">
        <input className="input w-16 tnum" type="number" min={0} max={512} value={settings.decay} onChange={(e) => onChange({ decay: Math.max(0, Number(e.target.value) || 0) })} />
      </Field>
      <Field label="Neutralization">
        <Select
          value={settings.neutralization}
          onChange={(v) => onChange({ neutralization: v })}
          options={(o?.neutralizations ?? ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]).map((n) => ({ value: n, label: localNeut.has(n) ? n : `${n} (export only)` }))}
        />
      </Field>
      <Field label="Truncation">
        <input className="input w-16 tnum" type="number" min={0} max={1} step={0.01} value={settings.truncation} onChange={(e) => onChange({ truncation: Number(e.target.value) })} />
      </Field>
      <Field label="Pasteurize">
        <Select value={settings.pasteurization} onChange={(v) => onChange({ pasteurization: v })} options={["ON", "OFF"]} />
      </Field>
      <Field label="NaN handling">
        <Select value={settings.nanHandling} onChange={(v) => onChange({ nanHandling: v })} options={["OFF", "ON"]} />
      </Field>
      {!compact && (
        <div className="pb-1 text-[11px] text-muted">
          {exportOnly ? (
            <Tip content="Non-USA regions and risk-model neutralizations are exported to BRAIN as-is; local simulation uses USA data with market neutralization.">
              <span className="cursor-help underline decoration-dotted">export-only settings: local proxy approximates</span>
            </Tip>
          ) : (
            localU && (
              <span>
                local proxy universe <b className="text-ink2">{localU}</b>
              </span>
            )
          )}
        </div>
      )}
    </div>
  );
}
