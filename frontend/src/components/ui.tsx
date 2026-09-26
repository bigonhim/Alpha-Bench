import * as Dialog from "@radix-ui/react-dialog";
import * as RTabs from "@radix-ui/react-tabs";
import * as RTooltip from "@radix-ui/react-tooltip";
import { AlertTriangle, CheckCircle2, CircleDashed, HelpCircle, Loader2, X, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { clsx } from "../lib/format";

export function Card({ title, actions, children, className, bodyClass }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string }) {
  return (
    <section className={clsx("card", className)}>
      {(title || actions) && (
        <div className="card-h">
          <span className="truncate">{title}</span>
          <div className="flex items-center gap-1.5 normal-case tracking-normal font-normal">{actions}</div>
        </div>
      )}
      <div className={clsx("px-3 pb-3", bodyClass)}>{children}</div>
    </section>
  );
}

const STATUS: Record<string, { cls: string; icon: ReactNode; label: string }> = {
  PASS: { cls: "text-[var(--good-text)]", icon: <CheckCircle2 size={13} />, label: "Pass" },
  FAIL: { cls: "text-[var(--bad-text)]", icon: <XCircle size={13} />, label: "Fail" },
  WARNING: { cls: "text-[var(--warn-text)]", icon: <AlertTriangle size={13} />, label: "Warning" },
  PENDING: { cls: "text-[var(--text-muted)]", icon: <CircleDashed size={13} />, label: "Pending" },
  UNSCORED: { cls: "text-[var(--text-muted)]", icon: <HelpCircle size={13} />, label: "BRAIN-only" },
};

/** Status is never color alone: icon + label always accompany it. */
export function StatusBadge({ status, label, compact = false }: { status: string; label?: string; compact?: boolean }) {
  const s = STATUS[status] || STATUS.PENDING;
  return (
    <span className={clsx("inline-flex items-center gap-1 font-medium", s.cls, compact ? "text-[11px]" : "text-[12px]")}>
      {s.icon}
      {label ?? s.label}
    </span>
  );
}

export function Spinner({ size = 14 }: { size?: number }) {
  return <Loader2 size={size} className="animate-spin text-[var(--text-muted)]" />;
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 py-10 text-center">
      <div className="text-[13px] font-medium text-ink2">{title}</div>
      {children && <div className="max-w-md text-[12px] text-muted">{children}</div>}
    </div>
  );
}

export function Tabs({ tabs, value, onChange, className }: { tabs: { id: string; label: ReactNode; content: ReactNode }[]; value: string; onChange: (v: string) => void; className?: string }) {
  return (
    <RTabs.Root value={value} onValueChange={onChange} className={className}>
      <RTabs.List className="flex gap-0.5 border-b border-line px-1">
        {tabs.map((t) => (
          <RTabs.Trigger
            key={t.id}
            value={t.id}
            className="relative px-2.5 py-1.5 text-[12px] font-medium text-muted outline-none transition-colors hover:text-ink data-[state=active]:text-ink data-[state=active]:after:absolute data-[state=active]:after:inset-x-1.5 data-[state=active]:after:-bottom-px data-[state=active]:after:h-0.5 data-[state=active]:after:rounded data-[state=active]:after:bg-[var(--accent)]"
          >
            {t.label}
          </RTabs.Trigger>
        ))}
      </RTabs.List>
      {tabs.map((t) => (
        <RTabs.Content key={t.id} value={t.id} className="outline-none fade-in">
          {t.content}
        </RTabs.Content>
      ))}
    </RTabs.Root>
  );
}

export function Modal({ open, onOpenChange, title, children, wide = false, footer }: { open: boolean; onOpenChange: (o: boolean) => void; title: ReactNode; children: ReactNode; wide?: boolean; footer?: ReactNode }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[1px]" />
        <Dialog.Content
          className={clsx(
            "card fixed left-1/2 top-1/2 z-50 flex max-h-[88vh] -translate-x-1/2 -translate-y-1/2 flex-col shadow-2xl outline-none fade-in",
            wide ? "w-[min(1100px,94vw)]" : "w-[min(640px,94vw)]",
          )}
        >
          <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
            <Dialog.Title className="text-[14px] font-semibold">{title}</Dialog.Title>
            <Dialog.Close className="btn btn-ghost !p-1" aria-label="Close">
              <X size={16} />
            </Dialog.Close>
          </div>
          <Dialog.Description className="sr-only">{typeof title === "string" ? title : "dialog"}</Dialog.Description>
          <div className="overflow-auto p-4">{children}</div>
          {footer && <div className="flex justify-end gap-2 border-t border-line px-4 py-2.5">{footer}</div>}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <RTooltip.Root delayDuration={250}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content side="top" sideOffset={4} className="z-50 max-w-xs rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1 text-[11.5px] text-ink shadow-lg">
          {content}
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  );
}

export function Field({ label, children, className, hint }: { label: string; children: ReactNode; className?: string; hint?: string }) {
  return (
    <div className={clsx("flex flex-col gap-1", className)}>
      <label className="lbl" title={hint}>
        {label}
      </label>
      {children}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="inline-flex cursor-pointer select-none items-center gap-2 text-[12px] text-ink2">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={clsx("relative h-4 w-7 rounded-full transition-colors", checked ? "bg-[var(--accent)]" : "bg-[var(--surface-3)]")}
      >
        <span className={clsx("absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all", checked ? "left-3.5" : "left-0.5")} />
      </button>
      {label}
    </label>
  );
}

export function Progress({ value, className }: { value: number; className?: string }) {
  return (
    <div className={clsx("h-1.5 w-full overflow-hidden rounded-full bg-[var(--surface-3)]", className)}>
      <div className="h-full rounded-full bg-[var(--accent)] transition-all" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

export function Kv({ k, v, mono }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5 text-[12px]">
      <span className="text-muted">{k}</span>
      <span className={clsx("tnum text-right text-ink", mono && "mono")}>{v}</span>
    </div>
  );
}

export function Select({ value, onChange, options, className, title }: { value: string | number; onChange: (v: string) => void; options: (string | number | { value: string | number; label: string })[]; className?: string; title?: string }) {
  return (
    <select className={clsx("select", className)} value={value} onChange={(e) => onChange(e.target.value)} title={title}>
      {options.map((o) => {
        const v = typeof o === "object" ? o.value : o;
        const l = typeof o === "object" ? o.label : String(o);
        return (
          <option key={String(v)} value={v}>
            {l}
          </option>
        );
      })}
    </select>
  );
}
