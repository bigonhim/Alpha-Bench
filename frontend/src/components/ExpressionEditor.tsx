import Editor, { type OnMount } from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";
import { useEffect, useRef } from "react";
import { monaco } from "../lib/monaco";
import { LANG, registerFastExpr } from "../lib/fastexpr-lang";
import { api } from "../lib/api";
import { useCatalog } from "../lib/hooks";
import { useUI } from "../lib/store";
import type { Analysis } from "../lib/types";

export function ExpressionEditor({
  value,
  onChange,
  onRun,
  onSave,
  onAnalysis,
  height = 132,
  readOnly = false,
}: {
  value: string;
  onChange: (v: string) => void;
  onRun?: () => void;
  onSave?: () => void;
  onAnalysis?: (a: Analysis | null) => void;
  height?: number;
  readOnly?: boolean;
}) {
  const { data: catalog } = useCatalog();
  const theme = useUI((s) => s.theme);
  const edRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);
  const runRef = useRef(onRun);
  const saveRef = useRef(onSave);
  const anRef = useRef(onAnalysis);
  runRef.current = onRun;
  saveRef.current = onSave;
  anRef.current = onAnalysis;
  const seq = useRef(0);

  // register the language/themes once the catalog is known, then (re)apply the theme; applying it
  // before registration would silently fall back to Monaco's default light theme
  useEffect(() => {
    if (catalog) registerFastExpr(monaco as unknown as typeof Monaco, catalog);
    monaco.editor.setTheme(theme === "dark" ? "af-dark" : "af-light");
  }, [catalog, theme]);

  // debounced analysis -> markers
  useEffect(() => {
    const id = ++seq.current;
    const t = setTimeout(async () => {
      const ed = edRef.current;
      if (!value.trim()) {
        anRef.current?.(null);
        if (ed?.getModel()) monaco.editor.setModelMarkers(ed.getModel()!, "fastexpr", []);
        return;
      }
      try {
        const a = await api.parse(value);
        if (id !== seq.current) return;
        anRef.current?.(a);
        const model = ed?.getModel();
        if (!model) return;
        const S = monaco.MarkerSeverity;
        monaco.editor.setModelMarkers(
          model,
          "fastexpr",
          a.diagnostics.map((d) => {
            const s = model.getPositionAt(d.start);
            const e = model.getPositionAt(Math.max(d.end, d.start + 1));
            return {
              startLineNumber: s.lineNumber,
              startColumn: s.column,
              endLineNumber: e.lineNumber,
              endColumn: e.column,
              message: d.message,
              severity: d.severity === "error" ? S.Error : d.severity === "warning" ? S.Warning : S.Info,
            };
          }),
        );
      } catch {
        /* server busy; ignore */
      }
    }, 160);
    return () => clearTimeout(t);
  }, [value]);

  const mount: OnMount = (ed) => {
    edRef.current = ed as unknown as Monaco.editor.IStandaloneCodeEditor;
    const KM = monaco.KeyMod;
    const KC = monaco.KeyCode;
    ed.addCommand(KM.CtrlCmd | KC.Enter, () => runRef.current?.());
    ed.addCommand(KM.CtrlCmd | KC.KeyS, () => saveRef.current?.());
    ed.addCommand(KM.Shift | KM.Alt | KC.KeyF, async () => {
      const a = await api.parse(ed.getValue());
      if (a.ok && a.pretty) {
        const hasAssign = /(^|;)\s*[A-Za-z_]\w*\s*=(?!=)/.test(ed.getValue());
        if (!hasAssign) ed.setValue(a.pretty);
      }
    });
    ed.focus();
  };

  return (
    <div className="overflow-hidden rounded-md border border-[var(--border)]" style={{ background: "var(--editor-bg)" }}>
      <Editor
        height={height}
        language={LANG}
        value={value}
        theme={theme === "dark" ? "af-dark" : "af-light"}
        onChange={(v) => onChange(v ?? "")}
        onMount={mount}
        loading={<div className="p-3 text-[12px] text-muted">Loading editor…</div>}
        options={{
          readOnly,
          minimap: { enabled: false },
          fontSize: 13.5,
          fontFamily: '"Cascadia Code", "JetBrains Mono", Consolas, monospace',
          fontLigatures: false,
          wordWrap: "on",
          lineNumbers: "on",
          lineNumbersMinChars: 2,
          glyphMargin: false,
          folding: false,
          scrollBeyondLastLine: false,
          automaticLayout: true,
          fixedOverflowWidgets: true,
          renderLineHighlight: "line",
          padding: { top: 8, bottom: 8 },
          quickSuggestions: { other: true, comments: false, strings: false },
          suggest: { showWords: false, preview: true },
          wordBasedSuggestions: "off",
          overviewRulerLanes: 0,
          scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
          tabSize: 2,
        }}
      />
    </div>
  );
}
