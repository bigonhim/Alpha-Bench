import type * as Monaco from "monaco-editor";
import type { Catalog, OpSpec } from "./types";

export const LANG = "fastexpr";
const WINDOW_SUGGESTIONS = ["5", "10", "20", "60", "120", "252"];
const GROUPS = ["market", "sector", "industry", "subindustry", 'bucket(rank(cap), range="0.1,1,0.1")'];

let catalog: Catalog | null = null;
let registered = false;
let opMap = new Map<string, OpSpec>();

export function setCatalog(c: Catalog) {
  catalog = c;
  opMap = new Map(c.operators.map((o) => [o.name, o]));
}

function variablesIn(text: string): string[] {
  const out = new Set<string>();
  const re = /(^|;)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) out.add(m[2]);
  return [...out];
}

/** Innermost unclosed call before `offset`: its name and the index of the argument being typed. */
export function callContext(text: string, offset: number): { name: string; argIndex: number; kw?: string } | null {
  let depth = 0;
  let arg = 0;
  let inStr: string | null = null;
  for (let i = offset - 1; i >= 0; i--) {
    const ch = text[i];
    if (inStr) {
      if (ch === inStr) inStr = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      inStr = ch;
      continue;
    }
    if (ch === ")") depth++;
    else if (ch === "(") {
      if (depth === 0) {
        const m = /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(text.slice(0, i));
        if (!m) return null;
        const seg = text.slice(i + 1, offset);
        const kwm = /([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^,]*$/.exec(seg.split(",").pop() || "");
        return { name: m[1], argIndex: arg, kw: kwm ? kwm[1] : undefined };
      }
      depth--;
    } else if (ch === "," && depth === 0) arg++;
    else if (ch === ";" && depth === 0) return null;
  }
  return null;
}

function paramDoc(o: OpSpec): string {
  return o.params
    .map((p) => {
      const kinds: Record<string, string> = { m: "matrix", g: "group", v: "vector field", w: "window (days)", i: "integer", n: "number", s: "text", b: "true/false", e: "choice" };
      const def = p.default !== undefined && p.default !== null && p.default !== "" ? ` = ${JSON.stringify(p.default)}` : "";
      const ch = p.choices ? ` (${p.choices.join(" | ")})` : "";
      return `- \`${p.name}\`: ${kinds[p.kind] || p.kind}${ch}${def}`;
    })
    .join("\n");
}

function snippetFor(o: OpSpec): string {
  let k = 1;
  const parts: string[] = [];
  for (const p of o.params) {
    const hasDefault = p.default !== undefined && p.default !== null;
    if (hasDefault && p.kind !== "m" && p.kind !== "g" && p.kind !== "v") continue;
    if (p.kind === "w") parts.push(`\${${k++}:20}`);
    else if (p.kind === "g") parts.push(`\${${k++}|subindustry,industry,sector,market|}`);
    else parts.push(`\${${k++}:${p.name}}`);
  }
  return `${o.name}(${parts.join(", ")})`;
}

export function registerFastExpr(monaco: typeof Monaco, c: Catalog) {
  setCatalog(c);
  const opNames = c.operators.map((o) => o.name).filter((n) => /^[a-z_]+$/.test(n));
  const fieldIds = c.fields.filter((f) => f.type !== "GROUP").map((f) => f.id);
  const groupIds = c.fields.filter((f) => f.type === "GROUP").map((f) => f.id);

  if (!registered) {
    monaco.languages.register({ id: LANG });
    monaco.languages.setLanguageConfiguration(LANG, {
      brackets: [["(", ")"]],
      autoClosingPairs: [
        { open: "(", close: ")" },
        { open: '"', close: '"' },
      ],
      surroundingPairs: [{ open: "(", close: ")" }],
      wordPattern: /[A-Za-z_][A-Za-z0-9_]*/,
    });
  }
  monaco.languages.setMonarchTokensProvider(LANG, {
    ops: opNames,
    fields: fieldIds,
    groups: groupIds,
    kwconst: ["true", "false", "nan", "gaussian", "uniform", "cauchy"],
    tokenizer: {
      root: [
        [/[A-Za-z_][A-Za-z0-9_]*(?=\s*=[^=])/, "variable.def"],
        [
          /[A-Za-z_][A-Za-z0-9_]*/,
          { cases: { "@ops": "keyword", "@groups": "type", "@fields": "field", "@kwconst": "constant", "@default": "identifier" } },
        ],
        [/(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/, "number"],
        [/"[^"]*"|'[^']*'/, "string"],
        [/&&|\|\||==|!=|<=|>=|[-+*/<>!?:=]/, "operator"],
        [/[;,()]/, "delimiter"],
      ],
    },
  } as Monaco.languages.IMonarchLanguage);

  if (registered) return;
  registered = true;

  monaco.editor.defineTheme("af-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "6da7ec", fontStyle: "bold" },
      { token: "field", foreground: "4fd1a5" },
      { token: "type", foreground: "b3a9f5" },
      { token: "number", foreground: "f39a6b" },
      { token: "string", foreground: "e6c07b" },
      { token: "constant", foreground: "e6c07b" },
      { token: "operator", foreground: "c3c2b7" },
      { token: "variable.def", foreground: "e87ba4", fontStyle: "italic" },
      { token: "identifier", foreground: "ffffff" },
    ],
    colors: {
      "editor.background": "#1a1a19",
      "editor.lineHighlightBackground": "#222220",
      "editorLineNumber.foreground": "#5d5c57",
      "editorCursor.foreground": "#3987e5",
      "editor.selectionBackground": "#2a4a70",
      "editorWidget.background": "#222220",
      "editorSuggestWidget.background": "#222220",
      "editorSuggestWidget.selectedBackground": "#2c3e55",
      "editorHoverWidget.background": "#222220",
    },
  });
  monaco.editor.defineTheme("af-light", {
    base: "vs",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "1c5cab", fontStyle: "bold" },
      { token: "field", foreground: "0b7a52" },
      { token: "type", foreground: "4a3aa7" },
      { token: "number", foreground: "b8461b" },
      { token: "string", foreground: "8a5a00" },
      { token: "constant", foreground: "8a5a00" },
      { token: "operator", foreground: "52514e" },
      { token: "variable.def", foreground: "a8336a", fontStyle: "italic" },
      { token: "identifier", foreground: "0b0b0b" },
    ],
    colors: {
      "editor.background": "#fcfcfb",
      "editor.lineHighlightBackground": "#f3f2ee",
      "editorLineNumber.foreground": "#a3a19a",
      "editorCursor.foreground": "#2a78d6",
    },
  });

  monaco.languages.registerCompletionItemProvider(LANG, {
    triggerCharacters: ["(", ",", " ", "="],
    provideCompletionItems(model, position) {
      if (!catalog) return { suggestions: [] };
      const text = model.getValue();
      const offset = model.getOffsetAt(position);
      const word = model.getWordUntilPosition(position);
      const range = new monaco.Range(position.lineNumber, word.startColumn, position.lineNumber, word.endColumn);
      const K = monaco.languages.CompletionItemKind;
      const ctx = callContext(text, offset);
      const suggestions: Monaco.languages.CompletionItem[] = [];
      if (ctx) {
        const op = opMap.get(ctx.name);
        if (op) {
          const param = ctx.kw ? op.params.find((p) => p.name === ctx.kw) : op.params[ctx.argIndex];
          if (param?.kind === "w") {
            WINDOW_SUGGESTIONS.forEach((w, i) =>
              suggestions.push({ label: w, kind: K.Value, insertText: w, range, sortText: `0${i}`, detail: "lookback window (days)" }),
            );
          } else if (param?.kind === "g") {
            GROUPS.forEach((g, i) =>
              suggestions.push({ label: g, kind: K.Enum, insertText: g, range, sortText: `0${i}`, detail: "neutralization group" }),
            );
          } else if (param?.kind === "e" && param.choices) {
            param.choices.forEach((ch) => suggestions.push({ label: ch, kind: K.EnumMember, insertText: ch, range }));
          }
          if (!ctx.kw) {
            for (const p of op.params) {
              if (p.default !== undefined && p.kind !== "m" && p.kind !== "g") {
                suggestions.push({
                  label: `${p.name}=`,
                  kind: K.Property,
                  insertText: `${p.name}=`,
                  range,
                  sortText: `1${p.name}`,
                  detail: `option (default ${JSON.stringify(p.default)})`,
                });
              }
            }
          }
        }
      }
      for (const v of variablesIn(text)) suggestions.push({ label: v, kind: K.Variable, insertText: v, range, sortText: `2${v}`, detail: "variable" });
      for (const f of catalog.fields) {
        suggestions.push({
          label: f.id,
          kind: f.type === "GROUP" ? K.Enum : K.Field,
          insertText: f.id,
          range,
          sortText: `${f.available ? "3" : "5"}${f.id}`,
          detail: `${f.dataset} · ${f.type}${f.available ? "" : " · BRAIN-only"}`,
          documentation: { value: `${f.description}\n\n*${f.available ? "Available locally" : "Not in the local dataset (BRAIN-only)"}*` },
        });
      }
      for (const o of catalog.operators) {
        suggestions.push({
          label: o.name,
          kind: K.Function,
          insertText: snippetFor(o),
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          range,
          sortText: `4${o.local ? "0" : "1"}${o.name}`,
          detail: `${o.signature}${o.local ? "" : "  (BRAIN-only)"}`,
          documentation: { value: `**${o.category}** — ${o.doc}\n\n${paramDoc(o)}\n\nExample: \`${o.example}\`` },
        });
      }
      return { suggestions };
    },
  });

  monaco.languages.registerHoverProvider(LANG, {
    provideHover(model, position) {
      if (!catalog) return null;
      const w = model.getWordAtPosition(position);
      if (!w) return null;
      const op = opMap.get(w.word);
      const range = new monaco.Range(position.lineNumber, w.startColumn, position.lineNumber, w.endColumn);
      if (op) {
        return {
          range,
          contents: [
            { value: `\`\`\`\n${op.signature}\n\`\`\`` },
            { value: `**${op.category}**${op.level === "consultant" ? " · consultant" : ""}${op.local ? "" : " · BRAIN-only (no local implementation)"}\n\n${op.doc}` },
            { value: `${paramDoc(op)}\n\nExample: \`${op.example}\`` },
          ],
        };
      }
      const f = catalog.fields.find((x) => x.id === w.word);
      if (f) {
        return {
          range,
          contents: [
            { value: `**${f.id}** · ${f.dataset} · ${f.type} · unit: ${f.unit}` },
            { value: `${f.description}\n\n${f.available ? "Available in the local dataset." : "Not available locally — BRAIN-only."}${f.verified ? "" : " Field id unverified."}` },
          ],
        };
      }
      return null;
    },
  });

  monaco.languages.registerSignatureHelpProvider(LANG, {
    signatureHelpTriggerCharacters: ["(", ","],
    signatureHelpRetriggerCharacters: [","],
    provideSignatureHelp(model, position) {
      const ctx = callContext(model.getValue(), model.getOffsetAt(position));
      if (!ctx) return null;
      const op = opMap.get(ctx.name);
      if (!op) return null;
      // parameter label offsets inside the signature text (skipping the variadic "...")
      const sig = op.signature;
      let pos = sig.indexOf("(") + 1;
      const offsets: [number, number][] = [];
      for (const part of sig.slice(pos, -1).split(", ")) {
        const name = part.split("=")[0];
        if (name !== "...") offsets.push([pos, pos + name.length]);
        pos += part.length + 2;
      }
      const params = op.params.map((p, i) => ({ label: offsets[i] ?? p.name, documentation: p.kind }));
      const active = ctx.kw ? Math.max(0, op.params.findIndex((p) => p.name === ctx.kw)) : Math.min(ctx.argIndex, Math.max(0, params.length - 1));
      return {
        value: {
          signatures: [{ label: sig, documentation: op.doc, parameters: params }],
          activeSignature: 0,
          activeParameter: active,
        },
        dispose() {},
      };
    },
  });
}
