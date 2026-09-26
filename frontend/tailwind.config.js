/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        page: "var(--page)",
        surface: "var(--surface-1)",
        surface2: "var(--surface-2)",
        surface3: "var(--surface-3)",
        ink: "var(--text-primary)",
        ink2: "var(--text-secondary)",
        muted: "var(--text-muted)",
        line: "var(--hairline)",
        accent: "var(--accent)",
        good: "var(--status-good)",
        warn: "var(--status-warning)",
        serious: "var(--status-serious)",
        bad: "var(--status-critical)",
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', '"Segoe UI"', "Roboto", "sans-serif"],
        mono: ['"Cascadia Code"', '"JetBrains Mono"', "Consolas", "ui-monospace", "monospace"],
      },
      fontSize: { "2xs": ["10px", "14px"] },
    },
  },
  plugins: [],
};
