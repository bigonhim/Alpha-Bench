import * as RTooltip from "@radix-ui/react-tooltip";
import { QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "sonner";
import App from "./App";
import { queryClient } from "./lib/query";
import { useUI } from "./lib/store";
import { connectEvents } from "./lib/ws";
import "./styles.css";

document.documentElement.setAttribute("data-theme", useUI.getState().theme);
connectEvents();

function ThemedToaster() {
  const theme = useUI((s) => s.theme);
  return <Toaster theme={theme} position="bottom-right" richColors closeButton />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <RTooltip.Provider>
          <App />
          <ThemedToaster />
        </RTooltip.Provider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
