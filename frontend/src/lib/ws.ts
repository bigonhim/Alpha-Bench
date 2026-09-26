import { toast } from "sonner";
import { queryClient } from "./query";
import { useJobs } from "./store";

let sock: WebSocket | null = null;
let retry = 0;
let invalidateTimer: number | undefined;

function scheduleInvalidate() {
  if (invalidateTimer) return;
  invalidateTimer = window.setTimeout(() => {
    invalidateTimer = undefined;
    queryClient.invalidateQueries({ queryKey: ["alphas"] });
    queryClient.invalidateQueries({ queryKey: ["dashboard"] });
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
  }, 1500);
}

export function connectEvents(): void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  sock = new WebSocket(`${proto}://${location.host}/ws`);
  const st = useJobs.getState();
  sock.onopen = () => {
    retry = 0;
    st.setConnected(true);
  };
  sock.onclose = () => {
    useJobs.getState().setConnected(false);
    const wait = Math.min(10000, 500 * 2 ** retry++);
    setTimeout(connectEvents, wait);
  };
  sock.onmessage = (m) => {
    let ev: any;
    try {
      ev = JSON.parse(m.data);
    } catch {
      return;
    }
    const s = useJobs.getState();
    switch (ev.type) {
      case "hello":
        for (const j of ev.jobs || []) s.upsertJob(j);
        break;
      case "job":
        s.upsertJob(ev.job, ev.results);
        if (ev.results?.length || ["done", "error", "cancelled"].includes(ev.job.status)) scheduleInvalidate();
        break;
      case "gp_generation":
        s.pushGP(ev.job_id, ev.stat);
        break;
      case "reengineer":
        s.setReport(ev.job_id, ev.report);
        break;
      case "data_ready":
        queryClient.invalidateQueries();
        toast.success("Dataset reloaded");
        break;
      case "toast":
        if (ev.level === "error") toast.error(ev.message);
        else if (ev.level === "success") toast.success(ev.message);
        else toast(ev.message);
        break;
      default:
        break;
    }
  };
}
