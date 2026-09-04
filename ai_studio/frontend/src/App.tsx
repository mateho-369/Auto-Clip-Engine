import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, StatusPayload } from "./api";
import { useToast } from "./main";
import { errText } from "./main";
import { StatusBadge } from "./ui";
import { ProjectsView } from "./views/Projects";
import { ProjectView } from "./views/Project";
import { AdminView, SERVICES } from "./views/Admin";
import { HistoryView } from "./views/History";

export type Route = { view: string; id?: string };

const NAV = [
  { key: "projects", icon: "🗂", label: "Projects" },
  { key: "history", icon: "🕘", label: "History" },
  { key: "services", icon: "🖥", label: "Services" },
  { key: "team", icon: "🤖", label: "AI Team" },
  { key: "characters", icon: "🧑‍🎤", label: "Characters" },
  { key: "voices", icon: "🎙", label: "Voices" },
  { key: "plugins", icon: "🧩", label: "Plugins / Engines" },
  { key: "settings", icon: "⚙️", label: "Settings" },
  { key: "memory", icon: "🔍", label: "Memory" },
];

export default function App() {
  const [route, setRoute] = useState<Route>(() => {
    const h = location.hash.replace(/^#\//, "");
    const [view, id] = h.split("/");
    return { view: view || "projects", id };
  });
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [statusErr, setStatusErr] = useState("");
  const toast = useToast();

  useEffect(() => {
    const onhash = () => {
      const h = location.hash.replace(/^#\//, "");
      const [view, id] = h.split("/");
      setRoute({ view: view || "projects", id });
    };
    window.addEventListener("hashchange", onhash);
    return () => window.removeEventListener("hashchange", onhash);
  }, []);

  const go = useCallback((view: string, id?: string) => {
    const h = "#/" + view + (id ? "/" + id : "");
    if (location.hash === h) return;
    location.hash = h;
  }, []);

  const refreshStatus = useCallback(async (silent = true) => {
    try {
      const s = await api<StatusPayload>("/status");
      setStatus(s);
      setStatusErr("");
      return s;
    } catch (e) {
      const msg = errText(e);
      setStatusErr(msg);
      if (!silent) toast(msg, "err");
      return null;
    }
  }, [toast]);

  useEffect(() => {
    refreshStatus(true);
    const t = setInterval(() => refreshStatus(true), 15000);
    return () => clearInterval(t);
  }, [refreshStatus]);

  const active = route.view === "project" ? "projects" : route.view;
  const header = status ? (
    <span className="row" style={{ gap: 6 }}>
      <span className="statusdot" style={{ background: statusErr ? "var(--red)" : "var(--green)" }} />
      <span className="hint">{statusErr ? "API unreachable — " + statusErr.slice(0, 80) :
        `v${status.version} · ${status.machine?.profile || "auto"}`}</span>
    </span>
  ) : null;

  return (
    <div className="shell">
      <div className="topbar">
        <span className="brand">◈ Khmer AI Content Studio</span>
        <span className="crumb">{route.view === "project" ? "project" : route.view}</span>
        <span className="spacer" />
        {header}
        <button className="btn tiny" onClick={() => refreshStatus(false)} title="re-probe">⟳</button>
      </div>
      <div className="body">
        <div className="side">
          {NAV.map((n) => (
            <button key={n.key} className={`nav ${active === n.key ? "on" : ""}`} onClick={() => go(n.key)}>
              <span>{n.icon}</span>{n.label}
            </button>
          ))}
          <div className="mini">Studio API :8000 · Ollama :11434 · RVC :9513 · ComfyUI :8188</div>
        </div>
        <div className="main">
          {statusErr && route.view === "projects" && (
            <div style={{ padding: "10px 18px 0" }}>
              <div className="errbar">⚠ Studio API error: {statusErr}</div>
            </div>
          )}
          {route.view === "projects" && <ProjectsView status={status} onOpen={go} />}
          {route.view === "history" && <HistoryView status={status} onOpen={go} />}
          {["services", "team", "plugins"].includes(route.view) && (
            <AdminView tab={route.view} status={status} onNavigate={go} />
          )}
          {route.view === "characters" && <AdminView tab="characters" status={status} onNavigate={go} />}
          {["voices", "settings", "memory"].includes(route.view) && (
            <AdminView tab={route.view} status={status} onNavigate={go} />
          )}
          {route.view === "project" && route.id && <ProjectView projectId={route.id} onOpen={go} />}
        </div>
      </div>
    </div>
  );
}

export { SERVICES };
