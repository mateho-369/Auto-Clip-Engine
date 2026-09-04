import React, { createContext, useCallback, useContext, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ApiError } from "./api";
import "./styles.css";
import App from "./App";

export interface Toast { id: number; msg: string; kind: "ok" | "err" | "warn" | "info"; }

const ToastCtx = createContext<(msg: string, kind?: Toast["kind"]) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

function ToastHost({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const seq = useRef(1);
  const push = useCallback((msg: string, kind: Toast["kind"] = "err") => {
    const id = seq.current++;
    setItems((x) => [...x.slice(-5), { id, msg, kind }]);
    setTimeout(() => setItems((x) => x.filter((t) => t.id !== id)), kind === "err" ? 9500 : 4500);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div id="toasts">
        {items.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`} onClick={() => setItems((x) => x.filter((y) => y.id !== t.id))}>
            {t.msg}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export function errText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  return e instanceof Error ? e.message : String(e);
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ToastHost>
      <App />
    </ToastHost>
  </React.StrictMode>
);
