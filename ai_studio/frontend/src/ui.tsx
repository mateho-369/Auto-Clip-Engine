import React from "react";

export const Panel = ({ title, right, children, className = "", scroll }: {
  title?: React.ReactNode; right?: React.ReactNode; children?: React.ReactNode;
  className?: string; scroll?: boolean;
}) => (
  <div className={`panel ${className}`}>
    {title !== undefined && (
      <div className="panel-h"><h3>{title}</h3>{right}</div>
    )}
    <div className={scroll ? "panel-scroll" : ""}>{children}</div>
  </div>
);

export const Badge = ({ kind = "", children, title }: { kind?: string; children: React.ReactNode; title?: string }) => (
  <span className={`badge ${kind}`} title={title}>{children}</span>
);

export const Bar = ({ pct }: { pct: number }) => (
  <div className="bar"><div style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} /></div>
);

export const Modal = ({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) => (
  <div className="modal-wrap" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
    <div className="modal">
      <div className="panel-h"><h3>{title}</h3>
        <button className="ghost" onClick={onClose}>×</button></div>
      <div className="panel-b">{children}</div>
    </div>
  </div>
);

export const Spinner = () => <span className="spin" />;

export const fmtDur = (s?: number | null) => {
  if (!s && s !== 0) return "—";
  const m = Math.floor(s / 60), sec = s - m * 60;
  return m ? `${m}m ${sec.toFixed(1)}s` : `${sec.toFixed(1)}s`;
};
export const fmtSize = (b?: number) => {
  if (!b) return "0 B";
  if (b > 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b > 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b > 1e3) return `${(b / 1e3).toFixed(0)} KB`;
  return `${b} B`;
};
export const fmtTime = (ts?: number) => (ts ? new Date(ts * 1000).toLocaleString() : "—");

export const StatusBadge = ({ status }: { status: string }) => {
  const map: Record<string, [string, string]> = {
    done: ["ok", "done"], completed: ["ok", "completed"], running: ["warn", "running"],
    queued: ["", "queued"], paused: ["warn", "paused"], failed: ["err", "failed"],
    skipped: ["", "skipped"], deferred: ["", "deferred"], blocked: ["err", "blocked"],
    review: ["warn", "review"], ready: ["blue", "ready"], draft: ["", "draft"],
    needs_review: ["warn", "needs approval"], cancelled: ["err", "cancelled"],
  };
  const [k, label] = map[status] || ["", status];
  return <Badge kind={k}>{label}</Badge>;
};

export const Empty = ({ text }: { text: string }) => (
  <div className="hint" style={{ padding: 14, textAlign: "center" }}>{text}</div>
);
