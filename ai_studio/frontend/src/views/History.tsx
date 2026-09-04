import React, { useEffect, useState } from "react";
import { api, ProjectRow, StatusPayload } from "../api";
import { useToast, errText } from "../main";
import { StatusBadge, fmtTime, Empty, Spinner, Badge } from "../ui";

export function HistoryView({ status, onOpen }: {
  status: StatusPayload | null; onOpen: (v: string, id?: string) => void;
}) {
  const [rows, setRows] = useState<ProjectRow[]>([]);
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const toast = useToast();

  useEffect(() => {
    (async () => {
      try {
        const [p, r] = await Promise.all([
          api<{ projects: ProjectRow[] }>("/projects", { query: { limit: 300 } }),
          api<{ runs: any[] }>("/runs", { query: { limit: 100 } }),
        ]);
        setRows(p.projects || []);
        setRuns(r.runs || []);
      } catch (e) { toast(errText(e), "err"); }
      setLoading(false);
    })();
  }, [toast]);

  return (
    <div className="pad">
      <h2 style={{ marginBottom: 12 }}>History</h2>
      {loading ? <Spinner /> : (
        <div className="split" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <div>
            <h3 style={{ marginBottom: 8 }}>Projects</h3>
            <table className="grid">
              <thead><tr><th>title</th><th>mode</th><th>type</th><th>status</th><th>updated</th></tr></thead>
              <tbody>
                {rows.map((p) => (
                  <tr key={p.id} style={{ cursor: "pointer" }} onClick={() => onOpen("project", p.id)}>
                    <td><b>{p.title}</b></td><td>{p.mode}</td><td><Badge>{p.content_type}</Badge></td>
                    <td><StatusBadge status={p.last_run_status || p.status} /></td>
                    <td className="mono">{fmtTime(p.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!rows.length && <Empty text="no projects yet" />}
          </div>
          <div>
            <h3 style={{ marginBottom: 8 }}>Recent runs</h3>
            <table className="grid">
              <thead><tr><th>run</th><th>trigger</th><th>status</th><th>profile</th><th>finished</th></tr></thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id} style={{ cursor: "pointer" }} onClick={() => onOpen("project", r.project_id)}>
                    <td className="mono">{r.id}</td><td>{r.trigger}</td>
                    <td><StatusBadge status={r.status} /></td><td>{r.machine_profile}</td>
                    <td className="mono">{fmtTime(r.finished_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!runs.length && <Empty text="no runs yet" />}
          </div>
        </div>
      )}
    </div>
  );
}
