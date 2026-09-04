import React, { useCallback, useEffect, useState } from "react";
import { api, ContentTypeMeta, Project, ProjectRow, StatusPayload, VoiceProfile } from "../api";
import { useToast, errText } from "../main";
import { Badge, Bar, Empty, Panel, StatusBadge, fmtTime, Spinner } from "../ui";
import { Wizard } from "./Wizard";

export function ProjectsView({ status, onOpen }: {
  status: StatusPayload | null; onOpen: (view: string, id?: string) => void;
}) {
  const [rows, setRows] = useState<ProjectRow[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [q, setQ] = useState(""); const [st, setSt] = useState(""); const [md, setMd] = useState("");
  const [loading, setLoading] = useState(true);
  const [wizard, setWizard] = useState(false);
  const toast = useToast();

  const load = useCallback(async () => {
    try {
      const r = await api<{ projects: ProjectRow[]; counts: Record<string, number> }>("/projects", {
        query: { search: q, status: st, mode: md },
      });
      setRows(r.projects || []);
      setCounts(r.counts || {});
      setLoading(false);
    } catch (e) { toast(errText(e), "err"); setLoading(false); }
  }, [q, st, md, toast]);

  useEffect(() => { load(); }, [load]);

  const del = async (p: ProjectRow) => {
    if (!confirm(`Delete "${p.title}"? This removes the project record${p.scene_count ? " and its data" : ""}.`)) return;
    try {
      await api(`/projects/${p.id}`, { method: "DELETE", query: { purge_files: true } });
      toast("project deleted", "ok"); load();
    } catch (e) { toast(errText(e), "err"); }
  };
  const dup = async (p: ProjectRow) => {
    try {
      const r = await api<{ project: Project }>(`/projects/${p.id}/duplicate`, { method: "POST", json: {} });
      toast(`duplicated → ${r.project.title}`, "ok"); load();
    } catch (e) { toast(errText(e), "err"); }
  };

  return (
    <div className="pad">
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2>Projects</h2>
        <div className="row">
          <input placeholder="search title / script / topic" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 260 }} />
          <select value={st} onChange={(e) => setSt(e.target.value)} style={{ width: 130 }}>
            <option value="">any status</option>
            {["draft", "ready", "review", "rendering", "done", "failed"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={md} onChange={(e) => setMd(e.target.value)} style={{ width: 130 }}>
            <option value="">any mode</option><option value="A">A · Director</option><option value="B">B · Auto</option>
          </select>
          <button className="btn primary" onClick={() => setWizard(true)}>+ New project</button>
        </div>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        {Object.entries(counts).map(([k, v]) => <Badge key={k}>{k}: {v}</Badge>)}
      </div>
      {loading ? <Spinner /> : rows.length === 0 ? <Empty text="No projects yet — create one with + New project." /> : (
        <div className="cards">
          {rows.map((p) => (
            <div key={p.id} className="project-card" onClick={() => onOpen("project", p.id)}>
              <ProjectThumb pid={p.id} />
              <div className="spread">
                <b style={{ fontSize: 13 }}>{p.title}</b>
                <StatusBadge status={p.last_run_status || p.status} />
              </div>
              <div className="hint" style={{ margin: "3px 0 6px" }}>
                {p.mode === "A" ? "Director script" : "Auto idea"} · {p.content_type} ·{" "}
                {p.target_duration}s · {p.scene_count} scenes · {p.run_count} runs
              </div>
              <div className="row" style={{ gap: 4 }}>
                <button className="btn tiny" onClick={(e) => { e.stopPropagation(); dup(p); }}>duplicate</button>
                <button className="btn tiny" onClick={(e) => { e.stopPropagation(); onOpen("project", p.id); }}>open</button>
                <button className="btn tiny danger" onClick={(e) => { e.stopPropagation(); del(p); }}>delete</button>
              </div>
              <div className="hint" style={{ marginTop: 5 }}>updated {fmtTime(p.updated_at)}</div>
            </div>
          ))}
        </div>
      )}
      {wizard && <Wizard onClose={() => setWizard(false)} onCreated={(id) => { setWizard(false); onOpen("project", id); }} />}
    </div>
  );
}

function ProjectThumb({ pid }: { pid: string }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    api<{ assets: any[] }>("/assets", { query: { project_id: pid, kind: "poster", limit: 1 } })
      .then((r) => { const a = r.assets?.[0]; if (a) setUrl(`/api/assets/${a.id}/stream`); })
      .catch(() => {});
  }, [pid]);
  return url ? <img className="thumb" src={url} /> : <div className="thumb">🗂</div>;
}
