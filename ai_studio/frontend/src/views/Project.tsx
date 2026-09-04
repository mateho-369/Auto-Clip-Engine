import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, Asset, Project, Run, Scene, StageRow, StageSpec, StylePreview } from "../api";
import { useToast, errText } from "../main";
import { Badge, Bar, Empty, Panel, StatusBadge, fmtDur, fmtSize, fmtTime, Spinner } from "../ui";

interface Live {
  run_id?: string; status?: string; stages?: StageRow[]; overall?: { pct?: number }; events?: any[];
  error?: string; deferred_stages?: string[]; final_path?: string; duration?: number;
}

export function ProjectView({ projectId, onOpen }: { projectId: string; onOpen: (v: string, id?: string) => void }) {
  const [proj, setProj] = useState<Project | null>(null);
  const [specs, setSpecs] = useState<StageSpec[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [live, setLive] = useState<Live | null>(null);
  const [log, setLog] = useState<any[]>([]);
  const [selScene, setSelScene] = useState(0);
  const [selStage, setSelStage] = useState("voice_final");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [liveMode, setLiveMode] = useState("");
  const toast = useToast();
  const wsRef = useRef<WebSocket | null>(null);

  const load = useCallback(async () => {
    try {
      const d = await api<{ project: Project; scenes: Scene[]; runs: Run[] }>(`/projects/${projectId}`);
      setProj(d.project);
      setRun((r) => (r && d.runs?.length && r.id !== d.runs[0].id) ? d.runs[0] : (r || d.runs?.[0] || null));
      const s = await api<{ roles: StageSpec[] }>("/settings");
      setSpecs(s.roles || []);
      const a = await api<{ assets: Asset[] }>("/assets", { query: { project_id: projectId, limit: 300 } });
      setAssets(a.assets || []);
      setErr("");
    } catch (e) { setErr(errText(e)); }
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  // live WS with SSE + polling fallback
  const connect = useCallback((runId: string) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    let ws: WebSocket;
    try { ws = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/events`); }
    catch { return; }
    setLiveMode("ws");
    ws.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        if (m.kind === "snapshot") { setLive(m.payload); if (m.payload?.status) flush(); }
        else if (m.kind === "stage_update" || m.kind === "stage_failed") { setLive((x) => x || {}); refresh(runId); }
        else if (m.kind === "log") { setLog((x) => [...x.slice(-200), m.payload]); }
        else if (m.kind === "run_finished") { setLive(m.payload); refresh(runId); }
      } catch {}
    };
    ws.onclose = () => { setLiveMode("poll"); /* SSE + polling keep it honest */ };
    wsRef.current = ws;
    return ws;
  }, []);

  const refresh = useCallback(async (runId: string) => {
    try {
      const s = await api<Live>(`/runs/${runId}/status?since=0`);
      setLive(s); setLiveMode((m) => m || "poll");
    } catch {}
  }, []);

  const flush = useCallback(() => { /* placeholder for animation flush */ }, []);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  const startRun = async (payload: any = {}) => {
    setBusy("starting");
    try {
      const r = await api<{ run_id: string }>(`/projects/${projectId}/runs`, { method: "POST", json: payload });
      setBusy("");
      toast("run started", "ok");
      setLive({ run_id: r.run_id, status: "running" });
      connect(r.run_id);
      poll(r.run_id);
    } catch (e) { toast(errText(e), "err"); setBusy(""); }
  };

  const poll = useCallback(async (runId: string) => {
    for (let i = 0; i < 1200; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      try {
        const s = await api<Live>(`/runs/${runId}/status?since=0`);
        setLive(s); setLiveMode((m) => m || "poll");
        if (s.status && !["running", "queued", "paused"].includes(s.status)) { load(); setLiveMode(""); return; }
      } catch { /* transient */ }
    }
  }, [load]);

  const act = async (verb: string, path: string, body?: any, okMsg = "") => {
    setBusy(verb);
    try {
      const r = await api(path, { method: "POST", json: body });
      if (okMsg) toast(okMsg, "ok");
      setBusy("");
      return r as any;
    } catch (e) { toast(errText(e), "err"); setBusy(""); return null; }
  };

  if (!proj) return <div className="pad">{err && <div className="errbar">⚠ {err}</div>}<Spinner /></div>;

  if (proj.status === "draft" && proj.mode === "B" && !proj.script) {
    return <ModeBGate proj={proj} onOpen={onOpen} onChanged={load} act={act} busy={busy} />;
  }

  const runRows = (live?.stages || run?.stages || []).slice();
  const deferredCount = runRows.filter((r) => r.status === "deferred").length;
  const stages = specs.length ? specs : STAGE_FALLBACK;
  const sc = proj.scenes || [];

  return (
    <div className="pad">
      {err && <div className="errbar">⚠ {err}</div>}
      {live?.error && <div className="errbar">⚠ run error: {live.error}</div>}
      <div className="spread" style={{ marginBottom: 10 }}>
        <h2>{proj.title}</h2>
        <Badge kind="blue">{proj.mode === "A" ? "Director" : "Auto"}</Badge>
        <Badge>{proj.content_type}</Badge>
        {proj.character_id ? <Badge kind="warn">🧑 character</Badge> : null}
        <StatusBadge status={live?.status || proj.status} />
        <span className="spacer" />
        <button className="btn" disabled={!!busy} onClick={() => startRun({})}>{busy === "starting" ? <Spinner /> : "▶ Run"}</button>
        <button className="btn" disabled={!live?.run_id || !!busy}
          onClick={() => act("pause", `/runs/${live!.run_id}/pause`)}>⏸</button>
        <button className="btn" disabled={!live?.run_id || !!busy}
          onClick={() => act("resume", `/runs/${live!.run_id}/resume`)}>▶</button>
        <button className="btn" disabled={!live?.run_id || !!busy}
          onClick={() => act("cancel", `/runs/${live!.run_id}/cancel`)}>■</button>
        <button className="btn" disabled={!live?.run_id || !!busy}
          onClick={() => act("continue", `/runs/${live!.run_id}/continue`)}>continue</button>
        <span className={`dot ${liveMode ? "on" : ""}`} title={liveMode ? `live via ${liveMode}` : "offline"} />
        <button className="btn warn" disabled={!!busy || !(deferredCount > 0)}
          title="re-render the deferred GPU stages, then QA + assembly (voice reused)"
          onClick={() => act("catchup", `/projects/${proj.id}/catchup`, {}, "GPU catch-up started")}>
          🖥 GPU catch-up{(deferredCount > 0) ? ` (${deferredCount})` : ""}</button>
      </div>

      <div className="split wide-right" style={{ gridTemplateColumns: "minmax(0,1fr) 430px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <PipelineDAG stages={stages} rows={runRows} onStage={(k) => setSelStage(k)} />
          <SceneBoard proj={proj} scenes={sc} rows={runRows} assets={assets} sel={selScene}
            onSel={setSelScene} onChanged={load} act={act} busy={busy} />
          <ScriptPanel proj={proj} onChanged={load} act={act} busy={busy} />
          <EventLog log={log} rows={runRows} />
        </div>
        <Inspector proj={proj} scenes={sc} stage={selStage} scene={selScene} assets={assets}
          rows={runRows} onChanged={load} act={act} busy={busy} specs={stages} />
      </div>
    </div>
  );
}

const STAGE_FALLBACK: StageSpec[] = [
  "script", "breakdown", "voice_base", "voice_final", "talking_head", "video", "video_fit", "sfx", "qa", "assemble",
].map((k, i) => ({ key: k, title: k.replace(/_/g, " "), emoji: "▪", role: "", blurb: "", model: "",
  per_scene: true, requires_gpu: false, resource: "cpu", depends: [], deferrable: false }));

function PipelineDAG({ stages, rows, onStage }: {
  stages: StageSpec[]; rows: StageRow[]; onStage: (k: string) => void;
}) {
  const byStage: Record<string, StageRow[]> = {};
  rows.forEach((r) => { (byStage[r.stage] = byStage[r.stage] || []).push(r); });
  const stFor = (k: string) => {
    const rs = byStage[k] || [];
    const counts: Record<string, number> = {};
    rs.forEach((r) => { counts[r.status] = (counts[r.status] || 0) + 1; });
    if (counts.failed || counts.blocked) return "failed";
    if (rs.some((r) => ["running", "queued"].includes(r.status))) return "running";
    if (counts.done === rs.length && rs.length) return "done";
    if (counts.deferred === rs.length && rs.length) return "deferred";
    return "";
  };
  return (
    <Panel title="Pipeline" right={<span className="hint">click a stage → inspector</span>}>
      <div className="dag">
        {stages.map((s) => {
          const st = stFor(s.key);
          return (
            <div key={s.key} className={`node ${st}`} onClick={() => onStage(s.key)}>
              <div className="t">{s.emoji} {s.title}</div>
              <div className="m">{st || "waiting"}{(byStage[s.key]?.length || 0) > 1 ? ` · ${byStage[s.key].length} scenes` : ""}</div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function SceneBoard({ proj, scenes, rows, assets, sel, onSel, onChanged, act, busy }: {
  proj: Project; scenes: Scene[]; rows: StageRow[]; assets: Asset[]; sel: number; onSel: (i: number) => void;
  onChanged: () => void; act: (v: string, p: string, b?: any, ok?: string) => Promise<any>; busy: string;
}) {
  const toast = useToast();
  const [draft, setDraft] = useState<Scene[]>(scenes);
  useEffect(() => { setDraft(scenes); }, [scenes]);
  const save = async () => {
    try {
      await api(`/projects/${proj.id}/scenes`, { method: "POST", json: { scenes: draft.map((d) => ({ ...d, idx: undefined })) } });
      toast("storyboard saved", "ok"); onChanged();
    } catch (e) { toast(errText(e), "err"); }
  };
  const uploadImage = async (idx: number, f: File) => {
    const fd = new FormData(); fd.append("image", f);
    try { await api(`/projects/${proj.id}/scenes/${idx}/image`, { method: "POST", form: fd }); toast("scene image uploaded", "ok"); onChanged(); }
    catch (e) { toast(errText(e), "err"); }
  };
  const patchScene = async (idx: number, meta: Record<string, any>) => {
    const sc = draft[idx]; if (!sc) return;
    await saveWith(async () => {
      const next = draft.map((s, i) => i === idx ? { ...s, meta: { ...s.meta, ...meta } } : s);
      setDraft(next);
    });
  };
  const saveWith = async (mutate?: () => void) => {
    if (mutate) mutate();
    await save();
  };
  const stageFor = (idx: number, stage: string) => rows.find((r) => r.stage === stage && r.scene_idx === idx);

  return (
    <Panel title={`Scene board (${scenes.length})`}
      right={<button className="btn tiny primary" onClick={save} disabled={busy === "board"}>save board</button>}>
      <table className="grid">
        <thead><tr><th style={{ width: 30 }}>#</th><th>text</th><th style={{ width: 140 }}>visual</th>
          <th style={{width: 110}}>mood</th><th style={{ width: 80 }}>⏱</th><th style={{ width: 120 }}>production</th></tr></thead>
        <tbody>
          {draft.map((s, i) => {
            const done = stageFor(i, "video_fit")?.status === "done";
            return (
              <tr key={i} onClick={() => onSel(i)} style={{ cursor: "pointer", background: sel === i ? "#242a35" : undefined }}>
                <td><b>{i + 1}</b>{s.meta?.side ? <><br /><Badge>{s.meta.side}</Badge></> : null}</td>
                <td>
                  <textarea className="scene-text" value={s.text} rows={2}
                    onChange={(e) => setDraft(draft.map((x, j) => j === i ? { ...x, text: e.target.value } : x))} />
                  {s.meta?.character_id ? <div className="hint">🧑 {s.meta.character_id.slice(0, 10)}</div> : null}
                </td>
                <td className="hint" style={{ fontSize: 11 }}>{s.visual_prompt?.slice(0, 90) || "—"}</td>
                <td><Badge>{s.mood_tag || "—"}</Badge></td>
                <td className="mono">{fmtDur(s.estimated_duration_sec)}</td>
                <td>
                  <VisualSourceControl value={s.meta?.visual_source || ""} hasChar={!!proj.character_id || !!s.meta?.character_id}
                    onChange={(v) => patchScene(i, { visual_source: v })} />
                  {hasChar(proj, s) && (
                    <div className="row" style={{ marginTop: 4 }}>
                      <label className="hint">render:</label>
                      <select value={s.meta?.render_mode || "broll"} style={{ width: "auto", padding: "2px 6px", fontSize: 11 }}
                        onChange={(e) => patchScene(i, { render_mode: e.target.value })}>
                        <option value="broll">b-roll</option><option value="talking_head">talking head</option>
                      </select>
                    </div>
                  )}
                  <div className="row" style={{ marginTop: 4 }}>
                    <input type="file" accept="image/*" style={{ display: "none" }} id={`img-${i}`}
                      onChange={(e) => e.target.files?.[0] && uploadImage(i, e.target.files[0])} />
                    <button className="btn tiny" onClick={() => document.getElementById(`img-${i}`)?.click()}>⬆ image</button>
                    <a className="btn tiny" href={`/api/projects/${proj.id}/scene/${i}/download`}>zip</a>
                  </div>
                  <div className="hint" style={{ marginTop: 3 }}>
                    {["voice_final", "video", "video_fit", "ambient"].map((k) => {
                      const r = stageFor(i, k);
                      return r ? <span key={k}>{k.replace("_final", "").replace("_fit", "")}:{r.status[0]} </span> : null;
                    })}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}

function hasChar(p: Project, s: Scene) { return !!p.character_id || !!s.meta?.character_id; }

export function VisualSourceControl({ value, hasChar, onChange }: {
  value: string; hasChar: boolean; onChange: (v: string) => void;
}) {
  const opts = [
    ["generated_video", "🎞 video"],
    ["illustration", "🖼 illustration"],
  ] as const;
  if (hasChar) opts.push(["character_demo", "🧑 gesture demo"] as const);
  return (
    <select value={value || "generated_video"} style={{ width: "auto", padding: "2px 6px", fontSize: 11 }}
      onChange={(e) => onChange(e.target.value)}>
      {opts.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
    </select>
  );
}

function ScriptPanel({ proj, onChanged, act, busy }: {
  proj: Project; onChanged: () => void; act: (v: string, p: string, b?: any, ok?: string) => Promise<any>; busy: string;
}) {
  const [text, setText] = useState(proj.script);
  const toast = useToast();
  useEffect(() => setText(proj.script), [proj.script]);
  const save = async () => {
    try {
      await api(`/projects/${proj.id}`, { method: "PATCH", json: { script: text, director_override: true } });
      toast("script updated", "ok"); onChanged();
    } catch (e) { toast(errText(e), "err"); }
  };
  return (
    <Panel title={proj.mode === "A" ? "Script (Director-locked)" : "Script (draft)"}
      right={<div className="row">
        {proj.mode === "B" && <button className="btn tiny"
          onClick={() => act("idea", `/projects/${proj.id}/regenerate-script`, {}, "new draft from the Controller")}
          title="ask the Controller for a fresh draft">regenerate script</button>}
        {proj.mode === "B" && <button className="btn tiny" onClick={() => act("idea", `/projects/${proj.id}/generate-idea`)}>auto-idea</button>}
        {proj.mode === "B" && <button className="btn tiny primary" onClick={() => act("approve", `/projects/${proj.id}/approve-script`, {}, "script approved")}>approve → run</button>}
        <button className="btn tiny" onClick={save}>save</button>
      </div>}>
      <textarea rows={6} style={{ margin: 10, width: "calc(100% - 20px)" }} value={text}
        onChange={(e) => setText(e.target.value)} />
      <div className="hint" style={{ padding: "0 12px 10px" }}>
        {proj.script_locked ? "Mode A: wording is verified against this paste — the studio never rewrites it." : "Mode B: edit before approving; production starts after approval."}
      </div>
    </Panel>
  );
}

function Waveform({ assetId }: { assetId: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const toast = useToast();
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const d = await api<{ peaks: number[]; duration?: number }>(`/assets/${assetId}/waveform`);
        const cv = ref.current;
        if (!cv || !alive) return;
        const ctx = cv.getContext("2d");
        if (!ctx) return;
        const w = (cv.width = (cv.clientWidth || 480) * (window.devicePixelRatio || 1));
        const h = (cv.height = 34 * (window.devicePixelRatio || 1));
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "var(--green)";
        const pk = d.peaks || [];
        const bw = w / Math.max(1, pk.length);
        pk.forEach((v, i) => {
          const hh = Math.max(1, Math.min(1, v) * h * 0.92);
          ctx.fillRect(i * bw, (h - hh) / 2, Math.max(1, bw * 0.7), hh);
        });
      } catch (e) {
        if (alive) toast(errText(e), "err");   // never hide an audible failure
      }
    })();
    return () => { alive = false; };
  }, [assetId]);
  return <canvas ref={ref} className="wave" title="audio waveform" style={{ height: 34, width: 140, flex: "0 0 140px" }} />;
}

function EventLog({ log, rows }: { log: any[]; rows: StageRow[] }) {
  return (
    <Panel title="Event log" scroll>
      <div className="log">
        {rows.filter((r) => r.error).map((r) => (
          <div key={r.id} className="l-err">[{r.stage}#{r.scene_idx}] {r.error}</div>
        ))}
        {rows.filter((r) => r.message).slice(-40).map((r) => (
          <div key={r.id} className={`l-${r.status === "failed" ? "err" : r.status === "done" ? "ok" : ""}`}>
            [{r.stage}#{r.scene_idx}] {r.status}: {r.message}
          </div>
        ))}
        {log.slice(-120).map((l, i) => (
          <div key={i} className={`l-${l.level || ""}`}>{(l.text || l.line || JSON.stringify(l)).slice(0, 400)}</div>
        ))}
      </div>
    </Panel>
  );
}

function Inspector({ proj, scenes, stage, scene, assets, rows, onChanged, act, busy, specs }: {
  proj: Project; scenes: Scene[]; stage: string; scene: number; assets: Asset[];
  rows: StageRow[]; onChanged: () => void; act: (v: string, p: string, b?: any, ok?: string) => Promise<any>;
  busy: string; specs: StageSpec[];
}) {
  const toast = useToast();
  const s = scenes[scene];
  const spec = specs.find((x) => x.key === stage);
  const stageRows = rows.filter((r) => r.stage === stage && r.scene_idx === scene);
  const row = stageRows[0];
  const a = (kind: string) => assets.find((x) => x.scene_idx === scene && x.kind === kind);
  const all = assets.find((x) => x.kind === "final");
  const zips = {};

  const regen = async () => {
    if (!proj.last_run_id && !row) { toast("no run yet — run the pipeline first, then regenerate per stage", "warn"); return; }
    const runId = row?.run_id || proj.last_run_id;
    try {
      const r = await api<{ run_id: string }>(`/runs/${runId}/stages/${stage}/regenerate`, {
        method: "POST", json: { scene_idx: scene },
      });
      toast(`regenerating ${stage} #${scene}…`, "ok"); onChanged();
    } catch (e) { toast(errText(e), "err"); }
  };

  const media = a("video_fit") || a("video") || a("talking_head");
  const voice = a("voice_final") || a("voice");
  const amb = a("ambient");
  const qa = a("qa");
  const instProps = (x: Asset | undefined) => x && (
    <details key={x.kind} style={{ marginBottom: 6 }}>
      <summary className="hint">{x.kind} · {x.engine || ""} · {fmtDur(x.duration)}</summary>
      <video controls preload="metadata" src={`/api/assets/${x.id}/stream`} style={{ marginTop: 4 }} />
      <div className="row" style={{ gap: 4, marginTop: 4 }}>
        <a className="btn tiny" href={`/api/assets/${x.id}/download`}>download</a>
        <a className="btn tiny" href={`/api/assets/${x.id}/stream`} target="_blank" rel="noreferrer">stream</a>
        {x.kind.startsWith("voice") || x.kind === "ambient" ? <Waveform assetId={x.id} /> : null}
      </div>
    </details>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <Panel title={`Scene ${scene + 1} · ${s?.text?.slice(0, 40) || "—"}`} scroll>
        {s ? (
          <div className="panel-b">
            <div className="kv">
              <dt>mood</dt><dd><Badge>{s.mood_tag || "—"}</Badge></dd>
              <dt>est / audio</dt><dd className="mono">{fmtDur(s.estimated_duration_sec)} / {fmtDur(s.audio_duration || s.estimated_duration_sec)}</dd>
              <dt>visual source</dt><dd>{s.meta?.visual_source || "generated_video"}</dd>
              <dt>render mode</dt><dd>{s.meta?.render_mode || "broll"}</dd>
              <dt>character</dt><dd>{s.meta?.character_id || proj.character_id || "none"}</dd>
              <dt>visual prompt</dt><dd className="hint">{s.visual_prompt || "—"}</dd>
              <dt>sfx</dt><dd className="hint">{s.sfx_prompt || "—"}</dd>
            </div>
            <div style={{ marginTop: 10 }}>
              {instProps(media)}{instProps(voice)}{instProps(amb)}
              {qa && (
                <details>
                  <summary className="hint">QA json</summary>
                  <pre className="fix" style={{ maxHeight: 220, overflow: "auto" }}>
                    {JSON.stringify(a("qa")?.meta, null, 1) || "—"}
                  </pre>
                </details>
              )}
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <button className="btn tiny" onClick={regen} disabled={!!busy}>⟳ regenerate {stage}#{scene}</button>
              <a className="btn tiny" href={`/api/runs/${row?.run_id || proj.last_run_id || ""}/scenes/${scene}/bundle`}>bundle</a>
            </div>
          </div>
        ) : <Empty text="no scene" />}
      </Panel>
      <Panel title={`Stage ${stage}`} scroll>
        <div className="panel-b">
          <div className="kv">
            <dt>label</dt><dd>{spec?.title || stage}</dd>
            <dt>resource</dt><dd>{spec?.resource} {spec?.requires_gpu ? "· GPU" : ""}</dd>
            <dt>model</dt><dd>{spec?.model || "—"}</dd>
            <dt>status</dt><dd>{row ? <StatusBadge status={row.status} /> : "not run"}</dd>
            <dt>engine</dt><dd>{row?.engine || "—"}</dd>
            <dt>attempts</dt><dd>{row?.attempt || 0}</dd>
            <dt>took</dt><dd>{row?.duration_ms ? `${(row.duration_ms / 1000).toFixed(1)}s` : "—"}</dd>
          </div>
          {row?.error && <div className="errbar" style={{ marginTop: 8 }}>⚠ {row.error}</div>}
          {row?.message && <div className="hint" style={{ marginTop: 8 }}>{row.message}</div>}
          <div style={{ marginTop: 8 }}>
            {scenes.map((x, i) => {
              const r = rows.find((y) => y.stage === stage && y.scene_idx === i);
              return (
                <div key={i} className={`scene-chip ${r ? "st-" + r.status : ""}`} style={{ display: "inline-block", margin: 3 }}>
                  {i + 1}{r ? ":" + r.status[0] : ""}
                </div>
              );
            })}
          </div>
        </div>
      </Panel>
      <Panel title="Final cut" scroll>
        <div className="panel-b">
          {all ? (
            <>
              <video controls preload="metadata" src={`/api/assets/${all.id}/stream`} />
              <div className="row" style={{ marginTop: 6 }}>
                <a className="btn tiny primary" href={`/api/assets/${all.id}/download`}>⬇ final mp4</a>
                <a className="btn tiny" href={`/api/projects/${proj.id}/download?kind=all`}>project zip</a>
                <a className="btn tiny" href={`/api/projects/${proj.id}/download?kind=bundle`}>.json</a>
              </div>
              <div className="hint" style={{ marginTop: 6 }}>{fmtDur(all.duration)} · {fmtSize(all.size_bytes)}</div>
            </>
          ) : <Empty text="no final cut yet" />}
        </div>
      </Panel>
    </div>
  );
}

function ModeBGate({ proj, onOpen, onChanged, act, busy }: {
  proj: Project; onOpen: (v: string, id?: string) => void; onChanged: () => void;
  act: (v: string, p: string, b?: any, ok?: string) => Promise<any>; busy: string;
}) {
  const [script, setScript] = useState(proj.script || "");
  const approve = async () => {
    const r = await act("approve", `/projects/${proj.id}/approve-script`, { script, start: true }, "approved — production running");
    if (r?.project) onChanged();
  };
  return (
    <div className="pad">
      <div className="errbar" style={{ background: "#2e2a1d" }}>
        ⚠ Auto mode: the Controller writes a draft; you must approve it before any GPU work starts.
      </div>
      <Panel title="Draft script (approve before production)">
        <div className="panel-b">
          <textarea rows={10} value={script} style={{ fontFamily: "inherit" }}
            onChange={(e) => setScript(e.target.value)} placeholder="generating…" />
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn primary" disabled={!script || !!busy}
              onClick={() => act("idea", `/projects/${proj.id}/generate-idea`)}>regenerate draft</button>
            <button className="btn primary" disabled={!script || !!busy} onClick={approve}>✅ approve & run</button>
          </div>
        </div>
      </Panel>
    </div>
  );
}
