import React, { useCallback, useEffect, useState } from "react";
import { api, Character, StatusPayload, StylePreview, VoiceProfile } from "../api";
import { useToast, errText } from "../main";
import { Badge, Empty, Panel, Spinner } from "../ui";

const URLS: Record<string, string> = {
  studio: "http://127.0.0.1:8000",
  ollama: "http://127.0.0.1:11434",
  rvc: "http://127.0.0.1:9513",
  comfy: "http://127.0.0.1:8188",
};
export const SERVICES = {
  studio: { name: "Studio API", port: "8000", key: "studio" },
  ollama: { name: "Ollama", port: "11434", key: "ollama" },
  rvc: { name: "RVC", port: "9513", key: "rvc" },
  comfy: { name: "ComfyUI", port: "8188", key: "comfy" },
};

export function AdminView({ tab, status, onNavigate }: {
  tab: string; status: StatusPayload | null; onNavigate: (v: string, id?: string) => void;
}) {
  if (tab === "services") return <Services status={status} />;
  if (tab === "team") return <Team status={status} />;
  if (tab === "plugins") return <Plugins status={status} />;
  if (tab === "characters") return <Characters />;
  if (tab === "voices") return <Voices />;
  if (tab === "settings") return <Settings status={status} />;
  if (tab === "memory") return <Memory />;
  return null;
}

// ---------------------------------------------------------------- Services
function Services({ status }: { status: StatusPayload | null }) {
  const [probe, setProbe] = useState<any>(null);
  const [fix, setFix] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const deep = async () => {
    setBusy(true);
    try {
      const [p, f] = await Promise.all([
        api("/settings/probe"),
        api("/status", { query: { deep: true } }).catch(() => null),
      ]);
      setProbe(p);
      setFix(f);
    } catch (e) { toast(errText(e), "err"); }
    setBusy(false);
  };
  useEffect(() => { deep(); }, []);

  const services = [
    { key: "studio", name: "Studio API", url: URLS.studio,
      ok: !!status, hint: "this server", det: `v${status?.version || "?"} · ${status?.data_dir || ""}` },
    { key: "ollama", name: "Ollama", url: URLS.ollama,
      ok: !!(status?.capabilities?.ollama), hint: "llm roles", det: status?.plan?.ollama?.engine || "offline" },
    { key: "rvc", name: "RVC (Applio/RVC-WebUI)", url: URLS.rvc,
      ok: !!(status?.capabilities?.rvc_http || status?.capabilities?.rvc_cli),
      hint: "timbre conversion", det: status?.plan?.rvc?.reason || "" },
    { key: "comfy", name: "ComfyUI", url: URLS.comfy,
      ok: !!(status?.capabilities?.comfyui), hint: "Wan · MMAudio · FLUX", det: status?.plan?.video?.reason || "" },
  ];

  return (
    <div className="pad">
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2>Services</h2>
        <button className="btn" onClick={deep} disabled={busy}>{busy ? <Spinner /> : "⟳ re-probe"}</button>
      </div>
      {services.map((s) => (
        <div key={s.key} className="svc">
          <span className="statusdot" style={{ background: s.ok ? "var(--green)" : "var(--red)" }} />
          <div className="s-name">{s.name}</div>
          <div className="grow">
            <div className="s-url">{s.url} <span className="hint">— {s.hint}</span></div>
            <div className="hint">{s.det}</div>
          </div>
          <a className="btn tiny" href={s.url} target="_blank" rel="noreferrer">open ↗</a>
        </div>
      ))}
      <div className="hint" style={{ marginTop: 10 }}>
        Fix commands are produced by the backend's own <code>python -m ai_studio --check</code> — see the
        Settings tab or run it in the studio directory. The exact text is surfaced verbatim above each engine.
      </div>
      {probe && <FixPanel probe={probe} status={status} />}
    </div>
  );
}

function FixPanel({ probe, status }: { probe: any; status: StatusPayload | null }) {
  const cmd = (label: string, lines: string[]) => lines.length ? (
    <div key={label} style={{ marginTop: 8 }}>
      <b className="hint">{label}</b>
      <pre className="fix">{lines.join("\n")}</pre>
    </div>
  ) : null;
  return (
    <Panel title="Backend setup commands (verbatim from --check)" scroll>
      <div className="panel-b">
        {cmd("TTS model", probe?.tts?.fix || [])}
        {cmd("RVC", probe?.rvc?.fix || [])}
        {cmd("Video / ComfyUI", probe?.video?.fix || [])}
        {cmd("SFX", probe?.sfx?.fix || [])}
        {cmd("Talking head", probe?.talking_head?.fix || [])}
        {cmd("Illustration", probe?.illustration?.fix || [])}
        {(probe?.tts?.fix || probe?.rvc?.fix || probe?.video?.fix || probe?.sfx?.fix ||
          probe?.talking_head?.fix || probe?.illustration?.fix) ? null :
          <Empty text="backend reported no fix commands (all engines resolve)" />}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- AI Team
function Team({ status }: { status: StatusPayload | null }) {
  const [settings, setSettings] = useState<any>(null);
  const [models, setModels] = useState<string[]>([]);
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const toast = useToast();
  const load = useCallback(async () => {
    try {
      const s = await api("/settings");
      setSettings(s);
      const v = await api<{ voices: VoiceProfile[] }>("/voices");
      setVoices(v.voices || []);
    } catch (e) { toast(errText(e), "err"); }
    try { const m = await api<{ models: string[] }>("/ollama/models"); setModels(m.models || []); }
    catch { setModels([]); }
  }, [toast]);
  useEffect(() => { load(); }, [load]);

  if (!settings) return <div className="pad"><Spinner /></div>;
  const roles = settings.llm_roles?.keys || ["controller", "auto_idea", "qa"];

  const saveRole = async (role: string, patch: any) => {
    const cur = { ...settings };
    cur.ollama.roles[role] = { ...cur.ollama.roles[role], ...patch };
    try { await api("/settings", { method: "POST", json: cur }); toast(`${role}: saved`, "ok"); load(); }
    catch (e) { toast(errText(e), "err"); }
  };
  const saveEngine = async (section: string, engine: string) => {
    const cur = { ...settings };
    cur[section] = { ...cur[section], engine };
    try { await api("/settings", { method: "POST", json: cur }); toast(`${section} engine: ${engine}`, "ok"); load(); }
    catch (e) { toast(errText(e), "err"); }
  };

  return (
    <div className="pad">
      <h2 style={{ marginBottom: 10 }}>AI Team</h2>
      <p className="hint" style={{ marginBottom: 12 }}>
        Per-role model assignment for the three agents, and the engine pick for every production stage.
        Options come from the backend (Ollama's installed models, the studio's own engine list).
        Changes save immediately.
      </p>
      <div className="spread" style={{ marginBottom: 8 }}><h3>LLM roles (Ollama)</h3></div>
      {roles.map((role: string) => {
        const rc = settings.ollama?.roles?.[role] || {};
        return (
          <div key={role} className="svc">
            <input type="checkbox" checked={!!rc.enabled} style={{ width: "auto" }}
              onChange={(e) => saveRole(role, { enabled: e.target.checked })} />
            <div className="s-name">{settings.llm_roles?.labels?.[role] || role}</div>
            <select value={rc.model || ""} onChange={(e) => saveRole(role, { model: e.target.value })} style={{ width: 220 }}>
              {(models.length ? models : [rc.model || "sailor2:8b", rc.fallback_model]).map((m) =>
                <option key={m} value={m}>{m}</option>)}
            </select>
            <input value={rc.temperature ?? 0.6} type="number" step={0.05} style={{ width: 80 }}
              onChange={(e) => saveRole(role, { temperature: parseFloat(e.target.value) })} />
            <span className="hint">temp</span>
          </div>
        );
      })}
      <div className="spread" style={{ margin: "16px 0 8px" }}><h3>Engines</h3></div>
      {(["tts", "rvc", "video", "sfx", "talking_head", "illustration"] as const).map((sec) => {
        const opts: Record<string, string[]> = {
          tts: ["auto", "sherpa", "piper", "kokoro", "placeholder"],
          rvc: ["auto", "http", "cli", "bypass"],
          video: ["auto", "comfyui", "previz", "defer", "off"],
          sfx: ["auto", "mmaudio", "procedural", "defer", "off"],
          talking_head: ["auto", "sadtalker", "still"],
          illustration: ["auto", "comfyui", "pil"],
        };
        const plan = status?.plan?.[sec];
        return (
          <div key={sec} className="svc">
            <div className="s-name">{sec}</div>
            <select value={settings.settings?.[sec]?.engine || "auto"} style={{ width: 160 }}
              onChange={(e) => saveEngine(sec, e.target.value)}>
              {opts[sec].map((o) => <option key={o}>{o}</option>)}
            </select>
            <div className="grow hint">
              {plan?.engine ? `resolved: ${plan.engine}` : "not resolved yet"}{plan?.reason ? ` — ${plan.reason}` : ""}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- Plugins
function Plugins({ status }: { status: StatusPayload | null }) {
  const [wfs, setWfs] = useState<any[]>([]);
  const [previews, setPreviews] = useState<{ sub: StylePreview[]; title: StylePreview[] }>({ sub: [], title: [] });
  const toast = useToast();
  useEffect(() => {
    api<{ workflows: any[] }>("/workflows").then((r) => setWfs(r.workflows || [])).catch((e) => toast(errText(e), "err"));
    api<{ subtitle_styles: StylePreview[]; title_styles: StylePreview[] }>("/style-previews")
      .then((r) => setPreviews({ sub: r.subtitle_styles || [], title: r.title_styles || [] })).catch(() => {});
  }, [toast]);
  const caps = status?.capabilities || {};
  const engines = [
    ["sherpa TTS", caps.sherpa_tts, "models/tts/vits-mms-khm (model.onnx + tokens.txt)"],
    ["sherpa python", !!caps.sherpa_python, "pip install sherpa-onnx"],
    ["RVC http", !!caps.rvc_http, "start Applio/RVC-WebUI on :9513"],
    ["RVC cli", !!caps.rvc_cli, "set rvc.webui_dir in settings"],
    ["ComfyUI", !!caps.comfyui, "start ComfyUI on :8188"],
    ["SadTalker", !!caps.sadtalker, "set talking_head.sadtalker_dir (infer.py checkout)"],
    ["Ollama", !!caps.ollama, "ollama serve on :11434"],
    ["ffmpeg", !!caps.ffmpeg, "imageio-ffmpeg bundled; Windows: ffmpeg on PATH"],
  ] as const;
  return (
    <div className="pad">
      <h2 style={{ marginBottom: 10 }}>Plugins / Engines</h2>
      <div className="cards" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(300px,1fr))" }}>
        {engines.map(([name, ok, hint]) => (
          <div key={name} className="svc" style={{ margin: 0 }}>
            <span className="statusdot" style={{ background: ok ? "var(--green)" : "var(--red)" }} />
            <div className="grow"><b>{name}</b><div className="hint">{hint}</div></div>
            <Badge kind={ok ? "ok" : "err"}>{ok ? "ready" : "missing"}</Badge>
          </div>
        ))}
      </div>
      <h3 style={{ margin: "16px 0 8px" }}>ComfyUI workflows (backend templates)</h3>
      <table className="grid">
        <thead><tr><th>name</th><th>placeholders</th><th>dir</th></tr></thead>
        <tbody>{wfs.map((w) => (
          <tr key={w.path}><td className="mono">{w.name}</td>
            <td className="mono">{(w.placeholders || []).join(", ")}</td>
            <td className="hint">{w.dir}</td></tr>
        ))}</tbody>
      </table>
      <h3 style={{ margin: "16px 0 8px" }}>Subtitle / title styles</h3>
      <div className="gallery">
        {previews.sub.map((s) => (
          <div key={s.key} className="g-item">
            {s.url ? <video src={s.url} muted loop autoPlay playsInline /> : <div className="hint" style={{ padding: 20 }}>unavailable</div>}
            <div className="g-cap"><b>{s.label}</b>{s.error ? <Badge kind="err">broken</Badge> : null}</div>
          </div>
        ))}
        {previews.title.map((s) => (
          <div key={s.key} className="g-item">
            {s.url ? <video src={s.url} muted loop autoPlay playsInline /> : <div className="hint" style={{ padding: 20 }}>unavailable</div>}
            <div className="g-cap">title · <b>{s.label}</b></div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Characters
function Characters() {
  const [chars, setChars] = useState<Character[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [moodMap, setMoodMap] = useState<Record<string, string>>({});
  const [sel, setSel] = useState<string>("");
  const toast = useToast();
  const load = useCallback(async () => {
    try {
      const r = await api<{ characters: Character[]; expression_labels: string[]; mood_to_expression: Record<string, string> }>("/characters");
      setChars(r.characters || []); setLabels(r.expression_labels || []); setMoodMap(r.mood_to_expression || {});
      if (!sel && r.characters?.length) setSel(r.characters[0].id);
    } catch (e) { toast(errText(e), "err"); }
  }, [sel, toast]);
  useEffect(() => { load(); }, [load]);

  const create = async () => {
    const name = prompt("Character name:");
    if (!name) return;
    try { await api("/characters", { method: "POST", json: { name } }); toast("character created", "ok"); load(); }
    catch (e) { toast(errText(e), "err"); }
  };
  const upload = async (cid: string, label: string, f: File) => {
    const fd = new FormData();
    fd.append("expression_label", label); fd.append("image", f);
    try { await api(`/characters/${cid}/images`, { method: "POST", form: fd }); toast(`image (${label}) uploaded`, "ok"); load(); }
    catch (e) { toast(errText(e), "err"); }
  };
  const del = async (cid: string) => {
    if (!confirm("delete character?")) return;
    try { await api(`/characters/${cid}`, { method: "DELETE" }); toast("deleted", "ok"); setSel(""); load(); }
    catch (e) { toast(errText(e), "err"); }
  };
  const cur = chars.find((c) => c.id === sel);

  return (
    <div className="pad">
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2>Characters (NPCs)</h2>
        <button className="btn primary" onClick={create}>+ new character</button>
      </div>
      <div className="split" style={{ gridTemplateColumns: "280px 1fr" }}>
        <Panel title="Characters" scroll>
          {chars.map((c) => (
            <div key={c.id} className="svc" style={{ cursor: "pointer", margin: 0, borderBottom: "1px solid var(--line)", borderRadius: 0 }}
              onClick={() => setSel(c.id)}>
              <b className="grow">{c.name}</b>
              <span className="hint">{c.images.length} imgs</span>
              <button className="btn tiny danger" onClick={(e) => { e.stopPropagation(); del(c.id); }}>×</button>
            </div>
          ))}
          {!chars.length && <Empty text="no characters yet" />}
        </Panel>
        <div>
          {cur ? (
            <Panel title={`${cur.name} — expressions`} scroll>
              <div className="panel-b">
                <p className="hint" style={{ marginBottom: 10 }}>
                  Mood → expression matching (from the backend): the scene's mood_tag maps to the
                  nearest uploaded label (e.g. {Object.entries(moodMap).slice(0, 3).map(([m, l]) => `${m}→${l}`).join(", ")}…).
                </p>
                {labels.map((l) => {
                  const img = cur.images.find((i) => i.expression_label.toLowerCase() === l.toLowerCase());
                  return (
                    <div key={l} className="svc">
                      <div style={{ width: 54, height: 66, background: "#0f1013", borderRadius: 4, overflow: "hidden" }}>
                        {img?.url ? <img src={img.url} style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : "—"}
                      </div>
                      <b className="grow">{l}</b>
                      <input type="file" accept="image/*" style={{ display: "none" }}
                        id={`img-${l}`} onChange={(e) => e.target.files?.[0] && upload(cur.id, l, e.target.files[0])} />
                      <button className="btn tiny" onClick={() => document.getElementById(`img-${l}`)?.click()}>
                        {img ? "replace" : "upload"}
                      </button>
                      {img && <button className="btn tiny danger" onClick={async () => {
                        await api(`/characters/${cur.id}/images/${img.id}`, { method: "DELETE" }); load();
                      }}>×</button>}
                    </div>
                  );
                })}
              </div>
            </Panel>
          ) : <Empty text="select or create a character" />}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Voices
function Voices() {
  const [data, setData] = useState<{ voices: VoiceProfile[]; discovered: any[] } | null>(null);
  const toast = useToast();
  const load = useCallback(async () => {
    try { setData(await api("/voices")); }
    catch (e) { toast(errText(e), "err"); }
  }, [toast]);
  useEffect(() => { load(); }, [load]);
  if (!data) return <div className="pad"><Spinner /></div>;

  const act = async (fn: () => Promise<any>, ok: string) => {
    try { await fn(); toast(ok, "ok"); load(); } catch (e) { toast(errText(e), "err"); }
  };

  return (
    <div className="pad">
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2>Voice profiles (RVC)</h2>
        <button className="btn primary" onClick={() => {
          const name = prompt("voice name:"); if (!name) return;
          const fd = new FormData(); fd.append("name", name);
          act(() => api("/voices", { method: "POST", form: fd }), "voice registered (add .pth/sample later)");
        }}>+ new voice</button>
        <button className="btn" onClick={() => act(() => api("/voices/import-discovered", { method: "POST" }), "imported discovered voices")}>
          ⟳ import discovered
        </button>
      </div>
      <table className="grid">
        <thead><tr><th>name</th><th>engine</th><th>pitch</th><th>pth</th><th>sample</th><th>training</th><th /></tr></thead>
        <tbody>
          {data.voices.map((v) => (
            <tr key={v.id}>
              <td><b>{v.name}</b></td><td>{v.engine}</td><td className="mono">{v.pitch}</td>
              <td>{v.pth_exists ? "✓" : "—"}</td>
              <td className="mono">{v.sample_seconds ? `${v.sample_seconds}s` : "—"}</td>
              <td>{v.training_status || ""}</td>
              <td><div className="row">
                <a className="btn tiny" href={v.sample_url || "#"}>sample</a>
                <button className="btn tiny" onClick={() => act(() => api(`/voices/${v.id}/train`, { method: "POST" }), "training started")}>train</button>
                <button className="btn tiny" onClick={() => act(() => api(`/voices/${v.id}/select`, { method: "POST" }), "set as house voice")}>use</button>
                <button className="btn tiny" onClick={() => act(() => api(`/voices/${v.id}?purge_files=true`, { method: "DELETE" }), "deleted")}>×</button>
              </div></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- Settings
function Settings({ status }: { status: StatusPayload | null }) {
  const [s, setS] = useState<any>(null);
  const toast = useToast();
  const load = useCallback(async () => {
    try { setS(await api("/settings")); } catch (e) { toast(errText(e), "err"); }
  }, [toast]);
  useEffect(() => { load(); }, [load]);
  if (!s) return <div className="pad"><Spinner /></div>;
  const cfg = s.settings || {};
  const save = async (patch: any) => {
    try { await api("/settings", { method: "POST", json: mergeSettings(cfg, patch) }); toast("settings saved", "ok"); load(); }
    catch (e) { toast(errText(e), "err"); }
  };
  const num = (k: string, v: any, path: string[]) => (
    <label className="fld"><span>{k}</span>
      <input type="number" value={v} onChange={(e) => save({ [path[0]]: { [path[1]]: parseFloat(e.target.value) } })} />
    </label>
  );
  return (
    <div className="pad">
      <h2 style={{ marginBottom: 10 }}>Settings</h2>
      <div className="split">
        <Panel title="Machine / VRAM" scroll>
          <div className="panel-b">
            <label className="fld"><span>machine profile</span>
              <select value={cfg.machine?.profile || "auto"} onChange={(e) => save({ machine: { ...cfg.machine, profile: e.target.value } })}>
                {Object.entries(s.machine_profiles || {}).map(([k, v]: any) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </label>
            {num("VRAM limit (MB)", cfg.vram?.limit_mb, ["vram", "limit_mb"])}
            <div className="hint">{s.vram?.detected ? `detected: ${JSON.stringify(s.vram.detected)}` : ""}</div>
            <label className="fld"><span>serialize GPU jobs (8GB safety)</span>
              <input type="checkbox" checked={cfg.vram?.serialize_gpu !== false}
                onChange={(e) => save({ vram: { ...cfg.vram, serialize_gpu: e.target.checked } })} style={{ width: "auto" }} />
            </label>
          </div>
        </Panel>
        <Panel title="Pipeline / Assembly / Styles" scroll>
          <div className="panel-b">
            <label className="fld"><span>scene target seconds</span>
              <input type="number" value={cfg.pipeline?.scene_target_seconds} onChange={(e) => save({ pipeline: { ...cfg.pipeline, scene_target_seconds: parseFloat(e.target.value) } })} />
            </label>
            <label className="fld"><span>max scenes</span>
              <input type="number" value={cfg.pipeline?.max_scenes} onChange={(e) => save({ pipeline: { ...cfg.pipeline, max_scenes: parseInt(e.target.value) } })} />
            </label>
            <label className="fld"><span>burn captions (default)</span>
              <input type="checkbox" checked={!!cfg.assembly?.burn_captions}
                onChange={(e) => save({ assembly: { ...cfg.assembly, burn_captions: e.target.checked } })} style={{ width: "auto" }} />
            </label>
            <label className="fld"><span>subtitle style</span>
              <select value={cfg.assembly?.subtitle_style || "clean"} onChange={(e) => save({ assembly: { ...cfg.assembly, subtitle_style: e.target.value } })}>
                {Object.entries(s.subtitle_styles || {}).map(([k, v]: any) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </label>
            <label className="fld"><span>title style (empty = none)</span>
              <select value={cfg.assembly?.title_style || ""} onChange={(e) => save({ assembly: { ...cfg.assembly, title_style: e.target.value } })}>
                <option value="">none</option>
                {Object.entries(s.title_styles || {}).map(([k, v]: any) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </label>
            <label className="fld"><span>tts pace</span>
              <select value={cfg.tts?.pace || "natural"} onChange={(e) => save({ tts: { ...cfg.tts, pace: e.target.value } })}>
                {Object.entries(s.pace_presets || {}).map(([k, v]: any) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </label>
            <label className="fld"><span>line gap seconds (0.3–3.0)</span>
              <input type="number" min={0.3} max={3} step={0.05} value={cfg.tts?.line_gap_sec} onChange={(e) => save({ tts: { ...cfg.tts, line_gap_sec: parseFloat(e.target.value) } })} />
            </label>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function mergeSettings(base: any, patch: any): any {
  const out: any = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    if (v && typeof v === "object" && !Array.isArray(v) && base[k] && typeof base[k] === "object") {
      out[k] = { ...base[k], ...v };
    } else out[k] = v;
  }
  return out;
}

// ---------------------------------------------------------------- Memory
function Memory() {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<any>(null);
  const toast = useToast();
  const search = async (query: string) => {
    setQ(query);
    try { setRows(await api("/memory/search", { query: { q: query, limit: 60 } })); }
    catch (e) { toast(errText(e), "err"); }
  };
  useEffect(() => { search(""); }, []);
  return (
    <div className="pad">
      <h2 style={{ marginBottom: 10 }}>Memory</h2>
      <input placeholder="search every script, prompt and scene ever stored…"
        value={q} onChange={(e) => search(e.target.value)} style={{ maxWidth: 480, marginBottom: 12 }} />
      {rows && (
        <div className="split" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <Panel title="Projects" scroll><div className="panel-b">
            {(rows.projects || []).map((p: any) => (
              <div key={p.id} className="hint" style={{ padding: "4px 0", borderBottom: "1px solid var(--line)" }}>
                <b>{p.title}</b> ({p.mode}, {p.status}) — {p.excerpt?.slice(0, 80)}
              </div>
            ))}
          </div></Panel>
          <Panel title="Prompts" scroll><div className="panel-b">
            {(rows.prompts || []).map((p: any) => (
              <div key={p.id} className="hint" style={{ padding: "4px 0", borderBottom: "1px solid var(--line)" }}>
                [{p.stage} · {p.role} · {p.model}] {p.user_excerpt || p.response_excerpt || ""}
              </div>
            ))}
          </div></Panel>
        </div>
      )}
    </div>
  );
}
