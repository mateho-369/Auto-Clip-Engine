import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, wsUrl } from './api';

/* ------------------------------------------------------------------ types */
type View = 'project' | 'team' | 'plugins' | 'services';
type StageStatus = 'pending' | 'queued' | 'running' | 'done' | 'failed' | 'skipped' | 'deferred' | 'blocked' | 'cancelled';

interface ProjectRow {
  id: string;
  title: string;
  mode: string;
  status: string;
  content_type: string;
  language?: string;
  updated_at?: number;
  target_duration?: number;
  scene_count?: number;
  run_count?: number;
  last_run_status?: string;
  script_excerpt?: string;
}

interface ContentTypeCard {
  key: string;
  label: string;
  description: string;
  default_duration?: number;
}

interface SettingsData {
  settings: Record<string, any>;
  plan?: Record<string, any>;
  roles?: any[];
  llm_roles?: { keys: string[]; labels: Record<string, string> };
  content_types: ContentTypeCard[];
  style_guideline?: string;
  defaults?: Record<string, any>;
}

interface ProjectDetail {
  project: Record<string, any>;
  scenes: Record<string, any>[];
  runs: Record<string, any>[];
  latest_run_id: string;
  assets: Record<string, any>[];
  prompts: Record<string, any>[];
  integrity: Record<string, any>;
  disk?: Record<string, any>;
}

interface RunSnapshot {
  run?: Record<string, any>;
  stages?: Record<string, any>[];
  by_stage?: Record<string, any>;
  overall?: Record<string, any>;
  graph?: { STAGES?: any[]; stages?: any[]; nodes?: any[] };
  log?: any[];
  final?: Record<string, any>;
  assets?: any[];
  plan?: Record<string, any>;
}

interface StatusData {
  studio: string;
  version: string;
  data_dir: string;
  db: Record<string, any>;
  machine: Record<string, any>;
  plan: Record<string, any>;
  active_runs: string[];
  ffmpeg?: string;
}

interface ProbeData {
  capabilities?: Record<string, boolean>;
  plan?: Record<string, any>;
}

const STAGE_ORDER = ['script', 'breakdown', 'voice_base', 'voice_final', 'video', 'video_fit', 'sfx', 'qa', 'assemble'];

function labelFor(tag: string): string {
  const map: Record<string, string> = {
    script: 'Script',
    breakdown: 'Breakdown',
    voice_base: 'Khmer voice',
    voice_final: 'Your timbre',
    video: 'Animator',
    video_fit: 'Duration match',
    sfx: 'SFX',
    qa: 'QA',
    assemble: 'Assembly',
  };
  return map[tag] || tag;
}

function assetKindLabel(kind: string): string {
  return kind.replace(/_/g, ' ');
}

function statusClass(status: string): string {
  if (['done', 'skipped', 'deferred'].includes(status)) return 'ok';
  if (['failed', 'blocked', 'cancelled'].includes(status)) return 'err';
  if (['running', 'queued'].includes(status)) return 'run';
  return 'muted';
}

function formatDate(ts?: number): string {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString();
}

function isRunning(status?: string): boolean {
  return ['queued', 'running', 'paused', 'needs_review'].includes(status || '');
}

export default function App() {
  const [projects, setProjects] = useState<ProjectRow[]>([]);
  const [selected, setSelected] = useState<string>('');
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [settings, setSettings] = useState<SettingsData | null>(null);
  const [status, setStatus] = useState<StatusData | null>(null);
  const [probe, setProbe] = useState<ProbeData | null>(null);
  const [workflows, setWorkflows] = useState<any[]>([]);
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [view, setView] = useState<View>('project');
  const [search, setSearch] = useState('');
  const [selectedScene, setSelectedScene] = useState<number | null>(null);
  const [selectedStage, setSelectedStage] = useState<string>('');
  const [toasts, setToasts] = useState<{ id: number; kind: 'info' | 'ok' | 'err'; text: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ mode: 'A', content_type: 'explainer', script: '', topic_hint: '', target_duration: 30, style_notes: '' });
  const [ngWord, setNgWord] = useState('');
  const [ngTopic, setNgTopic] = useState('');
  const toastId = useRef(0);

  const toast = useCallback((kind: 'info' | 'ok' | 'err', text: string) => {
    const id = ++toastId.current;
    setToasts((ts) => [...ts, { id, kind, text }]);
    window.setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), kind === 'err' ? 9000 : 5000);
  }, []);

  const request = useCallback(async <T,>(fn: () => Promise<T>, fallback: string): Promise<T | null> => {
    try {
      return await fn();
    } catch (e: any) {
      toast('err', `${fallback}: ${e.message || e}`);
      return null;
    }
  }, [toast]);

  const loadProjects = useCallback(async () => {
    const data = await request(() => api.get('/api/projects'), 'Could not load projects');
    if (data) setProjects((data as any).projects || []);
  }, [request]);

  const loadDetail = useCallback(async (id: string) => {
    const data = await request(() => api.get(`/api/projects/${id}`), 'Could not load project');
    if (data) {
      setDetail(data as ProjectDetail);
      setRun(null);
      if ((data as ProjectDetail).latest_run_id) {
        const snap = await request(() => api.get(`/api/runs/${(data as ProjectDetail).latest_run_id}/status`), 'Could not load run');
        if (snap) setRun(snap as RunSnapshot);
      }
    }
  }, [request]);

  const loadSettings = useCallback(async () => {
    const data = await request(() => api.get('/api/settings'), 'Could not load settings');
    if (data) setSettings(data as SettingsData);
  }, [request]);

  const loadStatus = useCallback(async () => {
    const data = await request(() => api.get('/api/status'), 'Could not load studio status');
    if (data) setStatus(data as StatusData);
  }, [request]);

  const loadWorkflows = useCallback(async () => {
    const data = await request(() => api.get('/api/workflows'), 'Could not load workflows');
    if (data) setWorkflows((data as any).workflows || []);
  }, [request]);

  const openNewProject = useCallback(() => {
    setSelected('');
    setDetail(null);
    setRun(null);
    setSelectedScene(null);
    setSelectedStage('');
    setForm({
      mode: 'A',
      content_type: form.content_type || 'explainer',
      script: '',
      topic_hint: '',
      target_duration: 30,
      style_notes: '',
    });
    setView('project');
  }, [form.content_type]);

  useEffect(() => {
    (async () => {
      await Promise.all([loadProjects(), loadSettings(), loadStatus(), loadWorkflows()]);
    })();
  }, [loadProjects, loadSettings, loadStatus, loadWorkflows]);

  useEffect(() => {
    if (selected) void loadDetail(selected);
  }, [selected, loadDetail]);

  // keep project list fresh
  useEffect(() => {
    const t = window.setInterval(() => { void loadProjects(); void loadStatus(); }, 15000);
    return () => window.clearInterval(t);
  }, [loadProjects, loadStatus]);

  const activeRunId = detail?.latest_run_id && isRunning(detail.runs?.[0]?.status) ? detail.latest_run_id : '';

  // websocket for the active run
  useEffect(() => {
    if (!activeRunId) return;
    const ws = new WebSocket(wsUrl(`/api/runs/${activeRunId}/events`));
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.kind === 'snapshot') {
          setRun(msg.payload || null);
        } else if (msg.kind === 'stage_failed') {
          toast('err', `${msg.payload?.stage || ''} ${msg.payload?.scene_idx !== undefined ? '# ' + msg.payload.scene_idx : ''}: ${msg.payload?.error || 'stage failed'}`);
        } else if (msg.kind === 'stage_done') {
          toast('ok', `${msg.payload?.stage || ''} ${msg.payload?.scene_idx !== undefined ? 'scene ' + msg.payload.scene_idx : ''} completed`);
        }
        // a lightweight re-fetch is cheaper and keeps stage statuses authoritative
        void (async () => {
          const snap = await api.get(`/api/runs/${activeRunId}/status`).catch(() => null);
          if (snap) setRun(snap as RunSnapshot);
        })();
      } catch { /* no-op */ }
    };
    ws.onerror = () => toast('err', 'Live pipeline socket disconnected; polling will continue');
    ws.onclose = () => {
      // reconnect once with backoff in a later render cycle via detail re-fetch
      toast('info', 'Live pipeline socket closed; polling active run');
    };
    return () => ws.close();
  }, [activeRunId, toast]);

  const refreshServices = useCallback(async () => {
    const p = await request(() => api.post('/api/settings/probe', {}), 'Service probe failed');
    if (p) {
      setProbe(p as ProbeData);
      toast('ok', 'Service probe refreshed');
    }
  }, [request, toast]);

  async function createProject() {
    setLoading(true);
    const payload = {
      mode: form.mode,
      content_type: form.content_type,
      script: form.script,
      topic_hint: form.topic_hint,
      target_duration: Number(form.target_duration || 30),
      style_notes: form.style_notes,
      generate_now: false,
    };
    const data = await request(() => api.post('/api/projects', payload), 'Create project failed');
    if (data) {
      const p = (data as any).project;
      toast('ok', `Created ${p.title}`);
      setProjects((prev) => [p, ...prev]);
      setSelected(p.id);
      setForm({ mode: form.mode, content_type: form.content_type, script: '', topic_hint: '', target_duration: form.target_duration, style_notes: '' });
      if (form.mode === 'B') {
        const gen = await request(() => api.post(`/api/projects/${p.id}/generate-idea`, {}), 'Auto-idea generation failed');
        if (gen) toast('ok', `Generated idea: ${(gen as any).script ? 'ready to review' : 'script generated'}`);
        void loadDetail(p.id);
      }
    }
    setLoading(false);
  }

  async function updateProject(patch: Record<string, any>, success = 'Saved') {
    if (!selected) return;
    const data = await request(() => api.patch(`/api/projects/${selected}`, patch), 'Save failed');
    if (data) {
      toast('ok', success);
      await loadDetail(selected);
    }
  }

  async function startRun(force: string[] = []) {
    if (!selected) return;
    const data = await request(() => api.post(`/api/projects/${selected}/runs`, { force_stages: force.length ? force : null }), 'Start run failed');
    if (data) {
      toast('ok', `Run started (${(data as any).jobs} jobs)`);
      toast('info', 'Pipeline is queued; watch the graph for live stage status');
      await loadDetail(selected);
    }
  }

  async function controlRun(op: 'pause' | 'resume' | 'cancel') {
    const id = activeRunId;
    if (!id) return;
    const data = await request(() => api.post(`/api/runs/${id}/${op}`, {}), `Run ${op} failed`);
    if (data) {
      toast('ok', `Run ${op}${(data as any).continued ? ' (continued)' : ''}`);
      await loadDetail(selected);
    }
  }

  async function regenerateStage(stage: string, scene_idx = -1) {
    const id = detail?.latest_run_id;
    if (!id) {
      toast('err', 'No completed run to regenerate from');
      return;
    }
    const data = await request(() => api.post(`/api/runs/${id}/stages/${stage}/regenerate`, { scene_idx }), `Regenerate ${stage} failed`);
    if (data) {
      toast('ok', `Regenerated ${stage}${scene_idx >= 0 ? ' scene ' + scene_idx : ''}`);
      toast('info', 'A new run has been created; its output will replace the old one when it finishes');
      await loadDetail(selected);
    }
  }

  const filteredProjects = useMemo(() => {
    const q = (search || '').toLowerCase();
    const list = q ? projects.filter((p) => `${p.title} ${p.status} ${p.content_type} ${p.mode}`.toLowerCase().includes(q)) : projects;
    return [...list].sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
  }, [projects, search]);

  const services = useMemo(() => {
    const c = settings?.settings || {};
    const caps = probe?.capabilities || {};
    return [
      { name: 'Studio backend', url: window.location.origin, desc: 'This app’s own API and UI', status: 'up', fix: '' },
      { name: 'Ollama', url: c.ollama?.host || 'http://127.0.0.1:11434', desc: 'Controller / scriptwriter / QA language models', status: caps.ollama === true ? 'up' : caps.ollama === false ? 'down' : 'unknown', fix: 'Start it with: ollama serve' },
      { name: 'RVC-WebUI inference', url: c.rvc?.api_base || 'http://127.0.0.1:9513', desc: 'Converts base Khmer speech into the Director’s own cloned voice', status: caps.rvc === true ? 'up' : caps.rvc === false ? 'down' : 'unknown', fix: 'Start RVC WebUI on port 9513' },
      { name: 'ComfyUI', url: c.video?.comfy_host || 'http://127.0.0.1:8188', desc: 'Runs the video model (Wan2.1/2.2) and the SFX model (MMAudio)', status: caps.comfyui === true ? 'up' : caps.comfyui === false ? 'down' : 'unknown', fix: 'Start ComfyUI on port 8188' },
    ];
  }, [settings, probe]);

  const stageRows = useMemo(() => run?.stages || [], [run]);
  const stageMap = useMemo(() => {
    const m: Record<string, Record<string, any>[]> = {};
    for (const r of stageRows) (m[r.stage] ||= []).push(r);
    return m;
  }, [stageRows]);

  const selectedSceneData = detail?.scenes?.find((s) => Number(s.idx) === selectedScene) || null;
  const selectedAssetKinds = useMemo(() => {
    const kinds: Record<string, any[]> = {};
    for (const a of detail?.assets || []) {
      if (selectedSceneData !== null && a.scene_idx !== selectedSceneData.idx) continue;
      (kinds[a.kind] ||= []).push(a);
    }
    return kinds;
  }, [detail, selectedSceneData]);

  async function saveSetting(patch: Record<string, any>) {
    const data = await request(() => api.post('/api/settings', patch), 'Settings save failed');
    if (data) {
      setSettings((s) => (s ? { ...s, settings: (data as any).settings } : s));
      setProbe((p) => (p ? { ...p, plan: (data as any).plan } : p));
      toast('ok', 'Settings saved — engines take effect on the next run');
    }
  }

  const llmRoles = settings?.llm_roles;
  const roleMap = settings?.settings?.ollama?.roles || {};
  const engineSections = [
    { key: 'tts', label: 'TTS / Khmer voice', options: ['auto', 'sherpa', 'piper', 'kokoro', 'placeholder'] },
    { key: 'video', label: 'Video model', options: ['auto', 'comfyui', 'previz', 'defer', 'off'] },
    { key: 'sfx', label: 'SFX model', options: ['auto', 'mmaudio', 'procedural', 'defer', 'off'] },
    { key: 'rvc', label: 'Voice conversion', options: ['auto', 'http', 'cli', 'bypass'] },
  ];

  /* ---------------------------------------------------------------- render */
  return (
    <div className="studio-shell">
      <aside className="left-rail">
        <div className="brand">
          <span className="brand-mark">◧</span>
          <div><strong>Studio</strong><small>{status?.version || ''}</small></div>
        </div>
        <div className="rail-actions">
          <button className="btn primary" onClick={openNewProject}>+ New project</button>
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search projects…" />
        </div>
        <div className="project-list">
          {filteredProjects.length === 0 && <div className="empty">No projects yet. Create one.</div>}
          {filteredProjects.map((p) => (
            <button key={p.id} className={`project-row ${selected === p.id ? 'active' : ''}`}
                    onClick={() => { setSelected(p.id); setView('project'); }}>
              <div className="project-row-title"><span className={`dot ${statusClass(p.status)}`} />{p.title}</div>
              <div className="project-row-meta">{p.mode} · {p.content_type} · {p.status}</div>
            </button>
          ))}
        </div>
        <div className="services-panel">
          <div className="section-title">Services</div>
          {services.map((s) => (
            <button key={s.name} className="service-row" title={s.fix ? `Fix: ${s.fix}` : s.desc} onClick={() => window.open(s.url, '_blank')}>
              <span className={`dot ${statusClass(s.status === 'up' ? 'done' : s.status === 'down' ? 'failed' : 'pending')}`} />
              <span className="service-name">{s.name}</span>
              <span className="service-status">{s.status}</span>
            </button>
          ))}
          <button className="icon-btn" onClick={refreshServices}>probe services</button>
        </div>
        <nav className="rail-nav">
          <button className={view === 'project' ? 'active' : ''} onClick={() => setView('project')}>Project</button>
          <button className={view === 'team' ? 'active' : ''} onClick={() => setView('team')}>Team</button>
          <button className={view === 'plugins' ? 'active' : ''} onClick={() => setView('plugins')}>Plugins</button>
          <button className={view === 'services' ? 'active' : ''} onClick={() => setView('services')}>Services</button>
        </nav>
      </aside>

      <main className="center-pane">
        <header className="topbar">
          <div className="project-title"><strong>{detail?.project?.title || 'New project'}</strong><span className="tag">{detail?.project?.content_type || form.content_type}</span></div>
          <div className="run-controls">
            {activeRunId && <button className="btn" onClick={() => controlRun('pause')}>Pause</button>}
            {activeRunId && <button className="btn" onClick={() => controlRun('resume')}>Resume</button>}
            {activeRunId && <button className="btn danger" onClick={() => controlRun('cancel')}>Cancel</button>}
            {selected && <button className="btn primary" onClick={() => void startRun()}>Start run</button>}
          </div>
          <div className="health"><span className={`dot ${status?.ffmpeg ? 'ok' : 'muted'}`} />{status?.ffmpeg ? 'online' : 'offline'}</div>
        </header>

        {view === 'project' && (
          <div className="workspace">
            {!detail?.project?.id ? (
              <NewProject form={form} setForm={setForm} contentTypes={settings?.content_types || []} onCreate={createProject} loading={loading} />
            ) : (
              <ProjectWorkspace
                detail={detail}
                run={run}
                selectedStage={selectedStage}
                setSelectedStage={setSelectedStage}
                selectedScene={selectedScene}
                setSelectedScene={setSelectedScene}
                selectedSceneData={selectedSceneData}
                selectedAssetKinds={selectedAssetKinds}
                stageMap={stageMap}
                onStart={() => startRun()}
                onUpdate={updateProject}
                onRegenerate={regenerateStage}
              />
            )}
          </div>
        )}

        {view === 'team' && (
          <TeamView
            settings={settings}
            llmRoles={llmRoles}
            roleMap={roleMap}
            engineSections={engineSections}
            onSave={(p: any) => void saveSetting(p)}
            voices={[]}
          />
        )}

        {view === 'plugins' && <PluginsView settings={settings} workflows={workflows} engineSections={engineSections} onSave={(p: any) => void saveSetting(p)} />}

        {view === 'services' && <ServicesView services={services} probe={probe} onRefresh={refreshServices} settings={settings} />}
      </main>

      <aside className="right-rail">
        <Inspector
          selectedScene={selectedSceneData}
          selectedStage={selectedStage}
          stageRows={stageRows}
          run={run}
          assets={selectedAssetKinds}
          prompts={detail?.prompts?.filter((p) => selectedSceneData === null || p.scene_idx === selectedSceneData.idx) || []}
          integrity={detail?.integrity}
          onRegenerate={regenerateStage}
        />
      </aside>

      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>{t.text}</div>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------ New project */
function NewProject({ form, setForm, contentTypes, onCreate, loading }: any) {
  const [mode, ct, script, topic, target, notes] = [form.mode, form.content_type, form.script, form.topic_hint, form.target_duration, form.style_notes];
  return (
    <div className="card new-project">
      <h2>New project</h2>
      <div className="mode-row">
        <label className={`mode-card ${mode === 'A' ? 'selected' : ''}`}>
          <input type="radio" name="mode" checked={mode === 'A'} onChange={() => setForm({ ...form, mode: 'A' })} />
          <strong>Director</strong><small>Paste a finished script. No agent rewrites it.</small>
        </label>
        <label className={`mode-card ${mode === 'B' ? 'selected' : ''}`}>
          <input type="radio" name="mode" checked={mode === 'B'} onChange={() => setForm({ ...form, mode: 'B' })} />
          <strong>Auto</strong><small>AI writes a Khmer script from a topic hint.</small>
        </label>
      </div>

      <fieldset>
        <legend>Content type</legend>
        <div className="content-type-grid">
          {(contentTypes || []).map((c: any) => (
            <button key={c.key} className={`content-card ${ct === c.key ? 'selected' : ''}`} onClick={() => setForm({ ...form, content_type: c.key, target_duration: c.default_duration || 30 })}>
              <strong>{c.label}</strong>
              <small>{c.description}</small>
            </button>
          ))}
        </div>
      </fieldset>

      {mode === 'A' ? (
        <label className="field">Script <textarea value={script} onChange={(e) => setForm({ ...form, script: e.target.value })} rows={7} placeholder="Paste the finished Khmer script…" /></label>
      ) : (
        <label className="field">Topic hint <input value={topic} onChange={(e) => setForm({ ...form, topic_hint: e.target.value })} placeholder="e.g. ការមិនបោះបង់ចិត្ត" /></label>
      )}

      <div className="two-col">
        <label className="field">Target duration (s) <input type="number" min={5} max={300} value={target} onChange={(e) => setForm({ ...form, target_duration: e.target.value })} /></label>
        <label className="field">Style notes <input value={notes} onChange={(e) => setForm({ ...form, style_notes: e.target.value })} placeholder="Optional director notes" /></label>
      </div>

      <button className="btn primary big" disabled={loading} onClick={onCreate}>{loading ? 'Creating…' : 'Create project'}</button>
    </div>
  );
}

/* ------------------------------------------------------- Project workspace */
function ProjectWorkspace({ detail, run, selectedStage, setSelectedStage, selectedScene, setSelectedScene, selectedSceneData, selectedAssetKinds, stageMap, onStart, onUpdate, onRegenerate }: any) {
  const p = detail?.project || {};
  const scenes = detail?.scenes || [];
  const latestRun = detail?.runs?.[0];
  return (
    <div className="workspace-scroll">
      <div className="card">
        <div className="section-title">Settings</div>
        <div className="two-col">
          <label className="field">Title <input defaultValue={p.title} onBlur={(e: any) => onUpdate({ title: e.target.value }, 'Title saved')} /></label>
          <label className="field">Content type
            <select value={p.content_type || 'explainer'} onChange={(e: any) => onUpdate({ content_type: e.target.value }, 'Content type saved')}>
              {['explainer', 'what_if', 'compare', 'choose', 'word_nuance', 'myth_vs_fact', 'quick_tip'].map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
        </div>
        <div className="two-col">
          <label className="field">Target duration (s) <input type="number" defaultValue={p.target_duration} onBlur={(e: any) => onUpdate({ target_duration: Number(e.target.value) }, 'Duration saved')} /></label>
          <label className="field">Status <span>{p.status}</span></label>
        </div>
        <div className="integrity">{detail?.['integrity']?.detail || ''}</div>
      </div>

      {!latestRun && <div className="card"><button className="btn primary" onClick={onStart}>Run pipeline</button></div>}

      <div className="card">
        <div className="section-title">Pipeline <span className="muted">live from API/WebSocket</span></div>
        <PipelineGraph run={run} stages={run?.graph?.STAGES || run?.graph?.stages || []} stageMap={stageMap} selectedStage={selectedStage} setSelectedStage={setSelectedStage} />
      </div>

      <div className="card">
        <div className="section-title">Scene board <span className="muted">{scenes.length} scenes</span></div>
        <div className="scene-grid">
          {scenes.map((s: any) => (
            <button key={s.idx} className={`scene-card ${selectedScene === s.idx ? 'selected' : ''}`} onClick={() => setSelectedScene(s.idx)}>
              <div className="scene-side">{s.meta?.content_side || '—'}</div>
              <div className="scene-text">{s.text}</div>
              <div className="scene-meta">{s.mood_tag} · {s.estimated_duration_sec}s</div>
              <div className="scene-prompt">{s.visual_prompt}</div>
            </button>
          ))}
        </div>
        {scenes.length === 0 && <div className="empty">No scenes yet — run the pipeline to segment.</div>}
      </div>
      {/* asset browser */}
      <div className="card">
        <div className="section-title">Project assets</div>
        <div className="asset-list">
          {(Object.keys(selectedAssetKinds) as string[]).map((kind) => (
            <div key={kind} className="asset-item"><strong>{assetKindLabel(kind)}</strong> {selectedAssetKinds[kind].map((a: any) => <span key={a.id}>{a.scene_idx >= 0 ? `scene ${a.scene_idx} ` : ''}{a.mime} · {a.size_human || a.size_bytes}B</span>)}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- Pipeline DAG */
function PipelineGraph({ run, stages, stageMap, selectedStage, setSelectedStage }: any) {
  const list = stages?.length ? stages : STAGE_ORDER.map((key) => ({ key, title: labelFor(key) }));
  const statuses = stageMap || {};
  return (
    <div className="dag">
      {list.map((s: any, i: number) => {
        const key = s.key || s.stage;
        const rows: any[] = statuses[key] || [];
        const status = rows.length === 0 ? (run?.run?.status ? 'queued' : 'pending') : rows.every((r) => r.status === 'done') ? 'done' : rows.some((r) => r.status === 'failed') ? 'failed' : rows.some((r) => r.status === 'running') ? 'running' : 'pending';
        return (
          <div key={key} className={`dag-node ${statusClass(status)} ${selectedStage === key ? 'selected' : ''}`} onClick={() => setSelectedStage(key)}>
            <span className="dag-index">{String(i + 1).padStart(2, '0')}</span>
            <div><strong>{s.title || labelFor(key)}</strong><small>{rows.length ? `${rows.filter((r) => r.status === 'done').length}/${rows.length} · ${status}` : status}</small></div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------- Team view */
function TeamView({ settings, llmRoles, roleMap, engineSections, onSave, voices }: any) {
  const [teamPatch, setTeamPatch] = useState<Record<string, any>>({});
  function patch(p: Record<string, any>) {
    setTeamPatch((prev) => ({ ...prev, ...p }));
    onSave(p);
  }
  return (
    <div className="card">
      <h2>AI Team</h2>
      <p className="muted">Assign model/engine per pipeline role. Changes are saved immediately; no silent failures.</p>
      <div className="team-grid">
        {(llmRoles?.keys || []).map((role: string) => {
          const rc = roleMap[role] || { enabled: true, model: 'sailor2:8b', fallback_model: 'llama3.2:3b', temperature: 0.6 };
          return (
            <div className="team-card" key={role}>
              <strong>{llmRoles.labels?.[role] || role}</strong>
              <label className="check"><input type="checkbox" checked={!!rc.enabled} onChange={(e) => patch({ ollama: { roles: { [role]: { enabled: e.target.checked } } } })} /> enabled</label>
              <label className="field">Model <input defaultValue={rc.model} onBlur={(e: any) => patch({ ollama: { roles: { [role]: { model: e.target.value } } } })} /></label>
              <label className="field">Fallback <input defaultValue={rc.fallback_model} onBlur={(e: any) => patch({ ollama: { roles: { [role]: { fallback_model: e.target.value } } } })} /></label>
              <div className="field">current: <code>{rc.enabled ? rc.model : 'off'}</code></div>
            </div>
          );
        })}
      </div>
      <div className="section-title">Engines (backend stage choices)</div>
      <div className="team-grid">
        {engineSections.map((sec: any) => (
          <div className="team-card" key={sec.key}>
            <strong>{sec.label}</strong>
            <label className="field">Engine <select defaultValue={settings?.settings?.[sec.key]?.engine || 'auto'} onChange={(e) => patch({ [sec.key]: { engine: e.target.value } })}>{sec.options.map((o: string) => <option key={o} value={o}>{o}</option>)}</select></label>
            <div className="field">current: <code>{settings?.settings?.[sec.key]?.engine || 'auto'}</code></div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* --------------------------------------------------------- Plugins view */
function PluginsView({ settings, workflows, engineSections, onSave }: any) {
  const [patch, setPatch] = useState<Record<string, any>>({});
  function save(p: Record<string, any>) { setPatch((x) => ({ ...x, ...p })); onSave(p); }
  return (
    <div className="card">
      <h2>Engines / Plugins</h2>
      <p className="muted">Swappable backends read from backend settings and the workflow folder.</p>
      <div className="team-grid">
        {engineSections.map((sec: any) => (
          <div className="team-card" key={sec.key}>
            <strong>{sec.label}</strong>
            <select defaultValue={settings?.settings?.[sec.key]?.engine || 'auto'} onChange={(e: any) => save({ [sec.key]: { engine: e.target.value } })}>{sec.options.map((o: string) => <option key={o} value={o}>{o}</option>)}</select>
            <div className="muted">current: {settings?.settings?.[sec.key]?.engine || 'auto'}</div>
          </div>
        ))}
      </div>
      <div className="section-title">ComfyUI workflows on disk</div>
      <table className="table">
        <thead><tr><th>Name</th><th>Dir</th><th>Size</th><th>Placeholders</th></tr></thead>
        <tbody>
          {(workflows || []).map((w: any) => (
            <tr key={w.path}><td>{w.name}</td><td>{w.dir}</td><td>{w.size}</td><td>{String((w.placeholders || []).length)}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* -------------------------------------------------------- Services view */
function ServicesView({ services, probe, onRefresh, settings }: any) {
  return (
    <div className="card">
      <h2>Services / URLs</h2>
      <p className="muted">Values are pulled from backend settings. Click a row to open the service where it has a UI.</p>
      <button className="btn primary" onClick={onRefresh}>Probe services</button>
      <table className="table">
        <thead><tr><th>Name</th><th>URL</th><th>Status</th><th>Job</th><th>Fix command</th></tr></thead>
        <tbody>
          {services.map((s: any) => (
            <tr key={s.name}>
              <td><strong>{s.name}</strong></td>
              <td><a href={s.url} target="_blank" rel="noreferrer">{s.url}</a></td>
              <td><span className={`dot ${statusClass(s.status === 'up' ? 'done' : s.status === 'down' ? 'failed' : 'pending')}`} />{s.status}</td>
              <td>{s.desc}</td>
              <td className="fix">{s.fix}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="muted">data dir: {settings?.settings?.paths?.data_dir || 'default'}</div>
    </div>
  );
}

/* -------------------------------------------------------- Right inspector */
function Inspector({ selectedScene, selectedStage, stageRows, run, assets, prompts, integrity, onRegenerate }: any) {
  const stageRowsFor = selectedStage ? stageRows.filter((r: any) => r.stage === selectedStage) : stageRows;
  return (
    <div className="inspector-scroll">
      <div className="section-title">Inspector</div>
      {selectedScene ? (
        <div className="card compact">
          <div className="section-title">Scene {selectedScene.idx} <span className="tag">{selectedScene.meta?.content_side || ''}</span></div>
          <p>{selectedScene.text}</p>
          <label className="field">Visual prompt <textarea defaultValue={selectedScene.visual_prompt} rows={4} /></label>
          <label className="field">Mood <input defaultValue={selectedScene.mood_tag} /></label>
          <div className="row-actions">
            <button className="btn" onClick={() => onRegenerate('voice_base', selectedScene.idx)}>Regenerate voice</button>
            <button className="btn" onClick={() => onRegenerate('video', selectedScene.idx)}>Regenerate video</button>
          </div>
        </div>
      ) : !selectedStage ? (
        <div className="empty muted">Select a scene or a pipeline stage.</div>
      ) : null}
      {selectedStage && (
        <div className="card compact">
          <div className="section-title">{labelFor(selectedStage)} <span className="tag">{selectedStage}</span></div>
          {stageRowsFor.length === 0 && <div className="muted">No stage rows yet.</div>}
          {stageRowsFor.map((r: any) => (
            <div key={r.id} className="stage-row">
              <span className={`dot ${statusClass(r.status)}`} />
              <strong>scene {r.scene_idx}</strong>
              <span>{r.status}</span>
              <span className="muted">{r.message}</span>
              {r.error && <div className="error-text">{r.error}</div>}
            </div>
          ))}
          <button className="btn danger" onClick={() => onRegenerate(selectedStage, selectedScene?.idx ?? -1)}>Regenerate {selectedStage}</button>
        </div>
      )}
      {Object.keys(assets || {}).length > 0 && (
        <div className="card compact">
          <div className="section-title">Assets</div>
          {Object.keys(assets).map((kind) => (
            <div key={kind} className="asset-inspector">
              <strong>{assetKindLabel(kind)}</strong>
              {assets[kind].map((a: any) => (
                <div key={a.id} className="asset-row">
                  {a.mime?.startsWith('audio/') && <audio controls src={a.url || `/api/assets/${a.id}/stream`} />}
                  {a.mime?.startsWith('video/') && <video controls src={a.url || `/api/assets/${a.id}/stream`} />}
                  {a.mime?.includes('json') && <pre>{JSON.stringify(a.meta || a, null, 2)}</pre>}
                  <a className="muted" href={a.download || `/api/assets/${a.id}/download`} download>download</a>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
      {(prompts || []).length > 0 && (
        <div className="card compact">
          <div className="section-title">Prompts</div>
          {prompts.slice(0, 12).map((p: any) => (
            <details key={p.id}><summary>{p.stage} · {p.engine || p.model}</summary><pre>{p.system}{'\n'}{p.user}{'\n'}→ {p.response}</pre></details>
          ))}
        </div>
      )}
      {integrity?.applies && <div className="card compact"><div className="section-title">Integrity</div><div className={integrity.ok ? 'ok-text' : 'err-text'}>{integrity.detail}</div></div>}
    </div>
  );
}
