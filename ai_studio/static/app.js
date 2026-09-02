/* Khmer AI Content Studio — console UI.
   Vanilla JS, no build step, no CDN: everything talks to the local FastAPI
   server under /api. Live progress uses WebSocket, then SSE, then polling. */
'use strict';

/* ------------------------------------------------------------------ helpers */
const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
const esc = (s) => String(s === null || s === undefined ? '' : s)
  .replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const isKhmer = (s) => /ក-៿/.test(s || '');
const n = (v, d) => (Number.isFinite(+v) ? +v : (d === undefined ? 0 : d));
const num = (v, d) => { const x = n(v, null); return x === null ? (d === undefined ? '—' : d) : (Math.round(x * 100) / 100); };
const dur = (s) => { s = n(s); if (!s) return '—'; return s < 60 ? s.toFixed(1) + 's' : Math.floor(s / 60) + 'm' + String(Math.round(s % 60)).padStart(2, '0'); };
const bytes = (b) => { b = n(b); const u = ['B', 'KB', 'MB', 'GB']; let i = 0; while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; } return (i ? b.toFixed(1) : b) + ' ' + u[i]; };
const ago = (ts) => {
  ts = n(ts); if (!ts) return '—';
  const d = Math.max(0, (Date.now() / 1000) - ts), u = [[86400, 'd'], [3600, 'h'], [60, 'm']];
  for (const [s, l] of u) if (d >= s) return Math.floor(d / s) + l + ' ago';
  return 'just now';
};
const clock = (ts) => { ts = n(ts); return ts ? new Date(ts * 1000).toLocaleString() : '—'; };
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const uniq = (a) => Array.from(new Set(a));
const safe = (v) => (v === undefined || v === null ? '?' : v);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, opts) {
  opts = opts || {};
  const init = { method: opts.method || 'GET', headers: {} };
  if (opts.json !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(opts.json); }
  if (opts.form !== undefined) { init.body = opts.form; }
  const r = await fetch('/api' + path, init);
  const ct = r.headers.get('content-type') || '';
  let data = null;
  if (ct.includes('json')) { try { data = await r.json(); } catch (e) { data = null; } }
  else { data = await r.text(); }
  if (!r.ok) {
    const msg = (data && (data.detail || data.error || data.message)) || (typeof data === 'string' ? data.slice(0, 300) : '') || ('HTTP ' + r.status);
    const err = new Error(msg); err.status = r.status; err.data = data; throw err;
  }
  return data;
}

function toast(msg, kind, ms) {
  const t = el(`<div class="toast ${kind || ''}">${esc(msg)}</div>`);
  t.onclick = () => t.remove();
  $('#toasts').appendChild(t);
  setTimeout(() => t.remove(), ms || (kind === 'err' ? 9000 : 4200));
  if (kind === 'err') console.warn('[studio]', msg);
}

function modal(title, bodyNode, opts) {
  opts = opts || {};
  const box = $('#modal-box');
  box.innerHTML = '';
  const head = el(`<div class="spread"><h3>${esc(title)}</h3><button class="close-x" title="close">×</button></div>`);
  head.querySelector('.close-x').onclick = closeModal;
  box.appendChild(head);
  box.appendChild(bodyNode);
  $('#modal').hidden = false;
  $('#modal').onclick = (e) => { if (e.target === $('#modal')) closeModal(); };
  if (opts.onOpen) opts.onOpen(box);
  return box;
}
function closeModal() { $('#modal').hidden = true; $('#modal-box').innerHTML = ''; }

const STAGE_ORDER = ['script', 'breakdown', 'voice_base', 'voice_final', 'video', 'video_fit', 'sfx', 'qa', 'assemble'];
const KIND_FOR_STAGE = {
  script: 'script', breakdown: 'scenes', voice_base: 'voice', voice_final: 'voice_final',
  video: 'video', video_fit: 'video_fit', sfx: 'ambient', qa: 'qa', assemble: 'final',
};
const STAGE_ICON = {
  script: '', breakdown: '🎞️', voice_base: '🗣️', voice_final: '🎙️', video: '️',
  video_fit: '⏱️', sfx: '🌿', qa: '🔍', assemble: '🎁',
};

/* ------------------------------------------------------------------- state */
const S = {
  view: 'projects',
  status: null, settings: null, plan: null, roles: [],
  projects: [], counts: {}, filters: { search: '', status: '', mode: '', sort: 'updated_desc' },
  project: null, projectId: null, scenes: [], integrity: null, disk: null, prompts: [], assets: [],
  runs: [], run: null, runId: null, stages: [], byStage: {}, overall: { pct: 0 }, events: [], lastEventId: 0,
  selScene: 0, selStage: 'voice_final', expanded: {}, bundle: null,
  mode: 'A', draft: {}, board: null, savingBoard: false,
  live: null, liveMode: 'off', pollTimer: null, log: [],
  voices: [], discovered: [], mem: null, memQ: '', wf: null, probe: null,
  style: null, seq: 0,
};

/* ------------------------------------------------------------------- boot */
(async function boot() {
  window.addEventListener('hashchange', route);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
  $$('#topbar .tab').forEach((b) => { b.onclick = () => go(b.dataset.nav); });
  $('.brand').onclick = () => go('projects');
  try {
    await refreshStatus();
    await loadSettings();
    loadVoicesAndStyle();
  } catch (e) { toast('studio API unreachable: ' + e.message, 'err'); }
  $('#boot').remove();
  $('#app').hidden = false;
  route();
  setInterval(() => { if (S.view === 'projects') refreshStatus().then(() => renderTop()).catch(() => {}); }, 15000);
})();

async function refreshStatus() {
  S.status = await api('/status');
  S.plan = (S.status && S.status.plan) || null;
}

async function loadVoicesAndStyle() {
  try { const st2 = await api('/style'); S.style = st2; S.moods = st2.moods || []; } catch (e) {}
  try { const v = await api('/voices'); S.voices = v.voices || []; } catch (e) {}
}

async function loadSettings() {
  const s = await api('/settings');
  S.settings = s.settings; S.plan = s.plan; S.roles = s.roles || [];
  S.probe = s.capabilities || S.probe;
  renderTop();
}

/* ------------------------------------------------------------------ routing */
function go(view, arg) { location.hash = '#/' + view + (arg ? '/' + arg : ''); }

function route() {
  const parts = (location.hash || '#/projects').replace(/^#\/?/, '').split('/');
  const view = parts[0] || 'projects';
  stopLive();
  S.view = view;
  $$('#topbar .tab').forEach((b) => b.classList.toggle('on', b.dataset.nav === view));
  const v = $('#view');
  v.innerHTML = '<div class="empty">…</div>';
  if (view === 'projects') renderProjects(v);
  else if (view === 'new') renderNew(v);
  else if (view === 'project') openProject(parts[1], v);
  else if (view === 'voices') renderVoices(v);
  else if (view === 'memory') renderMemory(v);
  else if (view === 'settings') renderSettings(v);
  else { v.innerHTML = '<div class="empty">unknown view</div>'; }
}

/* ---------------------------------------------------------------- top bar */
const renderTop = debounce(() => {
  const st = S.status || {}, p = S.plan || {}, s = S.settings || {};
  const mach = (st.machine || {}).profile || ((s.machine || {}).profile) || '?';
  const eng = (k) => ((p[k] || {}).engine) || '—';
  $('#machine-pill').innerHTML = `<b>${esc(mach)}</b> · VRAM ${n((st.machine || {}).vram_total_mb) ? n((st.machine || {}).vram_total_mb) + 'MB' : 'none'}`;
  $('#plan-pill').innerHTML = `🗣️ ${esc(eng('tts'))} · 🎙️ ${esc(eng('rvc'))} · 🖼️ ${esc(eng('video'))} · 🌿 ${esc(eng('sfx'))}`;
  const counts = S.counts || {};
  const cbtn = $('[data-count=projects]'); if (cbtn) cbtn.textContent = counts.total ? '(' + counts.total + ')' : '';
  const banner = $('#plan-banner');
  const notes = [];
  if (!(p.ollama || {}).available) notes.push('Ollama is offline — the Controller falls back to deterministic segmentation, so wording stays safe but scene ideas will be plainer. Start Ollama (<code>ollama serve</code>) then press <b>Probe</b> in Settings.');
  ['tts', 'rvc', 'video', 'sfx'].forEach((k) => {
    const e = (p[k] || {}).engine;
    if (e === 'placeholder') notes.push('Voice 3a is using the <b>placeholder</b> engine (silent-shaped audio). Install <code>sherpa-onnx</code> + <code>vits-mms-khm</code> for real Khmer speech.');
    if (e === 'defer') notes.push(`<b>${esc(k)}</b> is deferred to the GPU machine — the draft cut continues without it, and <b>GPU catch-up</b> fills it in later.`);
    if (e === 'bypass') notes.push('RVC timbre is bypassed — stage 3b just passes the 3a voice through. Add a voice profile under <b>Voices</b>.');
  });
  if ((p.hardware || {}).cpu_only) notes.push('Machine B (no CUDA): video/SFX are CPU previz + procedural ambience, or deferred.');
  if (notes.length) { banner.hidden = false; banner.innerHTML = notes.map((x) => '· ' + x).join('<br>'); banner.onclick = () => { banner.hidden = true; }; }
  else banner.hidden = true;
}, 120);

/* -------------------------------------------------------------- projects */
async function renderProjects(root) {
  root.innerHTML = `<div class="spread"><h2 style="margin:0">Projects <span class="mut">គម្រោង</span></h2>
    <div class="btn-group"><button class="btn primary" id="p-new">＋ New project</button>
    <button class="btn" id="p-reload">↻ Refresh</button></div></div>
    <div id="p-stats" class="row" style="margin:14px 0"></div>
    <div class="filters">
      <input type="search" id="f-search" placeholder="search title / script / topic…">
      <select id="f-status"><option value="">any status</option></select>
      <select id="f-mode"><option value="">A + B</option><option value="A">A · Director</option><option value="B">B · Auto</option></select>
      <select id="f-sort">
        <option value="updated_desc">recently updated</option><option value="created_desc">newest</option>
        <option value="title_asc">title A→Z</option><option value="title_desc">title Z→A</option>
        <option value="status_asc">status</option></select>
      <span class="hint" id="f-hint"></span>
    </div>
    <div class="plist" id="p-list"></div>`;
  $('#p-new').onclick = () => go('new');
  $('#p-reload').onclick = () => { loadProjects().then(() => toast('reloaded', 'ok', 1500)); };
  const st = (S.status || {}).db || {};
  $('#p-stats').innerHTML = [['projects', st.projects], ['runs', st.runs], ['completed', st.completed_runs],
    ['assets', st.assets], ['prompts', st.prompts], ['disk', bytes(st.disk_bytes)]]
    .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v === undefined ? '—' : v}</div></div>`)
    .join('');
  const sel = $('#f-status');
  uniq(Object.keys(S.counts || {})).forEach((k) => sel.appendChild(el(`<option value="${esc(k)}">${esc(k)} (${S.counts[k]})</option>`)));
  $('#f-search').value = S.filters.search; sel.value = S.filters.status;
  $('#f-mode').value = S.filters.mode; $('#f-sort').value = S.filters.sort;
  const ref = debounce(() => loadProjects(), 260);
  $('#f-search').oninput = (e) => { S.filters.search = e.target.value; ref(); };
  ['#f-status', '#f-mode', '#f-sort'].forEach((id) => {
    $(id).onchange = (e) => { S.filters[id.slice(3)] = e.target.value; loadProjects(); };
  });
  await loadProjects();
}

async function loadProjects() {
  const f = S.filters;
  const q = `?search=${encodeURIComponent(f.search)}&status=${f.status}&mode=${f.mode}&sort=${f.sort}`;
  const d = await api('/projects' + q);
  S.projects = d.projects || []; S.counts = d.counts || {};
  const list = $('#p-list'); if (!list) return;
  $('#f-hint').textContent = S.projects.length + ' shown · ' + Object.values(S.counts).reduce((a, b) => a + b, 0) + ' total';
  if (!S.projects.length) {
    list.innerHTML = `<div class="empty"><div class="big">🎬</div>nothing here yet.<br>
      <button class="btn primary" onclick="go('new')" style="margin-top:10px">Create the first project</button></div>`;
    return;
  }
  list.innerHTML = '';
  for (const p of S.projects) {
    const finalAsset = await finalOf(p.id);
    const row = el(`<div class="prow" data-open="${esc(p.id)}">
      <div class="thumb">${finalAsset ? '<img loading="lazy" src="/api/assets/' + finalAsset.thumb_id + '/stream">' : (p.mode === 'A' ? '📝' : '💡')}</div>
      <div>
        <div class="ttl">${esc(p.title || '(untitled)')}</div>
        <div class="ex km ${isKhmer(p.script_excerpt) ? 'km' : ''}">${esc((p.script_excerpt || '').replace(/\s+/g, ' ').slice(0, 150))}</div>
        <div class="meta" style="margin-top:5px">
          <span class="badge ${p.mode}">Mode ${p.mode}</span>
          <span class="badge ${esc(p.status)}">${esc(p.status)}</span>
          <span class="badge">${p.scene_count || 0} scenes</span>
          <span class="badge">${dur(targetOf(p))}</span>
          ${p.script_origin ? `<span class="badge">${esc(p.script_origin)}</span>` : ''}
          ${p.last_run_status ? `<span class="badge ${esc(p.last_run_status)}">run: ${esc(p.last_run_status)}</span>` : ''}
        </div>
      </div>
      <div class="dim tiny">${esc(clock(p.updated_at))}<br><span class="mut">${ago(p.updated_at)}</span></div>
      <div class="right btn-group" data-stop>
        ${finalAsset ? `<a class="btn small" href="${finalAsset.download}" download>⬇ mp4</a>` : ''}
        <button class="btn small" data-dup="${esc(p.id)}" title="duplicate project">⧉</button>
        <button class="btn small danger" data-del="${esc(p.id)}" title="delete">🗑</button>
      </div></div>`);
    row.onclick = (e) => { if (e.target.closest('[data-stop]')) return; go('project', p.id); };
    row.querySelector('[data-dup]').onclick = async (e) => {
      e.stopPropagation();
      try { const r = await api(`/projects/${p.id}/duplicate`, { method: 'POST', json: {} }); toast(r.note || 'duplicated', 'ok'); loadProjects(); }
      catch (err) { toast(err.message, 'err'); }
    };
    row.querySelector('[data-del]').onclick = async (e) => {
      e.stopPropagation();
      if (!confirm('Delete project "' + (p.title || p.id) + '" and its render history? Files on disk are kept unless you tick the box.')) return;
      try { await api(`/projects/${p.id}?purge_files=true`, { method: 'DELETE' }); toast('deleted', 'ok'); loadProjects(); }
      catch (err) { toast(err.message, 'err'); }
    };
    list.appendChild(row);
  }
}

const targetOf = (p) => Math.max(n(p.script_chars) * 0.11, n(p.target_duration) || 0);
const finalCache = {};
async function finalOf(pid) {
  if (finalCache[pid] !== undefined) return finalCache[pid];
  let out = null;
  try {
    const d = await api(`/assets?project_id=${pid}&kind=final&limit=1`);
    const a = (d.assets || [])[0];
    if (a) {
      let thumb = null;
      const t = await api(`/assets?project_id=${pid}&kind=poster&limit=1`);
      thumb = (t.assets || [])[0];
      out = { id: a.id, thumb_id: thumb ? thumb.id : a.id, download: `/api/assets/${a.id}/download` };
    }
  } catch (e) { out = null; }
  finalCache[pid] = out;
  return out;
}

/* ------------------------------------------------------------ new project */
function renderNew(root) {
  S.draft = S.draft || {};
  const d = S.draft;
  root.innerHTML = `<h2 style="margin-top:0">New project</h2>
  <div class="card">
    <div class="section-title" style="margin-top:0">1 · pick how much control you want</div>
    <div class="modes">
      <div class="mode ${S.mode === 'A' ? 'on' : ''}" data-mode="A">
        <div class="m-emoji">📝</div>
        <div class="m-title">Mode A — Director script</div>
        <div class="m-desc km">អ្នកខ្លួនឯងជាអ្នកគ្រប់គ្រង។ អត្ថបទរបស់អ្នកគជាច្បាប់។</div>
        <div class="m-desc">You paste the finished Khmer script. It is stored as ground truth:
        the studio only <b>segments</b> it into scenes — no agent may rewrite, paraphrase or
        "improve" a single word. Every prompt still gets logged so you can re-use the look later.</div>
        <ul><li>best when you already wrote the script</li><li>integrity check: scene texts must rejoin your paste</li></ul>
      </div>
      <div class="mode ${S.mode === 'B' ? 'on' : ''}" data-mode="B">
        <div class="m-emoji">💡</div>
        <div class="m-title">Mode B — Auto idea</div>
        <div class="m-desc km">ប្រាប់តែប្រធានបទ ទុកការសរសេរឱ្យ AI។</div>
        <div class="m-desc">Give a topic hint; the Controller writes the whole Khmer script under the
        fixed house style (calm, warm, "don't give up"). You then review / edit / regenerate before
        production starts — or skip the gate for full autonomy.</div>
        <ul><li>one-line topic is enough</li><li>approve, edit, or regenerate the script</li></ul>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="section-title" style="margin-top:0">2 · details</div>
    <div class="grid two">
      <label class="f"><span>title (optional)</span><input type="text" id="n-title" placeholder="e.g. កុំបោះបង់ · Don't give up" value="${esc(d.title || '')}"></label>
      <label class="f"><span>target length (seconds)</span><input type="number" id="n-dur" min="8" max="600" step="1" value="${n(d.target_duration) || 30}"></label>
    </div>
    <div id="n-a">
      <label class="f"><span>your finished script — one sentence per line, Khmer</span>
        <textarea class="script km" id="n-script" placeholder="ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។
វាគឺជាដំណើរ ដែលយើងត្រូវរៀនដើរម្ដងមួយជំហាន។">${esc(d.script || '')}</textarea></label>
      <div class="hint" id="n-count"></div>
    </div>
    <div id="n-b" class="hidden">
      <label class="f"><span>topic hint</span><input type="text" id="n-topic" placeholder="e.g. for students who failed an exam / អ្នកដែលបរាជ័យព្រឹត្តិទ្នីមួយ" value="${esc(d.topic_hint || '')}"></label>
      <label class="f"><span>extra style notes for this project (optional)</span>
        <textarea id="n-notes" placeholder="e.g. address it to young farmers, mention morning rain">${esc(d.style_notes || '')}</textarea></label>
      <div class="check"><input type="checkbox" id="n-auto" ${d.auto_approve ? 'checked' : ''}>
        <span>Skip the review gate and render immediately (full autonomy).</span></div>
    </div>
    <div class="grid two">
      <label class="f"><span>voice profile (RVC timbre, optional)</span><select id="n-voice"></select></label>
      <label class="f"><span>engine overrides for this project (leave empty = use Settings)</span>
        <div class="row">
          <select id="n-tts"><option value="">voice 3a: settings</option><option value="sherpa">sherpa-onnx</option><option value="piper">piper</option><option value="kokoro">kokoro</option><option value="placeholder">placeholder</option></select>
          <select id="n-video"><option value="">video: settings</option><option value="comfyui">ComfyUI + Wan</option><option value="previz">previz (CPU)</option><option value="defer">defer to GPU</option><option value="off">off</option></select>
          <select id="n-sfx"><option value="">sfx: settings</option><option value="mmaudio">MMAudio</option><option value="procedural">procedural</option><option value="defer">defer to GPU</option><option value="off">off</option></select>
        </div></label>
    </div>
    <div class="spread" style="margin-top:8px">
      <div class="hint">house style is always applied — <a href="#" id="n-style">see the guideline</a></div>
      <div class="btn-group">
        <button class="btn" id="n-sample">fill with sample script</button>
        <button class="btn primary" id="n-create">Create project ${S.mode === 'B' ? '& generate idea' : ''}</button>
      </div>
    </div>
  </div>
  <div class="card" id="n-fix">
    <h3>What the pipeline will do</h3>
    <div id="n-plan" class="grid two"></div>
  </div>`;

  $$('.mode').forEach((m) => { m.onclick = () => { S.mode = m.dataset.mode; renderNew(root); }; });
  const v = $('#n-voice');
  v.innerHTML = '<option value="">— none, use the base Khmer voice —</option>' +
    (S.voices || []).map((x) => `<option value="${esc(x.id)}">${esc(x.name)}</option>`).join('');
  $('#n-style').onclick = (e) => { e.preventDefault(); showStyle(); };
  $('#n-sample').onclick = async () => {
    if (!S.style) S.style = await api('/style');
    $('#n-script').value = SAMPLE;
    countScript();
  };
  $('#n-script').oninput = countScript;
  countScript();
  renderPlanTiles($('#n-plan'));
  $('#n-create').onclick = createProject;
}

const SAMPLE = `ជីវិតមនុស្ស មិនមែនជាប្រណាំងទេ។
វាគឺជាដំណើរ ដែលយើងត្រូវរៀនដើរម្ដងមួយជំហាន។
នៅថ្ងៃដែលអ្នកពិបាក កុំបោះបង់ខ្លួនឯង។
ផ្កាមិនបើកព្រមគ្នាទេ ប៉ុន្តែវាបើកក្នុងរដូវរបស់វា។
បើចង់បានថ្ងៃថ្មី សូមអត់ទោសឱ្យខ្លួនឯងចំពោះកំហុសចាស់។
ដកដង្ហើមវែងៗ រួចចាប់ផ្ដើមឡើងវិញដោយស្ងប់ស្ងាត់។
អ្នកកំពុងធ្វើបានល្អជាងអ្វីដែលអ្នកគិត។`;

function countScript() {
  const t = ($('#n-script') || {}).value || '';
  const lines = t.split(/\n+/).map((x) => x.trim()).filter(Boolean);
  const km = lines.filter(isKhmer).length;
  const c = $('#n-count'); if (!c) return;
  c.innerHTML = `${lines.length} line(s) · ${t.length} chars · <b>~${dur(estimateSec(t))}</b> spoken`
    + (lines.length && km < lines.length ? ` · ⚠ ${lines.length - km} line(s) are not Khmer` : '');
}
const estimateSec = (t) => {
  const km = (t.match(/ក-៿/g) || []).length;
  return km / 18 + (t.match(/\s/g) || []).length * 0.02 + (t.match(/។/g) || []).length * 0.25;
};

function renderPlanTiles(host) {
  if (!host) return;
  const p = S.plan || {};
  const roles = {}; (S.roles || []).forEach((r) => { roles[r.key] = r; });
  const rows = [
    ['1 · Scene breakdown', ((p.ollama || {}).available ? 'Ollama ' + (((S.settings || {}).controller || {}).model || 'sailor2:8b') : 'deterministic fallback'), 'splits your script into scenes + visual prompts'],
    ['2 · Auto idea', ((p.ollama || {}).available ? 'Ollama (auto_idea model)' : 'template fallback'), 'only runs in Mode B'],
    ['3a · Khmer voice', (p.tts || {}).engine || '?', 'sherpa-onnx vits-mms-khm when installed'],
    ['3b · Your timbre', (p.rvc || {}).engine || '?', 'RVC with your trained voice'],
    ['4 · Video', (p.video || {}).engine || '?', 'Wan via ComfyUI, or CPU previz draft'],
    ['5 · SFX director', (p.sfx || {}).engine || '?', 'MMAudio ambience from the mood tag'],
    ['6 · QA reviewer', ((p.ollama || {}).available ? 'Ollama (qa model)' : 'deterministic facts'), 'per-scene pass/fail + issues'],
    ['7 · Final assembly', 'ffmpeg', 'mux + ducking + subtitles + thumbnail'],
  ];
  host.innerHTML = rows.map(([k, v, s]) =>
    `<div class="tile"><div class="spread"><b>${esc(k)}</b><span class="chip ${v && !/defer|off|fallback|template|placeholder|\?/.test(String(v)) ? 'on' : 'off'}">${esc(v)}</span></div>
     <div class="hint">${esc(s)}</div></div>`).join('');
}

async function createProject() {
  const body = {
    mode: S.mode,
    title: ($('#n-title').value || '').trim(),
    target_duration: +$('#n-dur').value || 30,
    style_notes: ($('#n-notes') || {}).value || '',
    voice_profile_id: ($('#n-voice') || {}).value || '',
    settings: {},
  };
  if (S.mode === 'A') {
    body.script = $('#n-script').value || '';
    if (!body.script.trim()) { toast('Mode A needs your finished script pasted in', 'warn'); return; }
  } else {
    body.topic_hint = ($('#n-topic') || {}).value || '';
    body.generate_now = true;
    const notes = ($('#n-notes') || {}).value || '';
    if ($('#n-auto') && $('#n-auto').checked) { body.settings.pipeline = { auto_approve_mode_b: true, review_gate: 'never' }; }
  }
  const pick = (id, sec, key) => { const v = ($(id) || {}).value; if (v) { body.settings[sec] = Object.assign(body.settings[sec] || {}, { [key]: v }); } };
  pick('#n-tts', 'tts', 'engine'); pick('#n-video', 'video', 'engine'); pick('#n-sfx', 'sfx', 'engine');
  const btn = $('#n-create'); btn.disabled = true; btn.textContent = 'creating…';
  try {
    const r = await api('/projects', { method: 'POST', json: body });
    S.draft = {}; delete finalCache[r.project.id];
    toast('project created', 'ok', 1800);
    go('project', r.project.id);
  } catch (e) { toast(e.message, 'err'); btn.disabled = false; btn.textContent = 'Create project'; }
}

function showStyle() {
  const run = async () => {
    if (!S.style) S.style = await api('/style');
    const box = el(`<div><pre class="prompt-box" style="max-height:60vh">${esc(S.style.guideline)}</pre>
      <div class="section-title">mood → ambience map used by the SFX director</div>
      <div class="row">${Object.entries(S.style.ambience_examples || {}).map(([k, v]) =>
      `<span class="chip">${esc(k)} → ${esc(Array.isArray(v) ? v.join(', ') : v)}</span>`).join('')}</div></div>`);
    modal('Fixed house style guideline', box);
  };
  run().catch((e) => toast(e.message, 'err'));
}

/* ================================================================== project */
async function openProject(id, root) {
  if (!id) { go('projects'); return; }
  S.projectId = id; S.selScene = 0; S.selStage = 'voice_final'; S.bundle = null;
  root.innerHTML = '<div class="empty">loading project…</div>';
  try { await refreshProject(); } catch (e) { root.innerHTML = `<div class="empty">⚠ ${esc(e.message)}</div>`; return; }
  renderProject(root);
}

async function refreshProject() {
  const d = await api(`/projects/${S.projectId}`);
  S.project = d.project; S.scenes = d.scenes || []; S.runs = d.runs || [];
  S.integrity = d.integrity; S.disk = d.disk; S.prompts = d.prompts || []; S.assets = d.assets || [];
  S.runId = S.runId && S.runs.some((r) => r.id === S.runId) ? S.runId : (d.latest_run_id || null);
  if (S.runId) await refreshRun().catch(() => {});
}

async function refreshRun(since) {
  const d = await api(`/runs/${S.runId}/status?since=${since || S.lastEventId || 0}`);
  S.run = d;                       // the whole snapshot: {run, stages, by_stage, overall, active, ...}
  S.stages = d.stages || []; S.byStage = d.by_stage || {}; S.overall = d.overall || { pct: 0 };
  S.graph = d.graph || []; S.log = d.log || []; S.final = d.final; S.runPlan = d.plan;
  if (d.events && d.events.length) {
    S.lastEventId = d.last_event_id || S.lastEventId;
    for (const ev of d.events) S.events.push(ev);
    if (S.events.length > 400) S.events = S.events.slice(-400);
  }
  if (d.active === false && S.live) stopLive();
  return d;
}

function renderProject(root) {
  const p = S.project || {}, mode = (p.mode || 'A').toUpperCase();
  root.innerHTML = `
  <div class="card">
    <div class="spread">
      <div>
        <div class="spread" style="justify-content:flex-start">
          <h2 style="margin:0">${esc(p.title || '(untitled)')}</h2>
          <span class="badge ${mode}">Mode ${mode}</span>
          <span class="badge ${esc(p.status)}">${esc(p.status)}</span>
          ${p.script_origin ? `<span class="badge" title="who wrote the script">${esc(p.script_origin)}</span>` : ''}
        </div>
        <div class="dim tiny" style="margin-top:3px">${esc(p.id)} · ${S.scenes.length} scene(s) · ${dur(sumAudio())} spoken ·
          ${S.disk ? bytes(S.disk.bytes) + ' on disk' : '—'} · updated ${ago(p.updated_at)}</div>
      </div>
      <div class="btn-group">
        ${mode === 'B' ? '<button class="btn" id="btn-idea">💡 Generate / rewrite idea</button>' : ''}
        ${mode === 'B' && p.status !== 'done' ? '<button class="btn" id="btn-approve">✅ Approve script</button>' : ''}
        ${!S.runId ? '<button class="btn primary" id="btn-run">▶ Start production</button>' : ''}
        ${S.run && S.run.active && !S.run.paused ? '<button class="btn warn" id="btn-pause">⏸ Pause</button>' : ''}
        ${S.run && S.run.paused ? '<button class="btn primary" id="btn-resume">▶ Resume</button>' : ''}
        ${S.run && S.run.active ? '<button class="btn danger" id="btn-cancel">■ Cancel</button>' : ''}
        ${S.run && !S.run.active && S.run.run && S.run.run.status !== 'completed' ? '<button class="btn" id="btn-continue">↻ Resume from last good stage</button>' : ''}
        ${deferredCount() ? `<button class="btn" id="btn-catchup" title="re-render video + SFX + QA + assembly on the GPU machine">🖥️ GPU catch-up (${deferredCount()})</button>` : ''}
        <button class="btn" id="btn-refresh">↻</button>
      </div>
    </div>
    ${integrityLine()}
  </div>
  <div class="split wide-left">
    <div>
      ${scriptCard(mode)}
      <div class="card" id="pipeline-card">
        <div class="spread"><h3 style="margin:0">Pipeline · stage by stage</h3>
          <div class="btn-group tiny dim">${liveLabel()}</div></div>
        <div id="overall"></div>
        <div class="stepper" id="stepper"></div>
      </div>
      ${boardCard()}
      ${downloadsCard()}
      ${historyCard()}
    </div>
    <div>
      <div class="card inspect" id="inspector">${inspectorHTML()}</div>
      <div class="card" id="log-card"><div class="spread"><h3 style="margin:0">Live log</h3>
        <span class="dim tiny" id="log-n">${S.log.length}</span></div>
        <div class="log" id="log"></div></div>
    </div>
  </div>`;
  wireProject();
  renderOverall(); renderStepper(); renderLog(); renderInspector();
  if (S.run && S.run.active) startLive();
}

const sumAudio = () => S.scenes.reduce((a, s) => a + (n(s.audio_duration) || n(s.estimated_duration_sec)), 0);
const deferredCount = () => S.stages.filter((r) => r.status === 'deferred').length;
const liveLabel = () => `<span id="live-mode">${S.liveMode === 'ws' ? ' live (websocket)' : S.liveMode === 'sse' ? '🟡 live (sse)' : S.liveMode === 'poll' ? '🟠 polling' : '○ idle'}</span>`;

function integrityLine() {
  const it = S.integrity || {};
  if (!it.applies) return '';
  return it.ok ? `<div class="integrity ok" style="margin-top:8px">✔ ${esc(it.detail || 'scene texts rejoin the Director script exactly')}</div>`
    : `<div class="integrity bad" style="margin-top:8px">⚠ ${esc(it.detail || 'scene text differs from the pasted script')} ·
       <button class="btn tiny" id="btn-fix-integrity">re-run scene breakdown</button></div>`;
}

function scriptCard(mode) {
  const p = S.project || {}, sc = p.script || '';
  const editable = mode === 'B' && (p.status === 'draft' || p.status === 'review' || p.status === 'ready');
  if (!sc) {
    return `<div class="card"><h3>Script</h3><div class="hint">${mode === 'A'
      ? 'No script yet — paste one below. Nothing in the pipeline will touch its wording.'
      : 'No script yet — press “Generate / rewrite idea” to let the Controller write one from your topic hint.'}</div>
      <textarea class="script km" id="scr-paste" placeholder="ដាក់អត្ថបទខ្មែររបស់អ្នកទីនេះ — one sentence per line">${''}</textarea>
      <div class="btn-group" style="margin-top:8px"><button class="btn primary" id="scr-save">Save script</button>
      ${mode === 'B' ? '<button class="btn" id="scr-generate">or generate one</button>' : ''}</div></div>`;
  }
  return `<div class="card">
    <div class="spread"><h3 style="margin:0">Script ${mode === 'A' ? '<span class="badge A">locked · director ground truth</span>' : '<span class="badge B">reviewable</span>'}</h3>
      <div class="btn-group tiny">
        <span class="dim">${sc.split(/\n+/).filter(Boolean).length} lines · ${sc.length} chars · ~${dur(sumAudio() || estimateSec(sc))}</span>
        ${mode === 'A' ? '<button class="btn tiny" id="scr-unlock" title="only the Director may do this">✎ edit anyway</button>' : ''}
      </div></div>
    <textarea class="script km" id="scr-box" ${editable || mode === 'B' ? '' : 'readonly'}>${esc(sc)}</textarea>
    ${mode === 'B' ? `<div class="row" style="margin-top:8px">
        <label class="f" style="flex:1 1 260px"><span>note for the rewriter (optional)</span>
          <input type="text" id="scr-note" placeholder="e.g. make the last line warmer, mention rice fields"></label>
        <div class="btn-group" style="margin-top:18px">
          <button class="btn" id="scr-regen">↻ Regenerate script</button>
          <button class="btn primary" id="scr-approve-run">✅ Approve & start production</button></div></div>` : ''}
    ${mode === 'A' ? '<div class="note-box">Stage 1 may only <b>split</b> this text into scenes. If a scene list drifts from these words the studio restores them and tells you.</div>' : ''}
  </div>`;
}

function boardCard() {
  const sc = S.scenes || [];
  const rows = sc.map((s) => `<tr data-ix="${s.idx}">
      <td class="ix">${s.idx + 1}</td>
      <td><textarea data-k="text" rows="2" ${S.project.mode === 'A' ? 'readonly' : ''}>${esc(s.text)}</textarea></td>
      <td><textarea data-k="visual_prompt" rows="2">${esc(s.visual_prompt)}</textarea></td>
      <td><input data-k="mood_tag" value="${esc(s.mood_tag || '')}" list="moods">
        <div class="bar mini" style="margin-top:5px"><i style="width:${Math.min(100, (n(s.audio_duration) / Math.max(0.1, n(s.estimated_duration_sec, 1)) * 100)).toFixed(0)}%"></i></div></td>
      <td><input type="number" step="0.1" min="1" max="120" data-k="estimated_duration_sec" value="${num(s.estimated_duration_sec, 0)}"></td>
      <td><input type="number" step="0.01" data-k="audio_duration" value="${num(s.audio_duration, 0)}" readonly></td>
      <td><textarea data-k="sfx_prompt" rows="2">${esc(s.sfx_prompt)}</textarea></td>
      <td class="right nowrap">
        <button class="btn tiny" data-seescene="${s.idx}" title="inspect this scene"></button>
        <a class="btn tiny" href="/api/projects/${esc(S.projectId)}/scene/${s.idx}/download" title="zip of this scene">⬇</a></td>
    </tr>`).join('');
  return `<div class="card">
    <div class="spread"><h3 style="margin:0">Storyboard <span class="mut">្ធុនប្រាស</span></h3>
      <div class="btn-group"><span class="dim tiny" id="board-note"></span>
      <button class="btn small" id="board-add">＋ scene</button>
      <button class="btn small" id="board-save">💾 Save</button></div></div>
    ${sc.length ? `<div class="board-scroll"><table class="board">
      <thead><tr><th>#</th><th>scene text (spoken)</th><th>visual prompt (for Wan / previz)</th><th>mood</th>
      <th>est. sec</th><th>voice sec</th><th>ambience prompt</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table></div>`
      : '<div class="hint">no scenes yet — run the pipeline (Stage 1 creates them).</div>'}
    <datalist id="moods">${(S.moods || []).map((m) => `<option value="${esc(m)}">`).join('')}</datalist>
  </div>`;
}

function downloadsCard() {
  const a = S.assets || [];
  const by = {};
  a.forEach((x) => { (by[x.kind] = by[x.kind] || []).push(x); });
  const order = ['final', 'poster', 'srt', 'silent', 'manifest', 'voice_final', 'voice', 'video_fit', 'video', 'ambient', 'qa', 'scenes', 'script'];
  const chips = order.filter((k) => by[k]).map((k) => {
    const rows = by[k].sort((x, y) => (x.scene_idx || 0) - (y.scene_idx || 0)).map((x) =>
      `<div class="scene-row" style="grid-template-columns:34px minmax(0,1fr) 92px 84px 46px">
        <span class="ix">${x.scene_idx >= 0 ? x.scene_idx + 1 : '·'}</span>
        <span class="txt dim tiny" title="${esc(x.path)}">${esc(x.relpath ? x.relpath.split('/').pop() : x.path.split('/').pop())}</span>
        <span class="mono tiny dim">${dur(x.duration)}</span>
        <span class="tiny mut right">${bytes(x.size_bytes)}</span>
        <span class="right"><a class="btn tiny" href="/api/assets/${esc(x.id)}/download" title="download">⬇</a></span></div>`).join('');
    return `<details ${k === 'final' ? 'open' : ''}><summary class="tiny">${esc(k)} <span class="mut">(${by[k].length})</span></summary>${rows}</details>`;
  }).join('');
  const final = by.final ? by.final[0] : null;
  return `<div class="card">
    <div class="spread"><h3 style="margin:0">Files & downloads</h3>
      <div class="btn-group">
        ${final ? `<a class="btn primary small" href="/api/assets/${esc(final.id)}/download" download>⬇ final .mp4</a>` : ''}
        <a class="btn small" href="/api/projects/${esc(S.projectId)}/download?kind=all" title="every intermediate + final, zipped">⬇ project zip</a>
        <a class="btn small" href="/api/projects/${esc(S.projectId)}/download?kind=bundle" title="script + scenes + prompts as JSON">⬇ .json</a>
      </div></div>
    ${final ? `<div class="row" style="margin:8px 0">
        <video controls preload="metadata" src="/api/assets/${esc(final.id)}/stream" style="flex:0 0 230px"></video>
        <dl class="kv" style="flex:1 1 200px">
          <dt>duration</dt><dd>${dur(final.duration)}</dd>
          <dt>size</dt><dd>${bytes(final.size_bytes)}</dd>
          <dt>engine</dt><dd>${esc((final.meta || {}).engine || 'ffmpeg')}</dd>
          <dt>path</dt><dd>${esc(final.path)}</dd></dl></div>` : ''}
    ${chips || '<div class="hint">nothing rendered yet.</div>'}
  </div>`;
}

function historyCard() {
  const rows = (S.runs || []).map((r) => {
    const s = r.summary || {};
    return `<tr data-openrun="${esc(r.id)}" style="cursor:pointer">
      <td class="mono tiny">${esc(r.id)}</td>
      <td><span class="sbadge ${esc(r.status)}">${esc(r.status)}</span></td>
      <td class="tiny">${esc(r.trigger || '')}</td>
      <td class="tiny mono">${esc(r.machine_profile || '')}</td>
      <td class="tiny right">${(s.done || 0)}/${(s.total || 0)} · ${num(r.overall && r.overall.pct)}%</td>
      <td class="tiny right dim">${clock(r.started_at)}</td>
      <td class="tiny right"><button class="btn tiny" data-viewrun="${esc(r.id)}">open</button></td></tr>`;
  }).join('');
  return `<div class="card"><div class="spread"><h3 style="margin:0">Run history</h3>
    <span class="dim tiny">${(S.runs || []).length} run(s) · resumable</span></div>
    ${rows ? `<table class="data"><thead><tr><th>run</th><th>status</th><th>trigger</th><th>machine</th><th>jobs</th><th>started</th><th></th></tr></thead><tbody>${rows}</tbody></table>`
      : '<div class="hint">no runs yet.</div>'}</div>`;
}

/* --------------------------------------------------------------- inspector */
function inspectorHTML() {
  const stage = S.selStage;
  const meta = (S.roles || []).find((r) => r.key === stage) || {};
  const sc = S.scenes[S.selScene] || {};
  return `<div class="spread"><h3 style="margin:0">${esc(meta.emoji || '')} ${esc(meta.title || stage)}</h3>
      <select id="insp-stage" style="width:auto">${(S.roles || []).map((r) =>
        `<option value="${esc(r.key)}" ${r.key === stage ? 'selected' : ''}>${esc(r.emoji || '')} ${esc(r.title)}</option>`).join('')}</select></div>
    <div class="btn-group tiny" style="margin:8px 0">
      <button class="btn small" id="insp-prev" ${S.selScene <= 0 ? 'disabled' : ''}>← scene</button>
      <span class="dim">scene <b>${S.selScene + 1}</b>/${S.scenes.length || 1}</span>
      <button class="btn small" id="insp-next" ${S.selScene >= (S.scenes.length - 1) ? 'disabled' : ''}>scene →</button>
    </div>
    <div class="note-box km" style="margin-bottom:8px">${esc((sc.text || '').slice(0, 180))}</div>
    <div id="insp-body"><div class="hint">loading…</div></div>`;
}

async function renderInspector() {
  const body = $('#insp-body'); if (!body) return;
  if (!S.runId) { body.innerHTML = '<div class="hint">no run yet — press “Start production”.</div>'; return; }
  let d;
  try { d = await api(`/runs/${S.runId}/scenes/${S.selScene}/bundle`); }
  catch (e) { body.innerHTML = `<div class="hint">nothing for this scene yet (${esc(e.message)})</div>`; return; }
  S.bundle = d;
  const a = d.assets || {}, pk = d.peaks || {}, st = (d.stages || []);
  const row = (kind, title, render) => (a[kind] ? `<div class="asset">
      <div class="asset-head"><span class="k">${esc(title)}</span>
        <span class="btn-group"><a class="btn tiny" href="${a[kind].download}" download>⬇</a>
        ${a[kind].url && (kind.startsWith('voice') || kind === 'ambient') ? `<button class="btn tiny" data-wave="${esc(a[kind].id)}">〰</button>` : ''}</span></div>
      ${render(a[kind])}</div>` : '');
  const stageInfo = st.find((x) => x.stage === S.selStage) || {};
  const meta = (S.roles || []).find((r) => r.key === S.selStage) || {};
  const html = []
    + (row('voice', '3a base voice (sherpa / placeholder)', (x) => `<audio controls preload="metadata" src="${x.url}"></audio>
        <canvas class="wave" data-wave-src="${x.id}" width="600" height="46"></canvas>
        <div class="tiny dim mono" style="margin-top:4px">${esc(x.meta.engine || '')} · ${dur(x.duration)} · ${esc(x.size_human)}</div>`))
    + (row('voice_final', '3b your timbre (RVC)', (x) => `<audio controls preload="metadata" src="${x.url}"></audio>
        <div class="tiny dim mono" style="margin-top:4px">${esc(x.meta.engine || '')} · converted=${String(!!x.meta.converted)} · ${dur(x.duration)}</div>
        ${x.meta.reason ? `<div class="note-box">${esc(x.meta.reason)}</div>` : ''}`))
    + (row('video', '4 silent clip', (x) => `<video controls preload="metadata" src="${x.url}"></video>
        <div class="tiny dim mono" style="margin-top:4px">${esc(x.meta.engine || '')} · ${dur(x.duration)}</div>`))
    + (row('video_fit', '4b duration-matched clip', (x) => `<video controls preload="metadata" src="${x.url}"></video>
        <div class="tiny dim mono">${esc(x.meta.engine || '')} · drift ${num(x.meta.drift_sec, 0)}s</div>`))
    + (row('ambient', '5 ambience / SFX', (x) => `<audio controls preload="metadata" src="${x.url}"></audio>
        <div class="tiny dim mono" style="margin-top:4px">${esc(x.meta.engine || '')} · ${esc((x.meta.layers || []).join(', '))}</div>`))
    + (row('qa', '6 QA report', (x) => qaHTML(x.meta)))
    ;
  body.innerHTML = `
    ${html || '<div class="hint">no output for this scene on this stage yet.</div>'}
    ${stageInfo.error ? `<div class="err-box">${esc(stageInfo.error)}</div>` : ''}
    ${stageInfo.status ? `<div class="spread tiny" style="margin:6px 0"><span class="sbadge ${esc(stageInfo.status)}">${esc(stageInfo.status)}</span>
       <span class="dim mono">${esc(stageInfo.engine || '')} · ${num(stageInfo.progress)}% · ${esc(stageInfo.message || '')}</span></div>` : ''}
    <div class="section-title">prompts used for this scene (SQLite memory)</div>
    ${(d.prompts || []).length ? (d.prompts || []).map((p2) => `<details class="mem-hit"><summary class="tiny">
        <b>${esc(p2.stage || p2.role)}</b> · ${esc(p2.model || p2.engine)} · ${clock(p2.created_at)}</summary>
        ${p2.system ? `<div class="section-title">system</div><pre class="prompt-box">${esc(p2.system)}</pre>` : ''}
        <div class="section-title">user</div><pre class="prompt-box">${esc(p2.user)}</pre>
        ${p2.response ? `<div class="section-title">response</div><pre class="prompt-box">${esc(p2.response)}</pre>` : ''}</details>`).join('')
      : '<div class="hint">none recorded.</div>'}
    <div class="section-title">re-run this stage with a different instruction</div>
    <label class="f"><span>visual prompt</span><input type="text" id="ov-vp" value="${esc((d.scene || {}).visual_prompt || '')}"></label>
    <div class="row">
      <label class="f" style="flex:1"><span>mood tag</span><input type="text" id="ov-mood" value="${esc((d.scene || {}).mood_tag || '')}"></label>
      <label class="f" style="flex:2"><span>ambience prompt</span><input type="text" id="ov-sfx" value="${esc((d.scene || {}).sfx_prompt || '')}"></label>
    </div>
    <div class="btn-group">
      <button class="btn primary small" id="insp-regen">↻ Regenerate ${esc(meta.title || S.selStage)} for scene ${S.selScene + 1}</button>
      <button class="btn small" id="insp-preview">🎨 previz test</button>
      ${S.project && S.project.mode === 'A' ? '<span class="hint">text stays locked to your script</span>' : ''}
    </div>`;
  $$('#insp-body video').forEach((v) => { v.onerror = () => v.replaceWith(el('<div class="note-box">this browser cannot play this codec — the file still downloads.</div>')); });
  $$('#insp-body canvas[data-wave-src]').forEach((c) => drawWave(c, c.dataset.waveSrc));
}

function qaHTML(meta) {
  const issues = meta.issues || [];
  return `<div class="spread tiny"><b>${meta.approved ? '✅ approved' : '⚠ flagged'}</b>
    <span class="dim">${esc(meta.engine || '')}</span></div>
    ${meta.summary ? `<div class="tiny dim" style="margin:4px 0">${esc(meta.summary)}</div>` : ''}
    ${issues.map((i) => `<div class="issue ${esc(i.severity || 'info')}"><b>${esc(i.severity || 'info')}</b> · ${esc(i.issue || '')}${i.fix ? ` <span class="dim">→ ${esc(i.fix)}</span>` : ''}</div>`).join('')
      || '<div class="tiny dim">no issues</div>'}`;
}

async function drawWave(canvas, assetId) {
  try {
    const d = await api(`/assets/${assetId}/waveform?bins=${Math.max(60, Math.floor(canvas.clientWidth || 600) / 4)}`);
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
    const h = canvas.height = 46 * (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#57c7a8';
    const pk = d.peaks || []; const bw = w / Math.max(1, pk.length);
    pk.forEach((v, i) => { const hh = Math.max(1, v * h * 0.92); ctx.fillRect(i * bw, (h - hh) / 2, Math.max(1, bw * 0.7), hh); });
  } catch (e) { /* waveform is decoration only */ }
}

function wireProject() {
  const on = (id, fn, ev) => { const x = $(id); if (x) x[ev || 'onclick'] = fn; };
  on('#btn-refresh', () => refreshProject().then(() => renderProject($('#view'))).catch((e) => toast(e.message, 'err')));
  on('#btn-run', () => startRun());
  on('#btn-pause', async () => { toast((await api(`/runs/${S.runId}/pause`, { method: 'POST' })).note || 'pausing after the current job…', 'warn'); });
  on('#btn-resume', async () => { await api(`/runs/${S.runId}/resume`, { method: 'POST' }); startLive(); renderProject($('#view')); });
  on('#btn-cancel', async () => { if (!confirm('Cancel this run? Finished stages stay reusable.')) return; await api(`/runs/${S.runId}/cancel`, { method: 'POST' }); toast('cancelled', 'warn'); stopLive(); setTimeout(() => refreshProject().then(() => renderProject($('#view'))), 1200); });
  on('#btn-continue', async () => { const r = await api(`/runs/${S.runId}/continue`, { method: 'POST' }); S.runId = r.run_id; startLive(); renderProject($('#view')); });
  on('#btn-catchup', async () => { const r = await api(`/projects/${S.projectId}/catchup`, { method: 'POST' }); S.runId = r.run_id; toast(r.note || 'catch-up started'); startLive(); renderProject($('#view')); });
  on('#btn-idea', () => generateIdea());
  on('#btn-approve', () => approveScript(true));
  on('#scr-generate', () => generateIdea());
  on('#scr-save', async () => {
    try { await api(`/projects/${S.projectId}`, { method: 'PATCH', json: { script: $('#scr-paste').value, director_override: true } }); toast('script saved', 'ok'); await refreshProject(); renderProject($('#view')); }
    catch (e) { toast(e.message, 'err'); }
  });
  on('#scr-unlock', () => {
    const b = $('#scr-box'); b.readOnly = false; b.focus(); b.style.outline = '1px solid #e5b567';
    toast('Unlocked for the Director. Save with the button below — the studio will re-verify the scene split against your new text.', 'warn', 7000);
    const bar = el('<div class="btn-group" style="margin-top:8px"><button class="btn primary small" id="scr-forced">Save (director override)</button><button class="btn small" id="scr-cancel">discard</button></div>');
    b.parentNode.insertBefore(bar, b.nextSibling);
    $('#scr-forced').onclick = async () => {
      try { const r = await api(`/projects/${S.projectId}`, { method: 'PATCH', json: { script: b.value, director_override: true } }); toast(r.note || 'saved', 'ok'); await refreshProject(); renderProject($('#view')); } catch (e) { toast(e.message, 'err'); }
    };
    $('#scr-cancel').onclick = () => { b.value = (S.project.script || ''); b.readOnly = true; bar.remove(); };
  });
  on('#scr-regen', () => regenerateScript());
  on('#scr-approve-run', () => approveScript(true));
  on('#board-save', () => saveBoard());
  on('#board-add', () => {
    const t = $('#board-add').closest('.card');
    const tb = $('.board tbody', t); if (!tb) return;
    tb.appendChild(el(`<tr data-new="1"><td class="ix">+</td><td><textarea data-k="text" rows="2"></textarea></td>
      <td><textarea data-k="visual_prompt" rows="2"></textarea></td><td><input data-k="mood_tag" value="sunrise-warm"></td>
      <td><input type="number" step="0.1" data-k="estimated_duration_sec" value="5"></td>
      <td><input data-k="audio_duration" value="0" readonly></td><td><textarea data-k="sfx_prompt" rows="2"></textarea></td>
      <td class="right"><button class="btn tiny" data-droprow="1">✕</button></td></tr>`));
    $('[data-droprow]', tb.lastElementChild).onclick = () => tb.lastElementChild.remove();
  });
  on('#btn-fix-integrity', () => rerunStage('breakdown', null, {}));
  on('#insp-prev', () => { S.selScene = Math.max(0, S.selScene - 1); renderInspector(); });
  on('#insp-next', () => { S.selScene = Math.min((S.scenes.length || 1) - 1, S.selScene + 1); renderInspector(); });
  on('#insp-regen', () => {
    const ovs = { visual_prompt: $('#ov-vp').value.trim(), mood_tag: $('#ov-mood').value.trim(), sfx_prompt: $('#ov-sfx').value.trim() };
    rerunStage(S.selStage, S.selScene, ovs);
  });
  on('#insp-preview', async () => {
    try {
      const r = await api('/preview/previz', { method: 'POST', json: { mood_tag: $('#ov-mood').value.trim(), visual_prompt: $('#ov-vp').value.trim(), duration: 2.5, seed: Math.floor(Math.random() * 9999) } });
      const box = el(`<div><video controls autoplay src="${r.url}" style="max-height:60vh"></video>
        <div class="hint" style="margin-top:8px">${esc(r.engine)} · ${dur(r.duration)} · seed ${esc(r.seed === undefined ? '?' : r.seed)}</div>
        <div class="btn-group" style="margin-top:8px"><a class="btn small" href="${r.url}" download>⬇ save clip</a></div></div>`);
      modal('previz test', box);
    } catch (e) { toast(e.message, 'err'); }
  });
  const sel = $('#insp-stage'); if (sel) sel.onchange = () => { S.selStage = sel.value; $('#inspector').innerHTML = inspectorHTML(); wireProject(); renderInspector(); };
  $$('#view [data-seescene]').forEach((b) => { b.onclick = () => { S.selScene = +b.dataset.seescene; S.selStage = S.selStage || 'voice_final'; renderInspector(); $('#inspector').scrollIntoView({ behavior: 'smooth', block: 'start' }); }; });
  $$('#view [data-viewrun]').forEach((b) => { b.onclick = (e) => { e.stopPropagation(); S.runId = b.dataset.viewrun; S.lastEventId = 0; renderProject($('#view')); }; });
  $$('#view tr[data-openrun]').forEach((tr) => { tr.onclick = () => { S.runId = tr.dataset.openrun; S.lastEventId = 0; renderProject($('#view')); }; });
}

/* ---------------------------------------------------- project actions */
async function startRun(payload) {
  try {
    const r = await api(`/projects/${S.projectId}/runs`, { method: 'POST', json: payload || { trigger: 'new' } });
    S.runId = r.run_id; S.lastEventId = 0;
    toast('run ' + r.run_id + ' started · ' + (r.job_count || '?') + ' jobs', 'ok');
    await refreshProject(); renderProject($('#view')); startLive();
  } catch (e) { toast(e.message, 'err'); }
}

async function rerunStage(stage, sceneIdx, overrides) {
  if (!S.runId) { toast('start a run first', 'warn'); return; }
  try {
    const r = await api(`/runs/${S.runId}/stages/${stage}/regenerate`, {
      method: 'POST', json: { scene_idx: sceneIdx, overrides: overrides || {}, skip_qa_gate: false } });
    S.runId = r.run_id; S.lastEventId = 0;
    toast(r.note || ('re-running ' + stage + (sceneIdx === null ? '' : ' for scene ' + (sceneIdx + 1))), 'ok');
    startLive();
    await refreshProject(); renderStepper(); renderOverall();
  } catch (e) { toast(e.message, 'err'); }
}

async function generateIdea() {
  const box = $('#view');
  toast('asking the Controller for a script…', '', 2000);
  try {
    const r = await api(`/projects/${S.projectId}/generate-idea`, { method: 'POST' });
    if (r.error) toast('idea failed: ' + r.error, 'err');
    await refreshProject(); renderProject(box);
  } catch (e) { toast(e.message, 'err'); }
}

async function approveScript(start) {
  const scr = ($('#scr-box') || {}).value;
  try {
    const r = await api(`/projects/${S.projectId}/approve-script`, { method: 'POST', json: { script: scr, start: !!start } });
    toast('script approved', 'ok', 1800);
    if (r.run_id) { S.runId = r.run_id; S.lastEventId = 0; startLive(); }
    await refreshProject(); renderProject($('#view'));
  } catch (e) { toast(e.message, 'err'); }
}

async function regenerateScript() {
  const note = ($('#scr-note') || {}).value || '';
  try {
    const r = await api(`/projects/${S.projectId}/regenerate-script`, { method: 'POST', json: { note } });
    toast('new draft from the Controller (' + (r.origin || 'auto') + ')', 'ok');
    await refreshProject(); renderProject($('#view'));
  } catch (e) { toast(e.message, 'err'); }
}

async function saveBoard() {
  const rows = $$('#view .board tbody tr');
  const scenes = rows.map((tr) => {
    const o = { idx: tr.dataset.ix };
    $$('[data-k]', tr).forEach((i) => { o[i.dataset.k] = i.value; });
    return o;
  });
  try {
    const r = await api(`/projects/${S.projectId}/scenes`, { method: 'POST', json: { scenes } });
    toast(r.note || 'storyboard saved', 'ok');
    if (r.integrity && !r.integrity.ok) toast('⚠ ' + r.integrity.detail, 'warn', 8000);
    await refreshProject(); renderProject($('#view'));
  } catch (e) { toast(e.message, 'err'); }
}

/* ------------------------------------------------------- live progress */
function startLive() {
  stopLive();
  if (!S.runId) return;
  const poll = async () => {
    try {
      await refreshRun();
      renderOverall(); renderStepper(); renderLog();
      if (S.view === 'project') renderInspector();
      if (S.run && S.run.active === false) { S.liveMode = 'idle'; stopLive(); refreshProject().then(() => { renderTop(); }); return; }
    } catch (e) { S.liveMode = 'poll'; }
  };
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  let ws = null;
  try {
    ws = new WebSocket(`${proto}://${location.host}/api/runs/${S.runId}/events`);
    S.live = ws; S.liveMode = 'ws';
    const dot = $('#live-dot'); if (dot) dot.className = 'dot on';
    let pending = false;
    ws.onmessage = (ev) => {
      if (pending) return; pending = true;
      setTimeout(async () => { pending = false; await poll(); }, 120);   // coalesce bursts
    };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
    ws.onclose = () => {
      if (S.live !== ws) return;
      S.live = trySSE(poll);
    };
  } catch (e) { S.live = trySSE(poll); }
  S.pollTimer = setInterval(poll, 5000);
  const lm = $('#live-mode'); if (lm) lm.textContent = liveLabel().replace(/<\/?span[^>]*>/g, '');
}

function trySSE(poll) {
  try {
    const es = new EventSource(`/api/runs/${S.runId}/stream`);
    S.liveMode = 'sse';
    const dot = $('#live-dot'); if (dot) dot.className = 'dot poll';
    let pending = false;
    es.onmessage = () => { if (pending) return; pending = true; setTimeout(async () => { pending = false; await poll(); }, 150); };
    es.onerror = () => { es.close(); S.liveMode = 'poll'; const d2 = $('#live-dot'); if (d2) d2.className = 'dot off'; };
    return es;
  } catch (e) { S.liveMode = 'poll'; return null; }
}

function stopLive() {
  if (S.live) { try { S.live.close(); } catch (e) {} S.live = null; }
  if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
  const dot = $('#live-dot'); if (dot) dot.className = 'dot off';
  S.liveMode = (S.run && S.run.active) ? S.liveMode : 'idle';
}

function renderOverall() {
  const host = $('#overall'); if (!host) return;
  const o = S.overall || { pct: 0, done: 0, total: 0 };
  const st = (S.run && S.run.run) || {};
  host.innerHTML = `<div class="spread tiny" style="margin:10px 0 8px">
      <div style="flex:1 1 220px"><div class="bar" style="height:8px"><i style="width:${n(o.pct)}%"></i></div></div>
      <span class="dim mono">${(o.done || 0)}/${(o.total || 0)} jobs · ${num(o.pct)}%</span>
      <span class="sbadge ${esc(st.status || 'queued')}">${esc(st.status || 'idle')}</span>
      ${st.started_at ? `<span class="dim tiny">${dur((Date.now() / 1000) - st.started_at)} elapsed</span>` : ''}
    </div>
    ${S.runPlan ? `<div class="row tiny" style="margin-bottom:6px">${['tts', 'rvc', 'video', 'sfx'].map((k) =>
      `<span class="chip ${(S.runPlan[k] || {}).run === false ? 'off' : 'on'}">${esc(k)}: ${esc((S.runPlan[k] || {}).engine || '?')}</span>`).join('')}</div>` : ''}`;
}

function renderStepper() {
  const host = $('#stepper'); if (!host) return;
  const byStage = S.byStage || {};
  const rows = (S.roles && S.roles.length ? S.roles : STAGE_ORDER.map((k) => ({ key: k, title: k })));
  host.innerHTML = rows.map((r, i) => {
    const k = r.key, agg = byStage[k] || {};
    const jobs = S.stages.filter((x) => x.stage === k);
    const status = agg.status || (jobs.length ? 'queued' : 'pending');
    const pct = agg.pct === undefined ? 0 : agg.pct;
    const expanded = S.expanded[k] !== undefined ? S.expanded[k] : (status === 'running' || status === 'failed' || jobs.length <= 1);
    const per = r.per_scene ? (S.scenes.length || agg.total || 0) : 1;
    return `<div class="stage ${esc(status)}" data-stage="${esc(k)}">
      <div class="stage-head">
        <div class="stage-num">${i + 1}</div>
        <div class="stage-title" data-toggle="${esc(k)}">${esc(r.emoji || '')} ${esc(r.title || k)}
          <small>${esc(r.blurb || '')}</small></div>
        <div class="stage-engine" data-toggle="${esc(k)}">${esc((agg.engines || []).join('/') || r.model || '—')}</div>
        <div class="bar" data-toggle="${esc(k)}"><i style="width:${pct}%"></i></div>
        <div class="stage-status" data-toggle="${esc(k)}"><span class="sbadge ${esc(status)}">${esc(status)}</span>
          <span class="dim tiny"> ${(agg.done || 0)}/${per}</span></div>
      </div>
      <div class="stage-body" ${expanded ? '' : 'hidden'}>
        <div class="tiny dim" style="margin-bottom:6px">${esc(agg.last_message || r.blurb || '')}
          ${agg.elapsed_ms ? ' · ' + dur(agg.elapsed_ms / 1000) : ''}</div>
        ${r.per_scene ? sceneRowsFor(k, jobs) : globalRowFor(k, jobs[0])}
        <div class="btn-group tiny" style="margin-top:8px">
          <button class="btn tiny" data-regen="${esc(k)}">↻ regenerate stage</button>
          ${deferrable(r) ? '<button class="btn tiny" data-catchup="1">🖥️ GPU catch-up</button>' : ''}
          ${r.role ? `<span class="chip" title="LLM role model">model: ${esc(r.model || '—')}</span>` : ''}
        </div>
      </div></div>`;
  }).join('');
  $$('#stepper .stage-head').forEach((h) => {
    h.onclick = () => { const key = h.dataset.toggle; const body = h.parentNode.querySelector('.stage-body'); S.expanded[key] = body.hidden; body.hidden = !body.hidden; };
  });
  $$('#stepper [data-regen]').forEach((b) => {
    b.onclick = (e) => { e.stopPropagation(); rerunStage(b.dataset.regen, null, {}); };
  });
  $$('#stepper [data-catchup]').forEach((b) => { b.onclick = (e) => { e.stopPropagation(); $('#btn-catchup') ? $('#btn-catchup').click() : toast('nothing deferred on this run', 'warn'); }; });
  $$('#stepper .scene-row[data-ix]').forEach((r) => {
    r.onclick = () => { S.selScene = +r.dataset.ix; S.selStage = r.dataset.stage; $$('#stepper .scene-row').forEach((x) => x.classList.remove('sel')); r.classList.add('sel'); if ($('#insp-stage')) $('#insp-stage').value = S.selStage; renderInspector(); };
  });
  $$('#stepper [data-regscene]').forEach((b) => {
    b.onclick = (e) => { e.stopPropagation(); rerunStage(b.dataset.regen2, +b.dataset.ix, {}); };
  });
}

const deferrable = (r) => r.deferrable || r.requires_gpu;

function sceneRowsFor(stage, jobs) {
  if (!jobs.length) return '<div class="hint tiny">queued — nothing started yet</div>';
  const byIx = {}; jobs.forEach((j) => { byIx[j.scene_idx] = j; });
  return S.scenes.map((s) => {
    const j = byIx[s.idx]; if (!j) return '';
    return `<div class="scene-row" data-ix="${s.idx}" data-stage="${esc(stage)}">
      <span class="ix">${s.idx + 1}</span>
      <span class="txt km" title="${esc(s.text)}">${esc(s.text.slice(0, 90))}</span>
      <span><span class="sbadge ${esc(j.status)}">${esc(j.status)}</span></span>
      <span class="dim tiny" title="${esc(j.error || j.message || '')}">${esc((j.message || j.error || '').slice(0, 110))}
        ${j.progress > 0 && j.progress < 100 ? `<div class="bar mini"><i style="width:${j.progress}%"></i></div>` : ''}</span>
      <span class="right"><span class="mono tiny mut">${esc(j.engine || '')}</span>
        <button class="btn tiny" data-regscene="1" data-regen2="${esc(stage)}" data-ix="${s.idx}" title="re-run only this scene">↻</button></span>
    </div>${j.error ? `<div class="err-box">${esc(j.error)}</div>` : ''}`;
  }).join('');
}

function globalRowFor(stage, j) {
  if (!j) return '<div class="hint tiny">queued</div>';
  return `<div class="scene-row" data-ix="-1" data-stage="${esc(stage)}" style="grid-template-columns:34px minmax(0,1fr) 92px minmax(0,1.1fr) 118px">
    <span class="ix">·</span><span class="txt dim">${esc(j.message || '')}</span>
    <span><span class="sbadge ${esc(j.status)}">${esc(j.status)}</span></span>
    <span class="dim tiny">${esc(j.error || '')}</span>
    <span class="right mono tiny mut">${esc(j.engine || '')}</span></div>`;
}

function renderLog() {
  const host = $('#log'); if (!host) return;
  const rows = (S.log || []).slice(-90).reverse();
  host.innerHTML = rows.map((e) => `<div><span class="t">${new Date((e.ts || 0) * 1000).toLocaleTimeString()}</span>
    <span class="k ${esc(e.kind)}">${esc(e.kind)}</span> ${esc(e.stage || '')}${e.scene_idx >= 0 ? '#' + e.scene_idx : ''}
    <span class="dim">${esc(shortPayload(e))}</span></div>`).join('') || '<div class="dim">no events yet</div>';
  const n2 = $('#log-n'); if (n2) n2.textContent = (S.log || []).length + ' events';
}

function shortPayload(e) {
  const p = e.payload || {};
  const bits = [];
  ['message', 'engine', 'note', 'pct', 'status', 'error', 'stage', 'jobs', 'path'].forEach((k) => {
    if (p[k] !== undefined && p[k] !== '') bits.push(k + '=' + (k === 'path' ? String(p[k]).split('/').pop() : p[k]));
  });
  return bits.join(' · ').slice(0, 160);
}

/* =================================================================== voices */
async function renderVoices(root) {
  root.innerHTML = '<div class="empty">loading voices…</div>';
  let d = { voices: [], discovered: [], rvc: {} };
  try { d = await api('/voices'); } catch (e) { toast(e.message, 'err'); }
  S.voices = d.voices || []; S.discovered = d.discovered || []; S.rvcInfo = d.rvc || {};
  const sel = (S.settings.rvc || {}).profile_id || (S.project && S.project.voice_profile_id) || '';
  root.innerHTML = `
  <div class="split">
    <div>
      <div class="card">
        <div class="spread"><h3 style="margin:0">Voice profiles <span class="mut">សំឡេងរបស់អ្នក</span></h3>
          <div class="btn-group">
            <button class="btn small" id="v-import">⇥ import from RVC folder (${(d.discovered || []).length})</button>
            <button class="btn primary small" id="v-new">＋ add a voice</button></div></div>
        <div class="hint" style="margin-bottom:10px">Stage 3a speaks Khmer with sherpa-onnx (vits-mms-khm).
          Stage 3b re-timbres that audio with <b>your</b> RVC model, so the words stay correct while the voice
          stays yours. Trained voices live in <code class="mono">${esc((d.rvc || {}).webui_dir || 'RVC-WebUI/asset/models')}</code>;
          the API is <code class="mono">${esc((d.rvc || {}).api_base || 'http://127.0.0.1:9513')}</code>.</div>
        ${(S.voices.length) ? S.voices.map((v) => `<div class="tile" style="margin-bottom:9px">
            <div class="spread"><b>${esc(v.name)}</b>
              <span class="btn-group tiny">
                <button class="btn tiny ${v.id === sel ? 'primary' : ''}" data-select="${esc(v.id)}">${v.id === sel ? '✓ default' : 'use as default'}</button>
                ${S.projectId ? `<button class="btn tiny" data-proj="${esc(v.id)}" title="use for this project only">this project</button>` : ''}
                <button class="btn tiny" data-preview="${esc(v.id)}">🔊 preview</button>
                <a class="btn tiny" href="${esc(v.sample_url || '#')}" ${v.sample_url ? '' : 'aria-disabled=true'}>sample</a>
                <button class="btn tiny" data-train="${esc(v.id)}">🎯 train</button>
                <button class="btn tiny danger" data-del="${esc(v.id)}">🗑</button></div></div>
            <div class="row tiny" style="margin-top:5px">
              <span class="chip ${v.pth_exists ? 'on' : 'off'}">.pth ${v.pth_exists ? 'found' : 'missing'}</span>
              <span class="chip ${v.index_path ? (v.index_exists ? 'on' : 'off') : 'off'}">index ${v.index_path ? (v.index_exists ? 'found' : 'missing') : 'none'}</span>
              <span class="chip ${v.sample_path ? 'on' : 'off'}">sample ${v.sample_seconds ? dur(v.sample_seconds) : '—'}</span>
              <span class="chip">pitch ${n(v.pitch, 0)}</span>
              ${v.training_status ? `<span class="chip ${v.training_status === 'done' ? 'on' : 'off'}">training: ${esc(v.training_status)}</span>` : ''}
            </div>
            ${v.notes ? `<div class="hint" style="margin-top:4px">${esc(v.notes)}</div>` : ''}
            ${v.sample_url ? `<audio controls preload="none" src="${esc(v.sample_url)}" style="margin-top:6px"></audio>` : ''}
          </div>`).join('')
          : `<div class="empty"><div class="big">🎙️</div>no voice profiles yet.
             ${d.discovered && d.discovered.length ? 'Press <b>import</b> to pick up the ones RVC WebUI already has,' : 'Train a model in RVC WebUI with 10–15 minutes of clean speech,'}
             then add it here.</div>`}
      </div>
      ${(d.discovered || []).length ? `<div class="card"><h3>Discovered on disk</h3>
        <table class="data"><thead><tr><th>name</th><th>.pth</th><th>index</th></tr></thead><tbody>
        ${d.discovered.map((x) => `<tr><td>${esc(x.name)}</td><td class="mono tiny">${esc((x.pth_path || '').split(/[\\\\/]/).pop())}</td>
          <td class="mono tiny">${esc(x.index_path ? x.index_path.split(/[\\\\/]/).pop() : '—')}</td></tr>`).join('')}
        </tbody></table></div>` : ''}
    </div>
    <div>
      <div class="card"><h3>Hear it now</h3>
        <label class="f"><span>text to speak</span>
          <textarea class="km" id="v-text">សួស្ដី។ ខ្ញុំកំពុងនិយាយដោយស្ងប់ស្ងាត់ ហើយខ្ញុំមិនបោះបង់ទេ។</textarea></label>
        <button class="btn primary" id="v-go">Synthesise 3a + convert 3b</button>
        <div id="v-out" style="margin-top:10px"></div>
        <div class="note-box" style="margin-top:8px">This uses the exact engines the pipeline will use, so it is
          the fastest way to check a new voice model before spending 10 minutes on a render.</div></div>
      <div class="card"><h3>How to train the model</h3>
        <ol class="hint" style="padding-left:18px;margin:0">
          <li style="margin-bottom:6px">Record 10–15 min of clean speech (no music, no room echo).</li>
          <li style="margin-bottom:6px">In RVC WebUI: <i>Train</i> → set <i>dataset path</i> to that folder → 200–400 epochs on 8&nbsp;GB, then index it.</li>
          <li style="margin-bottom:6px">Put <code>.pth</code> in <code>assets/weights/</code> and <code>.index</code> in <code>assets/index/</code>.</li>
          <li style="margin-bottom:6px">Press <i>import from RVC folder</i>, pick it as default, and the next run's stage 3b uses it.</li>
        </ol>
        <div class="note-box" style="margin-top:8px">You can also upload the .pth/.index + sample here — the studio
          normalises the sample for you and can launch the training command you configured in Settings.</div></div>
    </div>
  </div>`;
  const on = (id, fn) => { const x = $(id); if (x) x.onclick = fn; };
  on('#v-new', () => voiceForm());
  on('#v-import', async () => {
    try { const r = await api('/voices/import-discovered', { method: 'POST' }); toast(`imported ${r.added.length} of ${r.found} found`, 'ok'); renderVoices(root); }
    catch (e) { toast(e.message, 'err'); }
  });
  on('#v-go', async () => {
    const v = S.voices[0];
    if (!v) { toast('add a voice profile first', 'warn'); return; }
    $('#v-out').innerHTML = '<div class="hint">synthesising…</div>';
    try {
      const r = await api(`/voices/${v.id}/preview`, { method: 'POST', json: { text: $('#v-text').value } });
      $('#v-out').innerHTML = `<div class="asset"><div class="k tiny dim">3a base (${esc((r.tts || {}).engine || '')})</div>
          <audio controls src="${esc(r.base_url)}"></audio></div>
        <div class="asset" style="margin-top:8px"><div class="k tiny dim">3b your timbre (${esc((r.rvc || {}).engine || '')})</div>
          ${r.final_url ? `<audio controls src="${esc(r.final_url)}"></audio>` : `<div class="note-box">not converted: ${esc((r.rvc || {}).reason || 'RVC unavailable')}</div>`}</div>`;
    } catch (e) { $('#v-out').innerHTML = `<div class="err-box">${esc(e.message)}</div>`; }
  });
  $$('#view [data-select]').forEach((b) => { b.onclick = async () => {
    try { await api(`/voices/${b.dataset.select}/select`, { method: 'POST', json: {} }); toast('default voice set', 'ok'); await loadSettings(); renderVoices(root); } catch (e) { toast(e.message, 'err'); } }; });
  $$('#view [data-proj]').forEach((b) => { b.onclick = async () => {
    try { await api(`/voices/${b.dataset.proj}/select?project_id=${S.projectId}`, { method: 'POST', json: {} }); toast('this project will use that voice', 'ok'); await refreshProject(); renderProject($('#view')); } catch (e) { toast(e.message, 'err'); } }; });
  $$('#view [data-preview]').forEach((b) => { b.onclick = async () => {
    toast('synthesising…', '', 1500);
    try { const r = await api(`/voices/${b.dataset.preview}/preview`, { method: 'POST', json: { text: $('#v-text') ? $('#v-text').value : '' } });
      modal('voice preview', el(`<div><audio controls autoplay src="${esc(r.base_url)}" style="width:100%"></audio>
        ${r.final_url ? `<div class="section-title">after RVC</div><audio controls src="${esc(r.final_url)}" style="width:100%"></audio>` : ''}
        <div class="hint" style="margin-top:8px">3a ${esc((r.tts || {}).engine || '')} · 3b ${esc((r.rvc || {}).engine || '')} ${r.converted ? '(converted)' : '(bypassed)'}</div></div>`));
    } catch (e) { toast(e.message, 'err'); } }; });
  $$('#view [data-train]').forEach((b) => { b.onclick = async () => {
    if (!confirm('Launch the RVC training command configured in Settings? It runs on this machine and can take 20–60 min.')) return;
    try { const r = await api(`/voices/${b.dataset.train}/train`, { method: 'POST' }); watchTraining(r.job_id); toast('training started: ' + r.job_id); }
    catch (e) { toast(e.message, 'err'); } }; });
  $$('#view [data-del]').forEach((b) => { b.onclick = async () => {
    if (!confirm('Remove this voice profile from the studio? Files on disk stay.')) return;
    try { await api(`/voices/${b.dataset.del}`, { method: 'DELETE' }); renderVoices(root); } catch (e) { toast(e.message, 'err'); } }; });
}

function voiceForm() {
  const box = el(`<div>
    <div class="grid two">
      <label class="f"><span>name</span><input type="text" id="vf-name" placeholder="My voice"></label>
      <label class="f"><span>pitch shift (semitones)</span><input type="number" id="vf-pitch" min="-12" max="12" value="0"></label>
    </div>
    <label class="f"><span>notes (optional)</span><input type="text" id="vf-notes"></label>
    <div class="grid three">
      <label class="f"><span>RVC .pth weights (if already trained)</span><input type="file" id="vf-pth" accept=".pth,.onnx"></label>
      <label class="f"><span>.index (optional)</span><input type="file" id="vf-index" accept=".index"></label>
      <label class="f"><span>10–15 min training sample (audio or video)</span><input type="file" id="vf-sample" accept="audio/*,video/*,.zip"></label>
    </div>
    <div class="hint">Already trained the model in RVC WebUI? You only need the <code>.pth</code> here —
      or press "import from RVC folder" and skip uploading entirely.
      Otherwise just upload your raw recording as the training sample (a video works fine — only its
      audio track is used) and press "Save profile", then use the Train button to kick off RVC training.</div>
    <div class="spread" style="margin-top:12px"><span></span><button class="btn primary" id="vf-save">Save profile</button></div>
    <div id="vf-out"></div></div>`);
  modal('Add a voice profile', box);
  $('#vf-save').onclick = async () => {
    const fd = new FormData();
    fd.append('name', $('#vf-name').value || 'My Voice');
    fd.append('pitch', $('#vf-pitch').value || '0');
    fd.append('notes', $('#vf-notes').value || '');
    ['pth', 'index', 'sample'].forEach((k) => { const f = $('#vf-' + k).files[0]; if (f) fd.append(k === 'pth' ? 'pth' : k, f); });
    $('#vf-save').disabled = true; $('#vf-out').innerHTML = '<div class="hint">uploading…</div>';
    try {
      const r = await api('/voices', { method: 'POST', form: fd });
      const w = (r.voice.warnings || []);
      $('#vf-out').innerHTML = `<div class="note-box">saved · sample ${dur(r.voice.sample_seconds)}${w.length ? ' · ⚠ ' + esc(w.join('; ')) : ''}</div>
        ${r.training_command ? `<div class="section-title">suggested training command (copy into Settings → RVC)</div><pre class="prompt-box">${esc(r.training_command)}</pre>` : ''}`;
      toast('voice profile saved', 'ok');
      setTimeout(() => { closeModal(); renderVoices($('#view')); }, 1800);
    } catch (e) { $('#vf-out').innerHTML = `<div class="err-box">${esc(e.message)}</div>`; $('#vf-save').disabled = false; }
  };
}

async function watchTraining(jobId) {
  const box = el('<div><div class="log" id="tr-log">starting…</div><div class="spread" style="margin-top:10px"><span class="dim tiny" id="tr-st">running</span><button class="btn" id="tr-close">close</button></div></div>');
  modal('RVC training log', box);
  $('#tr-close').onclick = closeModal;
  const tick = async () => {
    if (closeModal.hidden) return;
    try {
      const d = await api('/training/' + jobId);
      const l = $('#tr-log'); if (!l) return;
      l.textContent = (d.lines || []).slice(-400).join('\\n'); l.scrollTop = l.scrollHeight;
      $('#tr-st').textContent = d.status + (d.ok === false ? ' — see the log above' : '');
      if (d.status !== 'running') { await loadSettings(); }
    } catch (e) { /* modal may be closed */ }
  };
  await tick();
  const t = setInterval(() => { if (!$('#tr-log')) { clearInterval(t); return; } tick(); }, 1500);
}

/* =================================================================== memory */
async function renderMemory(root) {
  root.innerHTML = `<div class="spread"><h2 style="margin:0">Memory <span class="mut">ความចាំ</span></h2>
    <div class="btn-group tiny"><button class="btn small" data-mtab="prompts">prompt ledger</button>
    <button class="btn small" data-mtab="assets">assets</button></div></div>
  <div class="hint" style="margin:6px 0 10px">SQLite history of every script, scene, prompt and output file — searchable and re-usable.</div>
  <div class="filters">
    <input type="search" id="m-q" placeholder="search scripts, prompts, scene text, visual prompts…" value="${esc(S.memQ || '')}">
    <button class="btn small" id="m-go">Search</button>
    <span class="dim tiny" id="m-hint"></span>
  </div>
  <div id="m-body"><div class="empty">…</div></div>`;
  const go2 = async () => {
    S.memQ = $('#m-q').value.trim();
    $('#m-hint').textContent = 'searching…';
    try {
      const d = await api('/memory/search?q=' + encodeURIComponent(S.memQ) + '&limit=80');
      S.mem = d; renderMem();
    } catch (e) { toast(e.message, 'err'); }
  };
  $('#m-go').onclick = go2;
  $('#m-q').onkeydown = (e) => { if (e.key === 'Enter') go2(); };
  $$('[data-mtab]').forEach((b) => { b.onclick = () => { S.memTab = b.dataset.mtab; renderMem(); }; });
  if (S.mem) renderMem(); else go2();
  async function renderMem() {
    const body = $('#m-body'); const tab = S.memTab || 'prompts';
    $$('[data-mtab]').forEach((b) => b.classList.toggle('primary', b.dataset.mtab === tab));
    if (tab === 'assets') {
      const d = await api('/assets?limit=300');
      const rows = (d.assets || []).map((a) => `<tr>
        <td class="mono tiny"><a href="#/project/${esc(a.project_id)}">${esc(a.project_id)}</a></td>
        <td>${esc(a.kind)}${a.scene_idx >= 0 ? ' #' + (a.scene_idx + 1) : ''}</td>
        <td class="mono tiny dim" title="${esc(a.path)}">${esc(a.relpath || a.path)}</td>
        <td class="tiny right">${dur(a.duration)}</td><td class="tiny right">${bytes(a.size_bytes)}</td>
        <td class="tiny right">${clock(a.created_at)}</td>
        <td class="right nowrap"><a class="btn tiny" href="/api/assets/${esc(a.id)}/stream">▶</a>
          <a class="btn tiny" href="/api/assets/${esc(a.id)}/download" download>⬇</a></td></tr>`).join('');
      body.innerHTML = `<div class="card"><table class="data"><thead><tr><th>project</th><th>kind</th><th>path</th><th></th><th></th><th>created</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
      if ($('#m-hint')) $('#m-hint').textContent = (d.assets || []).length + ' assets';
      return;
    }
    const m = S.mem || { projects: [], prompts: [], scenes: [] };
    const hl = (t) => esc(t || '').replace(new RegExp('(' + S.memQ.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi'), '<mark>$1</mark>');
    const proj = m.projects.map((p) => `<div class="mem-hit" data-nav="${esc(p.id)}">
        <div class="h">${hl(p.title)} <span class="badge ${esc(p.mode)}">Mode ${esc(p.mode)}</span> <span class="badge ${esc(p.status)}">${esc(p.status)}</span></div>
        <div class="b km">${hl(p.excerpt)}</div></div>`).join('');
    const prm = (m.prompts || []).map((p) => `<details class="mem-hit"><summary>
        <span class="mono tiny">${esc(p.stage || p.role)} · ${esc(p.model || '')}</span>
        <span class="dim tiny"> · ${clock(p.created_at)}</span></summary>
        <div class="section-title">user</div><pre class="prompt-box">${hl(p.user_excerpt || p.user)}</pre>
        ${p.response_excerpt || p.response ? `<div class="section-title">response</div><pre class="prompt-box">${hl(p.response_excerpt || p.response)}</pre>` : ''}
        <div class="btn-group tiny" style="margin-top:6px"><a class="btn tiny" href="#/project/${esc(p.project_id)}">open project</a></div></details>`).join('');
    const scn = (m.scenes || []).map((s) => `<div class="mem-hit" data-nav="${esc(s.project_id)}">
        <div class="h"><span class="mono tiny">scene ${s.idx + 1}</span> · <span class="badge">${esc(s.mood_tag || '')}</span></div>
        <div class="b km">${hl(s.text)}</div>
        <div class="b" style="margin-top:4px;color:var(--fg-mut)">${hl(s.visual_prompt)}</div></div>`).join('');
    body.innerHTML = `
      ${S.memQ ? `<div class="card"><h3>Projects <span class="mut tiny">${m.projects.length}</span></h3>${proj || '<div class="hint">none</div>'}</div>` : ''}
      <div class="card"><h3>Prompt ledger <span class="mut tiny">every model call · ${prm ? '' : 'latest'}</span></h3>
        ${prm || emptyPrompts()}</div>
      ${S.memQ ? `<div class="card"><h3>Scenes <span class="mut tiny">${m.scenes.length}</span></h3>${scn || '<div class="hint">none</div>'}</div>` : ''}`;
    if ($('#m-hint')) $('#m-hint').textContent = `${(m.prompts || []).length} prompts${S.memQ ? ' · ' + m.projects.length + ' projects · ' + m.scenes.length + ' scenes' : ''}`;
    $$('#m-body [data-nav]').forEach((x) => { x.onclick = (e) => { if (e.target.closest('a')) return; go('project', x.dataset.nav); }; });
  }
  function emptyPrompts() {
    return `<div class="hint">nothing recorded yet. Rows appear as soon as a run touches a model —
      each fallback engine (placeholder voice, previz, procedural ambience) logs the prompt it actually used,
      so you can copy it later.</div>`;
  }
}

/* ================================================================= settings */
async function renderSettings(root) {
  let meta = {};
  try { meta = await api('/settings'); } catch (e) { toast(e.message, 'err'); }
  S.settings = meta.settings || S.settings; S.plan = meta.plan || S.plan; S.roles = meta.roles || S.roles;
  const c = S.settings || {}, p = S.plan || {};
  const roleKeys = ((meta.llm_roles || {}).keys) || ['controller', 'auto_idea', 'qa'];
  const roleLabels = ((meta.llm_roles || {}).labels) || {};
  const profiles = meta.machine_profiles || {};
  const roles = (c.ollama || {}).roles || {};
  const eng = (k) => (p[k] || {}).engine || '—';
  const reason = (k) => (p[k] || {}).reason || '';
  const opt = (vals, cur) => vals.map((v) => `<option value="${esc(v)}" ${String(v) === String(cur) ? 'selected' : ''}>${esc(v)}</option>`).join('');
  const chk = (id, on, label) => `<div class="check"><input type="checkbox" id="${id}" ${on ? 'checked' : ''}><span>${label}</span></div>`;
  root.innerHTML = `
  <div class="spread"><h2 style="margin:0">Settings</h2>
    <div class="btn-group"><button class="btn small" id="s-probe">🔌 Re-probe services</button>
      <button class="btn small" id="s-reset">revert</button>
      <button class="btn primary" id="s-save">💾 Save</button></div></div>
  <div class="hint" id="s-note">saved to <code class="mono">${esc(c._config_path || 'studio_settings.json')}</code> · engine choices take effect on the next run</div>
  <div class="split">
    <div>
      <div class="card"><h3>Roles → models <span class="mut">same shape as ai_creator/team.py</span></h3>
        ${roleKeys.map((r) => { const o = roles[r] || {}; return `<div class="row" style="align-items:flex-end">
          <label class="f" style="flex:2 1 190px"><span>${esc(roleLabels[r] || r)}</span>
            <input type="text" data-set="ollama.roles.${r}.model" value="${esc(o.model || '')}" placeholder="sailor2:8b" list="models"></label>
          <label class="f" style="flex:1 1 150px"><span>fallback</span>
            <input type="text" data-set="ollama.roles.${r}.fallback_model" value="${esc(o.fallback_model || '')}" placeholder="llama3.2:3b" list="models"></label>
          <label class="f" style="flex:0 0 96px"><span>temp</span>
            <input type="number" step="0.05" min="0" max="2" data-set="ollama.roles.${r}.temperature" value="${n(o.temperature, 0.5)}"></label>
          <label class="f" style="flex:0 0 84px"><span>on</span>
            <input type="checkbox" style="width:auto;margin-top:9px" data-set="ollama.roles.${r}.enabled" ${o.enabled !== false ? 'checked' : ''}></label>
        </div>`; }).join('')}
        <datalist id="models"><option value="sailor2:8b"><option value="sailor2:1b"><option value="llama3.2:3b"><option value="qwen2.5:3b"></datalist>
        <div class="row" style="margin-top:6px">
          <label class="f" style="flex:1"><span>Ollama host</span><input type="text" data-set="ollama.host" value="${esc((c.ollama || {}).host)}"></label>
          <label class="f" style="flex:0 0 120px"><span>num_ctx</span><input type="number" step="512" data-set="ollama.num_ctx" value="${n((c.ollama || {}).num_ctx, 4096)}"></label>
          <label class="f" style="flex:0 0 130px"><span>timeout (s)</span><input type="number" data-set="ollama.request_timeout_sec" value="${n((c.ollama || {}).request_timeout_sec, 300)}"></label>
        </div>
        <div class="section-title">what each stage will actually run on this machine</div>
        <table class="data"><tbody>
          ${[['tts', '3a Khmer voice'], ['rvc', '3b your timbre'], ['video', '4 video'], ['sfx', '5 ambience']]
            .map(([k, label]) => `<tr><td class="nowrap">${esc(label)}</td>
              <td><span class="chip ${/defer|off|placeholder|bypass/.test(eng(k)) ? 'off' : 'on'}">${esc(eng(k))}</span></td>
              <td class="tiny dim">${esc(reason(k))}</td></tr>`).join('')}
          <tr><td>language model</td><td><span class="chip ${(p.ollama || {}).available ? 'on' : 'off'}">${(p.ollama || {}).available ? 'online' : 'offline'}</span></td>
            <td class="tiny dim">${esc((p.ollama || {}).reason || '')}</td></tr>
        </tbody></table>
      </div>

      <div class="card"><h3>Engines</h3>
        <div class="grid two">
          <label class="f"><span>3a voice</span><select data-set="tts.engine">${opt(['auto', 'sherpa', 'piper', 'kokoro', 'placeholder'], (c.tts || {}).engine)}</select>
            <div class="field-note">${esc(reason('tts'))}</div></label>
          <label class="f"><span>3b timbre</span><select data-set="rvc.engine">${opt(['auto', 'http', 'cli', 'bypass', 'off'], (c.rvc || {}).engine)}</select>
            <div class="field-note">${esc(reason('rvc'))}</div></label>
          <label class="f"><span>4 video</span><select data-set="video.engine">${opt(['auto', 'comfyui', 'previz', 'defer', 'off'], (c.video || {}).engine)}</select>
            <div class="field-note">${esc(reason('video'))}</div></label>
          <label class="f"><span>5 ambience</span><select data-set="sfx.engine">${opt(['auto', 'mmaudio', 'procedural', 'defer', 'off'], (c.sfx || {}).engine)}</select>
            <div class="field-note">${esc(reason('sfx'))}</div></label>
        </div>
        <div class="grid two">
          <label class="f"><span>sherpa model dir (model.onnx + tokens.txt)</span>
            <input type="text" data-set="tts.model_dir" value="${esc((c.tts || {}).model_dir || '')}" placeholder="models/tts/vits-mms-khm"></label>
          <label class="f"><span>speech speed (1.0 = natural · 0.9 = calmer)</span>
            <input type="number" step="0.05" min="0.5" max="2" data-set="tts.speed" value="${n((c.tts || {}).speed, 1)}"></label>
          <label class="f"><span>RVC WebUI folder (to find your trained voices)</span>
            <input type="text" data-set="rvc.webui_dir" value="${esc((c.rvc || {}).webui_dir || '')}" placeholder="C:\\Users\\you\\RVC-WebUI"></label>
          <label class="f"><span>RVC voices folder (relative ok)</span>
            <input type="text" data-set="rvc.models_dir" value="${esc((c.rvc || {}).models_dir || 'models/rvc')}"></label>
          <label class="f"><span>RVC API base (inference API, optional)</span>
            <input type="text" data-set="rvc.api_base" value="${esc((c.rvc || {}).api_base || 'http://127.0.0.1:9513')}"></label>
          <label class="f"><span>RVC training command (opt-in, streamed in Voices)</span>
            <input type="text" data-set="rvc.train_command" value="${esc((c.rvc || {}).train_command || '')}" placeholder="python infer_train.py -exp ${'{exp}'} -dataset ${'{sample_dir}'}"></label>
        </div>
        <div class="row">
          <label class="f" style="flex:1"><span>pitch shift (semitones, RVC) <b id="pitch-n">${n((c.rvc || {}).pitch, 0)}</b></span>
            <input type="range" min="-12" max="12" step="1" data-set="rvc.pitch" value="${n((c.rvc || {}).pitch, 0)}"></label>
          <label class="f" style="flex:1"><span>index rate (timbre strictness)</span>
            <input type="number" step="0.05" min="0" max="1" data-set="rvc.index_rate" value="${n((c.rvc || {}).index_rate, 0.75)}"></label>
          <label class="f" style="flex:1"><span>f0 method</span>
            <select data-set="rvc.f0_method">${opt(['rmvpe', 'pm', 'harvest', 'crepe', 'fcpe'], (c.rvc || {}).f0_method || 'rmvpe')}</select></label>
        </div>
        ${chk('rvc-clean', (c.rvc || {}).clean, 'Denoise the input before conversion (helps phone-recorded samples).')}
      </div>

      <div class="card"><h3>Video · Wan through ComfyUI <span class="mut">480p vertical is the house standard</span></h3>
        <div class="grid three">
          <label class="f"><span>width</span><input type="number" step="16" min="256" max="1280" data-set="video.width" value="${n((c.video || {}).width, 480)}"></label>
          <label class="f"><span>height</span><input type="number" step="16" min="256" max="1280" data-set="video.height" value="${n((c.video || {}).height, 854)}"></label>
          <label class="f"><span>fps</span><input type="number" step="1" min="8" max="30" data-set="video.fps" value="${n((c.video || {}).fps, 16)}"></label>
          <label class="f"><span>steps <span class="mut">20 = good/8GB</span></span><input type="number" step="1" min="4" max="60" data-set="video.steps" value="${n((c.video || {}).steps, 20)}"></label>
          <label class="f"><span>cfg</span><input type="number" step="0.5" min="1" max="20" data-set="video.cfg" value="${n((c.video || {}).cfg, 6)}"></label>
          <label class="f"><span>shift</span><input type="number" step="0.5" min="0" max="20" data-set="video.shift" value="${n((c.video || {}).shift, 8)}"></label>
          <label class="f"><span>max frames / clip</span><input type="number" step="4" min="17" max="121" data-set="video.max_frames" value="${n((c.video || {}).max_frames, 81)}"></label>
          <label class="f"><span>seed (-1 = per run)</span><input type="number" data-set="video.seed" value="${n((c.video || {}).seed, -1)}"></label>
          <label class="f"><span>workflow</span><select data-set="video.workflow" id="wf-select"><option value="">built-in default</option></select></label>
        </div>
        <div class="row">
          <label class="f" style="flex:1"><span>motion strength <b id="motion-n">${num((c.video || {}).motion_strength, 0.75)}</b></span>
            <input type="range" min="0" max="2" step="0.05" data-set="video.motion_strength" value="${n((c.video || {}).motion_strength, 0.75)}"></label>
          <label class="f" style="flex:2"><span>negative prompt</span><input type="text" data-set="video.negative_prompt" value="${esc((c.video || {}).negative_prompt || '')}"></label>
        </div>
        <div class="section-title">SFX director</div>
        <div class="grid three">
          <label class="f"><span>MMAudio workflow</span><select data-set="sfx.workflow" id="wf-select-sfx"><option value="mmaudio_small_480p">mmaudio_small_480p</option></select></label>
          <label class="f"><span>ambient gain</span><input type="number" step="0.05" min="0" max="2" data-set="sfx.ambient_gain" value="${n((c.sfx || {}).ambient_gain, 1)}"></label>
          <label class="f"><span>voice duck gain (lower = quieter ambience)</span><input type="number" step="0.02" min="0.05" max="1" data-set="sfx.voice_duck_gain" value="${n((c.sfx || {}).voice_duck_gain, 0.32)}"></label>
        </div>
      </div>

      <div class="card"><h3>VRAM safety <span class="mut">8 GB is the design point</span></h3>
        <div class="grid three">
          <label class="f"><span>hard cap (MB)</span><input type="number" step="128" min="1024" max="49152" data-set="vram.limit_mb" value="${n((c.vram || {}).limit_mb, 8192)}"></label>
          <label class="f"><span>keep free for desktop (MB)</span><input type="number" step="64" min="0" max="8192" data-set="vram.reserve_free_mb" value="${n((c.vram || {}).reserve_free_mb, 900)}"></label>
          <label class="f"><span>max scene seconds per model</span><input type="number" step="1" min="4" max="60" data-set="vram.max_scene_seconds_for_model" value="${n((c.vram || {}).max_scene_seconds_for_model, 14)}"></label>
        </div>
        ${chk('vram-ser', (c.vram || {}).serialize_gpu !== false, 'One GPU job at a time (video and MMAudio never share the card).')}
        ${chk('vram-down', (c.vram || {}).downscale_on_pressure !== false, 'Auto-drop steps / frames / resolution when free VRAM is tight.')}
        ${chk('vram-unload', (c.vram || {}).unload_llm_after_stage !== false, 'Ask Ollama to unload the LLM after text stages (frees VRAM for Wan).')}
        <div id="vram-card" class="note-box" style="margin-top:8px">${esc(vramSummary())}</div>
      </div>
    </div>

    <div>
      <div class="card"><h3>Machine profile</h3>
        <label class="f"><span>which machine is this</span><select data-set="machine.profile">
          ${Object.keys(profiles).length ? Object.entries(profiles).map(([k, v]) =>
            `<option value="${esc(k)}" ${((c.machine || {}).profile || 'auto') === k ? 'selected' : ''}>${esc(v.label || k)}</option>`).join('')
            : opt(['auto', 'machine_a', 'machine_b'], (c.machine || {}).profile || 'auto')}
        </select></label>
        <div class="hint">${esc(((profiles[(c.machine || {}).profile || 'auto'] || {}).desc) || 'detected at startup')}</div>
        ${chk('m-cpu', (c.machine || {}).force_cpu_only, 'Force CPU-only mode (no CUDA calls at all — Machine B / laptop on battery).')}
      </div>

      <div class="card"><h3>Pipeline</h3>
        <div class="grid two">
          <label class="f"><span>max scenes</span><input type="number" min="1" max="40" data-set="pipeline.max_scenes" value="${n((c.pipeline || {}).max_scenes, 12)}"></label>
          <label class="f"><span>duration tolerance (s)</span><input type="number" step="0.1" min="0.1" max="5" data-set="pipeline.duration_tolerance_sec" value="${n((c.pipeline || {}).duration_tolerance_sec, 0.9)}"></label>
          <label class="f"><span>scene length target (s)</span><input type="number" step="0.5" min="2" max="30" data-set="pipeline.scene_target_seconds" value="${n((c.pipeline || {}).scene_target_seconds, 6)}"></label>
          <label class="f"><span>retries per stage</span><input type="number" step="1" min="0" max="3" data-set="pipeline.retry_limit" value="${n((c.pipeline || {}).retry_limit, 1)}"></label>
          <label class="f"><span>review gate</span><select data-set="pipeline.review_gate">${opt(['auto', 'always', 'never'], (c.pipeline || {}).review_gate)}</select></label>
          <label class="f"><span>QA blocks assembly</span><select data-set="pipeline.require_qa_pass">${opt([false, true], (c.pipeline || {}).require_qa_pass)}</select></label>
        </div>
        ${chk('pl-autoB', (c.pipeline || {}).auto_approve_mode_b, 'Mode B full autonomy — approve the generated script and keep rendering.')}
        ${chk('pl-keep', (c.pipeline || {}).keep_intermediate !== false, 'Keep intermediates on disk (needed to regenerate a single stage later).')}
        <div class="section-title">concurrency</div>
        <div class="grid three">
          <label class="f"><span>LLM</span><input type="number" min="1" max="8" data-set="pipeline.concurrency.llm" value="${n(((c.pipeline || {}).concurrency || {}).llm, 1)}"></label>
          <label class="f"><span>voice</span><input type="number" min="1" max="8" data-set="pipeline.concurrency.tts" value="${n(((c.pipeline || {}).concurrency || {}).tts, 1)}"></label>
          <label class="f"><span>GPU</span><input type="number" min="1" max="4" data-set="pipeline.concurrency.gpu" value="${n(((c.pipeline || {}).concurrency || {}).gpu, 1)}"></label>
          <label class="f"><span>CPU</span><input type="number" min="1" max="16" data-set="pipeline.concurrency.cpu" value="${n(((c.pipeline || {}).concurrency || {}).cpu, 2)}"></label>
          <label class="f"><span>IO</span><input type="number" min="1" max="16" data-set="pipeline.concurrency.io" value="${n(((c.pipeline || {}).concurrency || {}).io, 4)}"></label>
        </div>
      </div>

      <div class="card"><h3>Final assembly</h3>
        <div class="grid two">
          <label class="f"><span>fps</span><input type="number" min="12" max="60" data-set="assembly.fps" value="${n((c.assembly || {}).fps, 24)}"></label>
          <label class="f"><span>crf (lower = better)</span><input type="number" min="14" max="34" data-set="assembly.crf" value="${n((c.assembly || {}).crf, 23)}"></label>
          <label class="f"><span>x264 preset</span><select data-set="assembly.preset">${opt(['ultrafast', 'veryfast', 'faster', 'medium'], (c.assembly || {}).preset)}</select></label>
          <label class="f"><span>audio kbps</span><input type="number" min="64" max="320" step="16" data-set="assembly.audio_kbps" value="${n((c.assembly || {}).audio_kbps, 160)}"></label>
          <label class="f"><span>loudness target (LUFS)</span><input type="number" step="0.5" min="-24" max="-9" data-set="assembly.loudnorm_target_lufs" value="${n((c.assembly || {}).loudnorm_target_lufs, -16)}"></label>
          <label class="f"><span>transition</span><select data-set="assembly.transition">${opt(['crossfade', 'cut', 'fade'], (c.assembly || {}).transition)}</select></label>
          <label class="f"><span>fade / crossfade (s)</span><input type="number" step="0.05" min="0" max="2" data-set="assembly.fade_sec" value="${n((c.assembly || {}).fade_sec, 0.35)}"></label>
        </div>
        ${chk('a-srt', (c.assembly || {}).emit_srt !== false, 'Write the .srt subtitle sidecar (Khmer text).')}
        ${chk('a-burn', (c.assembly || {}).burn_captions, 'Burn captions into the picture — needs a Khmer-capable font installed, otherwise use the .srt in your editor.')}
        ${chk('a-manifest', (c.assembly || {}).emit_manifest !== false, 'Write manifest.json (every stage, engine, prompt, file).')}
      </div>

      <div class="card"><h3>Paths & services</h3>
        <dl class="kv">
          <dt>data dir</dt><dd>${esc((S.status || {}).data_dir || '')}</dd>
          <dt>ffmpeg</dt><dd>${esc((S.status || {}).ffmpeg || '⚠ not found')}</dd>
          <dt>gpu</dt><dd>${esc(JSON.stringify(((S.status || {}).machine || {}).gpus || []))}</dd>
          <dt>vram free</dt><dd>${esc(String(safe(((S.status || {}).vram || {}).free_mb)))} / ${esc(String(((S.status || {}).vram || {}).limit_mb))} MB</dd>
          <dt>projects</dt><dd>${esc(String(((S.status || {}).db || {}).projects))} · assets ${esc(String(((S.status || {}).db || {}).assets))} · prompts ${esc(String(((S.status || {}).db || {}).prompts))}</dd>
        </dl>
        <div class="btn-group tiny" style="margin-top:8px">
          <a class="btn small" href="/api-summary" target="_blank">raw status</a>
          <a class="btn small" href="/api/prompts?limit=200" target="_blank">prompts json</a>
        </div>
        <div id="probe-out" style="margin-top:10px">${probeCardHtml()}</div>
      </div>
      <div class="card"><h3>House style (read-only)</h3>
        <div class="hint">every generated or tagged string inherits this guideline — it is compiled into the prompts, not a suggestion.</div>
        <button class="btn small" style="margin-top:8px" id="s-style">view guideline</button>
        <div class="section-title">ComfyUI workflow templates</div>
        <div id="wf-list" class="hint">loading…</div>
      </div>
    </div>
  </div>`;
  wireSettings(c);
}

function vramSummary() {
  const v = (S.status || {}).vram || {};
  const s = v.plan_6s || {};
  if (!Object.keys(s).length) return 'no VRAM reading yet (nvidia-smi missing on this machine — the cap is still enforced by serialising GPU jobs).';
  return `suggested 6 s clip at the current settings: ${esc(s.width)}×${esc(s.height)} @ ${esc(s.frames)} frames`
    + ((s.notes || []).length ? ` · ${esc(s.notes.join('; '))}` : '');
}

function probeCardHtml() {
  const pr = S.probe;
  if (!pr) return '<div class="hint">press <b>Re-probe services</b> to check Ollama, sherpa, RVC, ComfyUI and VRAM right now.</div>';
  const line = (k, d) => {
    const x = d || {};
    return `<div class="spread tiny" style="border-bottom:1px dashed var(--line-soft);padding:5px 0">
      <b>${esc(k)}</b><span class="chip ${x.ok || x.available || x.online ? 'on' : 'off'}">${esc(x.engine || x.model || x.detail || (x.ok ? 'ok' : 'unavailable'))}</span></div>
      ${x.reason || x.note || x.detail ? `<div class="hint" style="margin:2px 0 4px">${esc(x.reason || x.note || x.detail)}</div>` : ''}`;
  };
  return `<div class="section-title">last probe</div>`
    + ['tts', 'rvc', 'video', 'sfx'].map((k) => line(k, pr[k])).join('')
    + (pr.capabilities ? `<pre class="prompt-box" style="margin-top:8px">${esc(JSON.stringify(pr.capabilities, null, 1))}</pre>` : '');
}

function wireSettings(c) {
  const on = (id, fn, ev) => { const x = $(id); if (x) x[ev || 'onclick'] = fn; };
  $$('[data-set]').forEach((inp) => {
    if (inp.type === 'range') {
      inp.oninput = () => {
        const lbl = $('#' + (inp.dataset.set.split('.').pop() === 'pitch' ? 'pitch-n' : 'motion-n'));
        if (lbl) lbl.textContent = inp.value;
      };
    }
  });
  on('#s-style', () => showStyle());
  on('#s-probe', async () => {
    $('#probe-out').innerHTML = '<div class="hint">probing…</div>';
    try {
      const r = await api('/settings/probe', { method: 'POST' });
      S.probe = r; S.plan = r.plan || S.plan;
      $('#probe-out').innerHTML = probeCardHtml();
      await loadSettings(); renderTop();
      toast('probed — engine plan updated (rerender the settings page to see chips)', 'ok');
    } catch (e) { $('#probe-out').innerHTML = `<div class="err-box">${esc(e.message)}</div>`; }
  });
  on('#s-reset', async () => { await loadSettings(); renderSettings($('#view')); });
  on('#s-save', async () => {
    const patch = {};
    $$('[data-set]').forEach((inp) => {
      const path = inp.dataset.set.split('.');
      let v;
      if (inp.type === 'checkbox') v = inp.checked;
      else if (inp.type === 'number') v = parseFloat(inp.value);
      else if (inp.value === 'true') v = true; else if (inp.value === 'false') v = false; else v = inp.value;
      let o = patch;
      for (let i = 0; i < path.length - 1; i++) { o[path[i]] = o[path[i]] || {}; o = o[path[i]]; }
      o[path[path.length - 1]] = v;
    });
    try {
      const r = await api('/settings', { method: 'POST', json: patch });
      S.settings = r.settings; S.plan = r.plan;
      toast(r.note || 'saved', 'ok');
      await loadSettings(); renderSettings($('#view'));
    } catch (e) { toast(e.message, 'err'); }
  });
  (async () => {
    try {
      const d = await api('/workflows');
      const names = uniq((d.workflows || []).map((w) => String(w.name).replace(/\.json$/, '')));
      const opts = names.map((x) => `<option value="${esc(x)}">${esc(x)}</option>`).join('');
      ['#wf-select', '#wf-select-sfx'].forEach((id) => { const s = $(id); if (s && opts) s.innerHTML = (id === '#wf-select' ? '<option value="">built-in default</option>' : '') + opts; });
      $('#wf-list').innerHTML = (d.workflows || []).length
        ? `<table class="data"><tbody>${(d.workflows || []).map((w) => `<tr>
            <td class="mono tiny">${esc(String(w.name).replace(/\.json$/, ''))}</td>
            <td class="tiny dim">${(w.placeholders || []).length} placeholders</td>
            <td class="tiny right mono dim" title="${esc(w.path || '')}">${bytes(w.size)}</td></tr>`).join('')}</tbody></table>
          <div class="hint" style="margin-top:6px">${(d.dirs || []).length} folder(s): ${(d.dirs || []).map((x) => '<code>' + esc(x) + '</code>').join(' · ')}</div>
          ${d.known ? '<div class="hint">tokens: <code>' + esc((d.known || []).join(' ')) + '</code></div>' : ''}`
        : '<div class="hint">no workflow files found — the built-in defaults are used automatically.</div>';
    } catch (e) { $('#wf-list').innerHTML = `<div class="hint">${esc(e.message)}</div>`; }
  })();
}
