/* AI Content Creator — frontend logic (vanilla JS, no build step) */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  status: null,
  team: null,
  models: [],
  characters: [],
  voices: [],
  sfx: [],
  plan: null,
  activeCharId: null,
  style: "punchy, energetic, short-form",
  job: null,
  jobTimer: null,
  mediaRecorder: null,
  recChunks: [],
};

/* ---------------- helpers ---------------- */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-json */ }
  if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
  return data;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
function flash(el, msg, ok = true) {
  el.textContent = msg;
  el.style.color = ok ? "var(--good)" : "var(--bad)";
  setTimeout(() => { el.textContent = ""; }, 4000);
}

/* ---------------- steps ---------------- */
function gotoStep(n) {
  $$(".step").forEach((b) => b.classList.toggle("active", b.dataset.step === String(n)));
  for (let i = 1; i <= 4; i++) $("#panel-" + i).classList.toggle("hidden", i !== n);
  if (n === 2) renderTeamStep();
  if (n === 4) prepareRenderStep();
}
$$(".step").forEach((b) => b.addEventListener("click", () => gotoStep(b.dataset.step)));

/* ---------------- status ---------------- */
async function refreshStatus() {
  try {
    state.status = await api("/api/status");
    const st = state.status;
    const pO = $("#pill-ollama");
    if (st.ollama.online) {
      pO.textContent = `Ollama ● ${st.ollama.models.length} model${st.ollama.models.length === 1 ? "" : "s"}`;
      pO.className = "pill pill-on";
    } else {
      pO.textContent = "Ollama ○ offline (fallback mode)";
      pO.className = "pill pill-warn";
    }
    const engines = [];
    if (st.tts.xtts) engines.push("XTTS clone");
    if (st.tts.kokoro) engines.push("Kokoro");
    if (st.tts.gtts) engines.push("gTTS");
    const pT = $("#pill-tts");
    if (engines.length) { pT.textContent = `Voice ● ${engines.join(" · ")}`; pT.className = "pill pill-on"; }
    else { pT.textContent = "Voice ○ no engine (SFX only)"; pT.className = "pill pill-warn"; }
  } catch (e) { /* server warmup */ }
}

/* ---------------- step 1: character ---------------- */
let createFile = null;
$("#file-create").addEventListener("change", (e) => {
  createFile = e.target.files[0] || null;
  $("#dz-create .dz-title").textContent = createFile ? `Selected: ${createFile.name}` : "Drop a photo of your character here";
});
const dz = $("#dz-create");
["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) { createFile = f; $("#dz-create .dz-title").textContent = `Selected: ${f.name}`; }
});

$("#btn-create-char").addEventListener("click", async () => {
  if (!createFile) { flash($("#char-msg"), "Pick a photo of your character first.", false); return; }
  const btn = $("#btn-create-char");
  btn.disabled = true;
  const fd = new FormData();
  fd.append("name", $("#char-name").value || "My Character");
  fd.append("file", createFile);
  try {
    const r = await api("/api/characters/create", { method: "POST", body: fd });
    await loadCharacters();
    state.activeCharId = r.character.id;
    createFile = null;
    $("#char-name").value = "";
    $("#dz-create .dz-title").textContent = "Drop a photo of your character here";
    $("#file-create").value = "";
    flash($("#char-msg"), `Character "${r.character.name}" created — memory saved ✓`);
    markStepsDone(2);
    gotoStep(2);
  } catch (e) { flash($("#char-msg"), e.message, false); }
  finally { btn.disabled = false; }
});

function charCard(c) {
  const el = document.createElement("div");
  el.className = "card char-card";
  const faceBadge = c.face_detected
    ? '<span class="badge good">face detected ✓</span>'
    : '<span class="badge warn">face not found — using full photo</span>';
  const swatches = (c.palette || []).map((p) =>
    `<span class="swatch" style="background:rgb(${p[0]},${p[1]},${p[2]})"></span>`).join("");
  el.innerHTML = `
    <img class="char-avatar" src="${c.assets.avatar}" alt="">
    <div class="char-info">
      <div class="char-name">${esc(c.name)}</div>
      <div class="mini-row">${faceBadge}<span class="badge">${c.photos} photo${c.photos === 1 ? "" : "s"} remembered</span></div>
      <div class="swatches" title="remembered colors">${swatches}</div>
      <div class="mini-row">
        <label class="muted">Voice:</label>
        <select class="input char-voice" style="padding:6px 10px;font-size:12.5px;max-width:170px">
          <option value="">(none)</option>
          ${state.voices.map((v) => `<option value="${v.id}" ${c.voice_id === v.id ? "selected" : ""}>${esc(v.name)}</option>`).join("")}
        </select>
        <label class="btn btn-ghost btn-sm char-add-photo">＋ photos<input type="file" multiple accept="image/*" hidden></label>
        <button class="btn btn-ghost btn-sm char-del">✕</button>
      </div>
      <div class="char-sim"></div>
    </div>`;
  el.querySelector(".char-voice").addEventListener("change", async (e) => {
    try {
      await api("/api/characters", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: c.id, voice_id: e.target.value || null }) });
    } catch (err) { alert(err.message); }
  });
  el.querySelector(".char-add-photo input").addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    for (const f of files) {
      const fd = new FormData();
      fd.append("file", f);
      try {
        const r = await api(`/api/characters/${c.id}/photos`, { method: "POST", body: fd });
        const sim = r.similarity || {};
        const cls = sim.verdict === "same" ? "good" : sim.verdict === "maybe" ? "warn" : "bad";
        const el2 = el.querySelector(".char-sim");
        if (sim.verdict) el2.innerHTML = `<span class="badge ${cls}">new photo: looks ${esc(sim.verdict)}
          (face match ${100 - Math.min(100, sim.hamming * 2)}%, colors Δ ${Math.round((sim.palette_delta || 0) * 100)}%)</span>`;
        await loadCharacters();
        renderCharacters();
      } catch (err) { alert(err.message); }
    }
  });
  el.querySelector(".char-del").addEventListener("click", async () => {
    if (!confirm(`Delete character "${c.name}" and all its photos?`)) return;
    try {
      await api(`/api/characters/${c.id}`, { method: "DELETE" });
      if (state.activeCharId === c.id) state.activeCharId = null;
      await loadCharacters();
    } catch (e) { alert(e.message); }
  });
  return el;
}

async function loadCharacters() {
  state.characters = await api("/api/characters");
  if (state.characters.length && !state.activeCharId) state.activeCharId = state.characters[0].id;
  renderCharacters();
  fillCharSelect();
}
function renderCharacters() {
  const grid = $("#char-list");
  const create = $("#char-create");
  grid.innerHTML = "";
  if (!state.characters.length) { grid.style.display = "none"; create.style.display = ""; return; }
  create.style.display = "none";
  grid.style.display = "";
  state.characters.forEach((c) => grid.appendChild(charCard(c)));
}

/* ---------------- voices ---------------- */
async function loadVoices() {
  state.voices = await api("/api/voices");
  renderVoices();
}
function renderVoices() {
  const list = $("#voice-list");
  list.innerHTML = "";
  if (!state.voices.length) {
    list.innerHTML = '<div class="muted">No voices yet — record or upload a 30–60s sample of your voice.</div>';
    return;
  }
  const tpl = $("#tpl-voice-row");
  state.voices.forEach((v) => {
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.querySelector(".voice-name").textContent = v.name;
    node.querySelector(".voice-meta").textContent =
      `${v.duration}s · ${v.clone_engine === "xtts" ? "XTTS clone ready" : "clone pending (install TTS) or Kokoro fallback"}`;
    node.querySelector(".play").addEventListener("click", async () => {
      const res = await fetch(v.preview);
      const blob = await res.blob();
      new Audio(URL.createObjectURL(blob)).play();
    });
    node.querySelector(".del").addEventListener("click", async () => {
      await api(`/api/voices/${v.id}`, { method: "DELETE" });
      await loadVoices();
      await loadCharacters();
    });
    list.appendChild(node);
  });
}
$("#file-voice").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("name", f.name.replace(/\.[^.]+$/, ""));
  fd.append("file", f);
  try {
    await api("/api/voices", { method: "POST", body: fd });
    await loadVoices();
  } catch (err) { alert(err.message); }
});

$("#btn-mic").addEventListener("click", async () => {
  const btn = $("#btn-mic");
  const st = $("#rec-status");
  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    state.mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    state.recChunks = [];
    rec.ondataavailable = (e) => state.recChunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.textContent = "🎤 Record voice";
      st.textContent = "";
      const blob = new Blob(state.recChunks, { type: rec.mimeType || "audio/webm" });
      const fd = new FormData();
      fd.append("name", "Recorded voice");
      fd.append("file", new File([blob], "recording.webm"));
      try {
        await api("/api/voices", { method: "POST", body: fd });
        await loadVoices();
      } catch (err) { alert(err.message); }
    };
    rec.start();
    state.mediaRecorder = rec;
    btn.textContent = "⏹ Stop recording";
    st.textContent = "Recording… speak for 30–60 seconds, then stop.";
  } catch (e) {
    st.textContent = "Microphone unavailable: " + e.message;
  }
});

/* ---------------- step 2: AI team ---------------- */
async function loadTeam() {
  state.team = await api("/api/team");
}
async function loadModels() {
  const r = await api("/api/ollama/models");
  state.models = r.models;
  $("#model-count").textContent = r.online ? `${r.models.length} local model(s) found` : "";
  $("#ollama-offline-banner").classList.toggle("hidden", r.online);
  renderModelChips();
}
function renderModelChips() {
  const box = $("#model-chips");
  box.innerHTML = state.models.length
    ? state.models.map((m) => `<span class="model-chip">${esc(m)}</span>`).join("")
    : '<span class="muted">No models yet — run <code>ollama pull llama3.2:3b</code></span>';
}
$("#btn-refresh-models").addEventListener("click", async () => {
  $("#team-host").value = $("#team-host").value.trim();
  await refreshStatus();
  await loadModels();
});

function renderTeamStep() {
  if (!state.team) return;
  $("#team-host").value = state.team.ollama_host;
  const grid = $("#role-cards");
  grid.innerHTML = "";
  const order = ["planner", "scriptwriter", "sfx_director", "animator", "qa"];
  order.forEach((role) => {
    const cfg = state.team.roles[role];
    const card = document.createElement("div");
    card.className = "card role-card" + (role === "planner" ? " controller" : "");
    card.innerHTML = `
      <div class="role-top">
        <span class="role-name">${role === "planner" ? "👑 " : ""}${esc(state.team.roles_meta[role])}</span>
        <label class="switch"><input type="checkbox" class="role-on" ${cfg.enabled ? "checked" : ""}>
        <span class="slider-ui"></span></label>
      </div>
      <div class="role-desc">${esc(state.team.roles_desc[role])}</div>
      <div class="role-ctrl">
        <select class="input role-model" style="flex:1">
          ${[cfg.model, ...state.models.filter((m) => m !== cfg.model)].map((m) =>
            `<option ${m === cfg.model ? "selected" : ""}>${esc(m)}</option>`).join("")}
        </select>
        <label>temp <input type="number" class="role-temp" step="0.1" min="0" max="2" value="${cfg.temperature}"></label>
      </div>`;
    grid.appendChild(card);
  });
}
$("#btn-save-team").addEventListener("click", async () => {
  const roles = {};
  $$(".role-card").forEach((card, i) => {
    const order = ["planner", "scriptwriter", "sfx_director", "animator", "qa"];
    roles[order[i]] = {
      enabled: card.querySelector(".role-on").checked,
      model: card.querySelector(".role-model").value,
      temperature: parseFloat(card.querySelector(".role-temp").value || "0.7"),
    };
  });
  try {
    await api("/api/team", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ollama_host: $("#team-host").value, controller: roles.planner.model, roles }) });
    flash($("#team-save-msg"), "AI team saved ✓");
    await loadTeam();
  } catch (e) { flash($("#team-save-msg"), e.message, false); }
});

/* ---------------- step 3: create ---------------- */
$("#duration").addEventListener("input", (e) => ($("#dur-label").textContent = e.target.value + "s"));
$$(".chip").forEach((c) => c.addEventListener("click", () => {
  $$(".chip").forEach((x) => x.classList.remove("active"));
  c.classList.add("active");
  state.style = c.dataset.style;
}));

function fillCharSelect() {
  const sel = $("#select-char");
  sel.innerHTML = state.characters.map((c) =>
    `<option value="${c.id}" ${c.id === state.activeCharId ? "selected" : ""}>${esc(c.name)}</option>`).join("");
  sel.disabled = !state.characters.length;
  sel.title = state.characters.length ? "" : "Create a character first (step 1)";
  sel.onchange = (e) => { state.activeCharId = e.target.value; };
}

function markStepsDone(upTo) {
  $$(".step").forEach((b) => b.classList.toggle("done", Number(b.dataset.step) < upTo));
}

$("#btn-plan").addEventListener("click", async () => {
  const idea = $("#idea").value.trim();
  if (!idea) { flash($("#plan-msg"), "Write your video idea first.", false); return; }
  const btn = $("#btn-plan");
  btn.disabled = true;
  btn.textContent = "🧠 AI team is planning…";
  flash($("#plan-msg"), "Controller AI is delegating tasks — this can take a minute.");
  try {
    const plan = await api("/api/plan", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idea, target_duration: parseInt($("#duration").value, 10),
        style: state.style, character_id: state.activeCharId }) });
    state.plan = plan;
    renderActivity(plan.activity || []);
    renderPlanEditor();
    markStepsDone(4);
  } catch (e) {
    flash($("#plan-msg"), e.message, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "🧠 Run AI team — plan my video";
  }
});

function renderActivity(activity) {
  const box = $("#team-activity");
  const log = $("#activity-log");
  box.classList.remove("hidden");
  log.innerHTML = "";
  const icons = { ai: "🧠", fallback: "📋", skipped: "⏭️", error: "⚠️" };
  activity.forEach((a) => {
    const d = document.createElement("div");
    d.className = "activity-item";
    d.innerHTML = `<span class="a-icon">${icons[a.status] || "•"}</span>
      <span>${esc(a.label)}: <span class="a-model">${esc(a.model || "—")}</span>
      <span class="a-note">${esc(a.note)}</span></span>`;
    log.appendChild(d);
  });
}

function sfxOptions(sel) {
  return ["none", ...state.sfx.map((s) => s.name)].map((n) =>
    `<option ${n === sel ? "selected" : ""}>${esc(n)}</option>`).join("");
}
function enumOptions(list, sel) {
  return list.map((n) => `<option ${n === sel ? "selected" : ""}>${esc(n)}</option>`).join("");
}

function renderPlanEditor() {
  const pe = $("#plan-editor");
  pe.classList.remove("hidden");
  $("#plan-title").value = state.plan.title || "";
  $("#plan-total").textContent = `⏱ ${state.plan.total_duration}s · ${state.plan.scenes.length} scenes`;
  $("#scene-cards").innerHTML = "";
  state.plan.scenes.forEach((s, i) => $("#scene-cards").appendChild(sceneCard(s, i)));
}

function sceneCard(s, i) {
  const card = document.createElement("div");
  card.className = "card scene-card";
  card.innerHTML = `
    <div class="scene-head">
      <span class="scene-idx">${i + 1}</span>
      <input class="scene-hook" value="${esc(s.hook)}">
      <button class="btn btn-ghost btn-sm scene-del">✕</button>
    </div>
    <textarea class="input scene-script" rows="3" placeholder="What the character says…">${esc(s.script)}</textarea>
    <div class="scene-selects">
      <div class="mini"><label>SFX</label>
        <select class="input scene-sfx">${sfxOptions(s.sfx)}</select></div>
      <div class="mini"><label>SFX at (s)</label>
        <input class="input scene-sfxtime" type="number" step="0.1" min="0" max="3" value="${s.sfx_time}"></div>
      <div class="mini"><label>Character animation</label>
        <select class="input scene-anim">${enumOptions(["pop-in", "slide-left", "slide-right", "zoom", "bounce", "fade-in"], s.animation)}</select></div>
      <div class="mini"><label>Out transition</label>
        <select class="input scene-trans">${enumOptions(["fade", "slide", "zoom", "wipe", "cut"], s.transition)}</select></div>
      <div class="mini"><label>Background</label>
        <select class="input scene-bg">${enumOptions(["gradient-violet", "gradient-blue", "gradient-sunset", "gradient-forest", "solid-dark", "solid-black", "solid-navy", "pattern-dots", "pattern-grid"], s.background)}</select></div>
      <div class="mini"><label>Duration (s)</label>
        <input class="input scene-dur" type="number" step="0.5" min="2.5" max="12" value="${s.duration}"></div>
    </div>`;
  card.querySelector(".scene-del").addEventListener("click", () => {
    if (state.plan.scenes.length <= 1) { alert("Keep at least one scene."); return; }
    state.plan.scenes.splice(i, 1);
    $("#plan-total").textContent = `⏱ ~${state.plan.scenes.reduce((a, b) => a + parseFloat(b.duration || 4), 0).toFixed(1)}s · ${state.plan.scenes.length} scenes`;
    renderPlanEditor();
  });
  return card;
}

$("#btn-add-scene").addEventListener("click", () => {
  state.plan.scenes.push({
    hook: "New scene", script: "Add what the character should say here.",
    sfx: "ding", sfx_time: 0.3, animation: "pop-in", transition: "fade",
    background: "gradient-blue", duration: 5,
  });
  renderPlanEditor();
});

$("#btn-save-plan").addEventListener("click", async () => {
  const scenes = $$("#scene-cards .scene-card").map((card) => ({
    hook: card.querySelector(".scene-hook").value,
    script: card.querySelector(".scene-script").value,
    sfx: card.querySelector(".scene-sfx").value,
    sfx_time: parseFloat(card.querySelector(".scene-sfxtime").value || "0.3"),
    animation: card.querySelector(".scene-anim").value,
    transition: card.querySelector(".scene-trans").value,
    background: card.querySelector(".scene-bg").value,
    duration: parseFloat(card.querySelector(".scene-dur").value || "4"),
  }));
  try {
    const r = await api(`/api/plans/${state.plan.id}`, { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: $("#plan-title").value, logline: state.plan.logline, scenes }) });
    state.plan.total_duration = r.total_duration;
    flash($("#plan-save-msg"), `Plan saved ✓ (${r.total_duration}s) — ready to render in step 4.`);
  } catch (e) { flash($("#plan-save-msg"), e.message, false); }
});

/* ---------------- step 4: render ---------------- */
function prepareRenderStep() {
  const sel = $("#voice-select");
  const char = state.characters.find((c) => c.id === state.activeCharId);
  const def = char && char.voice_id ? char.voice_id : "";
  sel.innerHTML = `<option value="">Use character default${def ? " (" + (state.voices.find(v => v.id === def)?.name || "none") + ")" : " (no clone voice — TTS engine auto-pick)"}</option>` +
    state.voices.map((v) => `<option value="${v.id}">${esc(v.name)} (clone)</option>`).join("");
  if (!state.plan) {
    $("#render-msg").textContent = "Run the planner in step 3 first.";
  } else {
    $("#render-msg").textContent = `Plan: "${state.plan.title}" · ${state.plan.scenes.length} scenes · ~${state.plan.total_duration}s`;
  }
  const st = state.status;
  const hasVoice = st && st.tts && (st.tts.kokoro || st.tts.xtts || st.tts.gtts);
  $("#tts-note").classList.toggle("hidden", !!hasVoice);
}

$("#btn-render").addEventListener("click", async () => {
  if (!state.plan) { alert("Run the planner in step 3 first."); return; }
  const [w, h] = $("#res-select").value.split("x").map(Number);
  const btn = $("#btn-render");
  btn.disabled = true;
  try {
    const r = await api("/api/render", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_id: state.plan.id,
        voice_id: $("#voice-select").value || null,
        kokoro_voice: $("#kokoro-select").value, width: w, height: h }) });
    state.job = r.job_id;
    $("#render-progress").classList.remove("hidden");
    $("#render-result").classList.add("hidden");
    pollJob();
  } catch (e) {
    alert(e.message);
    btn.disabled = false;
  }
});

function pollJob() {
  clearInterval(state.jobTimer);
  let fails = 0;
  state.jobTimer = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${state.job}`);
      fails = 0;
      $("#progress-bar").style.width = j.progress + "%";
      $("#progress-pct").textContent = j.progress + "%";
      $("#progress-stage").textContent = j.stage || "Working…";
      if (j.status === "completed") {
        clearInterval(state.jobTimer);
        $("#btn-render").disabled = false;
        $("#render-result").classList.remove("hidden");
        $("#result-video").src = j.result.download_url;
        $("#dl-mp4").href = j.result.download_url;
        $("#dl-srt").href = j.result.srt_url;
        markStepsDone(5);
      } else if (j.status === "failed") {
        clearInterval(state.jobTimer);
        $("#btn-render").disabled = false;
        $("#progress-stage").textContent = "Render failed";
        alert("Render failed: " + (j.error || "unknown error"));
      }
    } catch (e) {
      fails += 1;
      // server may have restarted (jobs are in-memory) — stop instead of polling forever
      if (fails >= 10) {
        clearInterval(state.jobTimer);
        $("#btn-render").disabled = false;
        $("#progress-stage").textContent = "Lost contact with the render job — reload the page and generate again.";
        $("#progress-pct").textContent = "";
      }
    }
  }, 1200);
}
$("#btn-again").addEventListener("click", () => gotoStep(3));

/* ---------------- boot (robust: one failed call must not kill the UI) ---------------- */
async function tryLoad(fn, label) {
  try { await fn(); return true; }
  catch (e) { console.warn("boot:", label, "failed:", e.message); return false; }
}
(async function boot() {
  const first = await Promise.all([
    tryLoad(refreshStatus, "status"),
    tryLoad(loadCharacters, "characters"),
    tryLoad(loadVoices, "voices"),
    tryLoad(loadTeam, "team"),
    tryLoad(loadModels, "models"),
  ]);
  await tryLoad(async () => { state.sfx = await api("/api/sfx"); }, "sfx");
  fillCharSelect();
  if (first.some((ok) => !ok)) {
    // retry the failed ones once after a short delay (server may still be warming up)
    setTimeout(async () => {
      await tryLoad(refreshStatus, "status(retry)");
      await tryLoad(loadCharacters, "characters(retry)");
      await tryLoad(loadVoices, "voices(retry)");
      await tryLoad(loadTeam, "team(retry)");
      await tryLoad(loadModels, "models(retry)");
      await tryLoad(async () => { state.sfx = await api("/api/sfx"); }, "sfx(retry)");
      fillCharSelect();
    }, 2500);
  }
  setInterval(refreshStatus, 15000);
})();
