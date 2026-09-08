/**
 * AI Content Creator — Professional Studio Frontend
 * 
 * Supports Explain Mode (presenter + poses + supporting images), Standard Mode,
 * Manual & Auto operating modes, Undo/Redo, Live Canvas preview, and Subtitle templates.
 */

const state = {
  currentStep: 1,
  mode: "explain",          // "explain" | "standard"
  operatingMode: "auto",     // "auto" | "manual"
  contentGoal: "explain_one",
  sfxEnabled: true,
  characters: [],
  selectedCharId: null,
  voices: [],
  selectedVoiceId: null,
  plan: null,
  selectedSceneIdx: 0,
  subtitleTemplate: "classic_yellow",
  subtitleTemplates: {},
  undoStack: [],
  redoStack: [],
};

// Undo / Redo State Helpers
function pushState() {
  if (!state.plan) return;
  state.undoStack.push(JSON.stringify(state.plan));
  if (state.undoStack.length > 30) state.undoStack.shift();
  state.redoStack = [];
  updateUndoRedoButtons();
}

function undo() {
  if (state.undoStack.length === 0) return;
  state.redoStack.push(JSON.stringify(state.plan));
  const prev = state.undoStack.pop();
  state.plan = JSON.parse(prev);
  renderStudioEditor();
  updateUndoRedoButtons();
}

function redo() {
  if (state.redoStack.length === 0) return;
  state.undoStack.push(JSON.stringify(state.plan));
  const next = state.redoStack.pop();
  state.plan = JSON.parse(next);
  renderStudioEditor();
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  const btnUndo = document.getElementById("btn-undo");
  const btnRedo = document.getElementById("btn-redo");
  if (btnUndo) btnUndo.disabled = state.undoStack.length === 0;
  if (btnRedo) btnRedo.disabled = state.redoStack.length === 0;
}

// DOM Loaded Entry
document.addEventListener("DOMContentLoaded", () => {
  initUI();
  checkStatus();
  loadCharacters();
  loadVoices();
  loadTeamConfig();
  loadSubtitleTemplates();
  bindEvents();
  setupKeyboardShortcuts();
});

// UI Event Binding
function bindEvents() {
  // Navigation Steps
  document.querySelectorAll(".step").forEach(btn => {
    btn.addEventListener("click", () => {
      const step = parseInt(btn.dataset.step);
      setStep(step);
    });
  });

  // Mode Toggles
  document.querySelectorAll("#mode-toggle .seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#mode-toggle .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      if (state.plan) {
        pushState();
        state.plan.mode = state.mode;
        renderStudioEditor();
      }
    });
  });

  document.querySelectorAll("#op-mode-toggle .seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#op-mode-toggle .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.operatingMode = btn.dataset.op;
      const promptCard = document.getElementById("auto-prompt-card");
      if (promptCard) promptCard.style.display = state.operatingMode === "auto" ? "block" : "none";
    });
  });

  // Goal Picker
  const selGoal = document.getElementById("select-goal");
  if (selGoal) {
    selGoal.addEventListener("change", (e) => {
      state.contentGoal = e.target.value;
      if (state.plan) {
        pushState();
        state.plan.content_goal = state.contentGoal;
      }
    });
  }

  // SFX Toggle
  const btnSfx = document.getElementById("btn-sfx-toggle");
  if (btnSfx) {
    btnSfx.addEventListener("click", () => {
      state.sfxEnabled = !state.sfxEnabled;
      btnSfx.classList.toggle("active", state.sfxEnabled);
      btnSfx.innerText = state.sfxEnabled ? "🔊 SFX ON" : "🔇 SFX OFF";
      if (state.plan) {
        pushState();
        state.plan.sfx_enabled = state.sfxEnabled;
      }
    });
  }

  // Undo / Redo
  document.getElementById("btn-undo")?.addEventListener("click", undo);
  document.getElementById("btn-redo")?.addEventListener("click", redo);

  // Run AI Team / Plan
  document.getElementById("btn-plan")?.addEventListener("click", runPlan);

  // Character creation
  document.getElementById("btn-create-char")?.addEventListener("click", createCharacter);

  // Property Inputs
  bindPropertyInputs();

  // Tab Buttons
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  // Add & Delete Scene
  document.getElementById("btn-add-scene")?.addEventListener("click", addScene);
  document.getElementById("btn-del-scene")?.addEventListener("click", deleteSelectedScene);
  document.getElementById("btn-save-plan")?.addEventListener("click", savePlan);

  // Image Fetch / Pose Gen
  document.getElementById("btn-fetch-img")?.addEventListener("click", fetchSceneImage);
  document.getElementById("btn-gen-pose")?.addEventListener("click", generateCurrentPose);

  // Render Video
  document.getElementById("btn-render")?.addEventListener("click", startRender);
}

function setStep(step) {
  state.currentStep = step;
  document.querySelectorAll(".step").forEach(b => {
    b.classList.toggle("active", parseInt(b.dataset.step) === step);
  });
  document.querySelectorAll(".panel").forEach((p, idx) => {
    p.classList.toggle("hidden", idx + 1 !== step);
  });
  if (step === 3) {
    renderStudioEditor();
  }
}

// Bind Inspector Property Controls
function bindPropertyInputs() {
  const updateProp = (field, val) => {
    if (!state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
    pushState();
    state.plan.scenes[state.selectedSceneIdx][field] = val;
    renderStudioEditor();
  };

  document.getElementById("prop-hook")?.addEventListener("input", (e) => updateProp("hook", e.target.value));
  document.getElementById("prop-script")?.addEventListener("input", (e) => updateProp("script", e.target.value));
  document.getElementById("prop-pose")?.addEventListener("change", (e) => updateProp("pose", e.target.value));
  document.getElementById("prop-duration")?.addEventListener("change", (e) => updateProp("duration", parseFloat(e.target.value) || 4.0));
  document.getElementById("prop-transition")?.addEventListener("change", (e) => updateProp("transition", e.target.value));
  document.getElementById("prop-sfx")?.addEventListener("change", (e) => updateProp("sfx", e.target.value));
  document.getElementById("prop-sfx-time")?.addEventListener("change", (e) => updateProp("sfx_time", parseFloat(e.target.value) || 0.3));
  document.getElementById("prop-bg")?.addEventListener("change", (e) => updateProp("background", e.target.value));

  // Image controls
  document.getElementById("prop-img-needed")?.addEventListener("change", (e) => {
    if (!state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
    pushState();
    const sc = state.plan.scenes[state.selectedSceneIdx];
    if (!sc.image) sc.image = {};
    sc.image.needed = e.target.checked;
    renderStudioEditor();
  });

  document.querySelectorAll("#img-source-toggle .seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (!state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
      document.querySelectorAll("#img-source-toggle .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      pushState();
      const sc = state.plan.scenes[state.selectedSceneIdx];
      if (!sc.image) sc.image = {};
      sc.image.source = btn.dataset.src;
      renderStudioEditor();
    });
  });

  document.getElementById("prop-img-pos")?.addEventListener("change", (e) => {
    if (!state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
    pushState();
    const sc = state.plan.scenes[state.selectedSceneIdx];
    if (!sc.image) sc.image = {};
    sc.image.position = e.target.value;
    renderStudioEditor();
  });
}

// Server API Calls & Setup
async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const pOllama = document.getElementById("pill-ollama");
    const pTts = document.getElementById("pill-tts");
    if (pOllama) {
      pOllama.className = "pill " + (data.ollama.online ? "pill-local" : "pill-off");
      pOllama.innerText = data.ollama.online ? "Ollama Online" : "Ollama Offline";
    }
    if (pTts) {
      pTts.className = "pill " + (data.tts.available ? "pill-local" : "pill-off");
      pTts.innerText = data.tts.available ? "TTS Ready" : "TTS Offline";
    }
  } catch (e) {
    console.error("Status check failed", e);
  }
}

async function loadCharacters() {
  try {
    const res = await fetch("/api/characters");
    state.characters = await res.json();
    renderCharactersList();
  } catch (e) {
    console.error(e);
  }
}

function renderCharactersList() {
  const container = document.getElementById("char-list");
  const selChar = document.getElementById("select-char");
  if (!container) return;

  if (state.characters.length === 0) {
    container.style.display = "none";
    return;
  }
  container.style.display = "grid";
  container.innerHTML = "";
  if (selChar) selChar.innerHTML = "";

  state.characters.forEach((c, idx) => {
    if (idx === 0 && !state.selectedCharId) state.selectedCharId = c.id;

    // Character Card
    const card = document.createElement("div");
    card.className = "char-card " + (c.id === state.selectedCharId ? "selected" : "");
    card.innerHTML = `
      <img src="${c.assets.avatar}" class="char-avatar" alt="${c.name}">
      <div style="font-weight:700; color:#fff;">${c.name}</div>
      <div style="font-size:11px; color:var(--text-muted);">${c.photos} photos</div>
      <button class="btn btn-ghost btn-xs margin-top-xs btn-use-char" data-id="${c.id}">Use Character</button>
    `;
    card.querySelector(".btn-use-char")?.addEventListener("click", () => {
      state.selectedCharId = c.id;
      renderCharactersList();
      if (state.plan) loadAssetBrowser();
    });
    container.appendChild(card);

    if (selChar) {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.innerText = c.name;
      opt.selected = c.id === state.selectedCharId;
      selChar.appendChild(opt);
    }
  });
}

async function createCharacter() {
  const nameInput = document.getElementById("char-name");
  const fileInput = document.getElementById("file-create");
  const msg = document.getElementById("char-msg");
  if (!fileInput?.files[0]) {
    if (msg) msg.innerText = "Please select a photo first.";
    return;
  }
  const fd = new FormData();
  fd.append("name", nameInput?.value || "My Character");
  fd.append("file", fileInput.files[0]);

  if (msg) msg.innerText = "Creating character...";
  try {
    const res = await fetch("/api/characters/create", { method: "POST", body: fd });
    const data = await res.json();
    if (res.ok) {
      if (msg) msg.innerText = "Character created!";
      loadCharacters();
    } else {
      if (msg) msg.innerText = data.detail || "Error creating character.";
    }
  } catch (e) {
    if (msg) msg.innerText = e.message;
  }
}

async function loadVoices() {
  try {
    const res = await fetch("/api/voices");
    state.voices = await res.json();
    const sel = document.getElementById("voice-select");
    if (sel) {
      sel.innerHTML = '<option value="">Default Khmer / Kokoro Voice</option>';
      state.voices.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.innerText = v.name;
        sel.appendChild(opt);
      });
    }
  } catch (e) {
    console.error(e);
  }
}

async function loadTeamConfig() {
  try {
    const res = await fetch("/api/team");
    const data = await res.json();
    const container = document.getElementById("role-cards");
    if (!container) return;
    container.innerHTML = "";
    Object.keys(data.roles_meta || {}).forEach(r => {
      const card = document.createElement("div");
      card.className = "role-card";
      card.innerHTML = `
        <div style="font-weight:700; color:#fff;">${data.roles_meta[r]}</div>
        <div style="font-size:11px; color:var(--text-muted);">${data.roles_desc[r] || ""}</div>
      `;
      container.appendChild(card);
    });
  } catch (e) {
    console.error(e);
  }
}

async function loadSubtitleTemplates() {
  try {
    const res = await fetch("/api/subtitle-templates");
    state.subtitleTemplates = await res.json();
    renderSubtitleTemplateCards();
  } catch (e) {
    console.error(e);
  }
}

function renderSubtitleTemplateCards() {
  const container = document.getElementById("subtitle-template-cards");
  if (!container) return;
  container.innerHTML = "";

  Object.keys(state.subtitleTemplates).forEach(k => {
    const tmpl = state.subtitleTemplates[k];
    const card = document.createElement("div");
    card.className = "tmpl-card " + (k === state.subtitleTemplate ? "selected" : "");
    card.innerText = tmpl.name;
    card.addEventListener("click", () => {
      state.subtitleTemplate = k;
      renderSubtitleTemplateCards();
      renderCanvasPreview();
    });
    container.appendChild(card);
  });
}

// Run AI Team Pipeline
async function runPlan() {
  const ideaInput = document.getElementById("idea");
  const durInput = document.getElementById("duration");
  const msg = document.getElementById("plan-msg");

  const idea = ideaInput?.value.trim();
  if (!idea && state.operatingMode === "auto") {
    if (msg) msg.innerText = "Please enter a video prompt / idea first.";
    return;
  }

  if (!state.selectedCharId) {
    if (msg) msg.innerText = "Please select or create a character first.";
    return;
  }

  if (msg) msg.innerText = "🧠 Running AI Team...";
  const actBox = document.getElementById("team-activity");
  if (actBox) actBox.classList.remove("hidden");

  try {
    const res = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        idea: idea || "Presentation concept",
        target_duration: parseInt(durInput?.value || 25),
        character_id: state.selectedCharId,
        mode: state.mode,
        operating_mode: state.operatingMode,
        content_goal: state.contentGoal,
        sfx_enabled: state.sfxEnabled,
      }),
    });

    const data = await res.json();
    if (res.ok) {
      state.plan = data;
      state.selectedSceneIdx = 0;
      if (msg) msg.innerText = "Plan generated!";
      setStep(3);
      renderStudioEditor();
    } else {
      if (msg) msg.innerText = data.detail || "Error generating plan.";
    }
  } catch (e) {
    if (msg) msg.innerText = e.message;
  }
}

// Render Studio Editor Panels
function renderStudioEditor() {
  if (!state.plan || !state.plan.scenes) return;

  renderSceneListRail();
  renderInspectorProperties();
  renderTimeline();
  renderCanvasPreview();
  loadAssetBrowser();
}

function renderSceneListRail() {
  const rail = document.getElementById("scene-list-rail");
  if (!rail) return;
  rail.innerHTML = "";

  state.plan.scenes.forEach((sc, idx) => {
    const card = document.createElement("div");
    card.className = "scene-item-card " + (idx === state.selectedSceneIdx ? "selected" : "");
    card.innerHTML = `
      <div class="scene-item-head">
        <span>Scene ${idx + 1}</span>
        <span>${sc.duration}s</span>
      </div>
      <div class="scene-item-script">${sc.hook || sc.script || "Empty scene"}</div>
    `;
    card.addEventListener("click", () => {
      state.selectedSceneIdx = idx;
      renderStudioEditor();
    });
    rail.appendChild(card);
  });
}

function renderInspectorProperties() {
  if (!state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
  const sc = state.plan.scenes[state.selectedSceneIdx];

  const pHook = document.getElementById("prop-hook");
  const pScript = document.getElementById("prop-script");
  const pPose = document.getElementById("prop-pose");
  const pDur = document.getElementById("prop-duration");
  const pTrans = document.getElementById("prop-transition");
  const pSfx = document.getElementById("prop-sfx");
  const pSfxTime = document.getElementById("prop-sfx-time");
  const pBg = document.getElementById("prop-bg");

  if (pHook) pHook.value = sc.hook || "";
  if (pScript) pScript.value = sc.script || "";
  if (pDur) pDur.value = sc.duration || 4.0;
  if (pTrans) pTrans.value = sc.transition || "fade";
  if (pSfxTime) pSfxTime.value = sc.sfx_time || 0.3;
  if (pBg) pBg.value = sc.background || "gradient-violet";

  // Populate Poses dropdown
  if (pPose) {
    pPose.innerHTML = "";
    const standardPoses = ["idle", "point_left", "point_right", "point_up", "explain", "think", "wave", "laugh", "sleep", "eat", "sad", "surprised"];
    standardPoses.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.innerText = p.replace("_", " ").toUpperCase();
      opt.selected = sc.pose === p;
      pPose.appendChild(opt);
    });
  }

  // Populate SFX dropdown
  if (pSfx) {
    pSfx.innerHTML = '<option value="none">None</option>';
    ["whoosh", "pop", "ding", "click", "riser", "boom", "applause", "sparkle", "typing"].forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.innerText = s.toUpperCase();
      opt.selected = sc.sfx === s;
      pSfx.appendChild(opt);
    });
  }

  // Image controls
  const pImgNeeded = document.getElementById("prop-img-needed");
  const pImgQuery = document.getElementById("prop-img-query");
  const pImgPos = document.getElementById("prop-img-pos");

  const imgInfo = sc.image || {};
  if (pImgNeeded) pImgNeeded.checked = !!imgInfo.needed;
  if (pImgQuery) pImgQuery.value = imgInfo.query_or_prompt || "";
  if (pImgPos) pImgPos.value = imgInfo.position || "top";
}

function renderTimeline() {
  const rail = document.getElementById("timeline-cards-rail");
  const durLabel = document.getElementById("timeline-total-dur");
  if (!rail) return;
  rail.innerHTML = "";

  let totalDur = 0;
  state.plan.scenes.forEach((sc, idx) => {
    totalDur += sc.duration;
    const card = document.createElement("div");
    card.className = "timeline-card " + (idx === state.selectedSceneIdx ? "selected" : "");
    card.innerHTML = `
      <div class="timeline-card-title">Scene ${idx + 1}</div>
      <div class="timeline-card-dur">${sc.duration}s</div>
      <div class="timeline-badges">
        <span class="badge-mini">${sc.pose || "idle"}</span>
        ${sc.sfx !== "none" ? `<span class="badge-mini">🔊 ${sc.sfx}</span>` : ""}
      </div>
    `;
    card.addEventListener("click", () => {
      state.selectedSceneIdx = idx;
      renderStudioEditor();
    });
    rail.appendChild(card);
  });

  if (durLabel) durLabel.innerText = `${totalDur.toFixed(1)}s total`;
}

// Live Canvas Preview Rendering
function renderCanvasPreview() {
  const canvas = document.getElementById("preview-canvas");
  if (!canvas || !state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
  const ctx = canvas.getContext("2d");
  const sc = state.plan.scenes[state.selectedSceneIdx];
  const w = canvas.width;
  const h = canvas.height;

  // Background Gradient
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  if (sc.background === "gradient-blue") {
    grad.addColorStop(0, "#1e3a8a"); grad.addColorStop(1, "#0f172a");
  } else if (sc.background === "gradient-sunset") {
    grad.addColorStop(0, "#c2410c"); grad.addColorStop(1, "#431407");
  } else {
    grad.addColorStop(0, "#582cb0"); grad.addColorStop(1, "#1c0c3c");
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Render supporting image preview if present in Explain Mode
  if (state.mode === "explain" && sc.image && sc.image.needed) {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 4;
    ctx.fillStyle = "rgba(30, 32, 46, 0.85)";

    let boxX = w * 0.1, boxY = h * 0.12, boxW = w * 0.8, boxH = h * 0.36;
    if (sc.image.position === "side") {
      boxX = w * 0.05; boxY = h * 0.16; boxW = w * 0.52; boxH = h * 0.42;
    } else if (sc.image.position === "pip") {
      boxX = w * 0.48; boxY = h * 0.14; boxW = w * 0.46; boxH = h * 0.32;
    }

    ctx.fillRect(boxX, boxY, boxW, boxH);
    ctx.strokeRect(boxX, boxY, boxW, boxH);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 24px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`[ ${sc.image.query_or_prompt || "IMAGE"} ]`, boxX + boxW / 2, boxY + boxH / 2);
  }

  // Draw Character Anchor
  const charX = (state.mode === "explain" && sc.image && sc.image.needed) ? w * 0.72 : w / 2;
  const charY = h * 0.68;
  ctx.fillStyle = "rgba(99, 102, 241, 0.3)";
  ctx.beginPath();
  ctx.arc(charX, charY, 140, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 28px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`👤 Presenter (${sc.pose || "idle"})`, charX, charY);

  // Draw Captions Preview
  renderCanvasCaptions(ctx, w, h, sc.script || "Sample Khmer Caption Text");
}

function renderCanvasCaptions(ctx, w, h, text) {
  const tmplKey = state.subtitleTemplate;
  const y = h * 0.86;

  ctx.font = "bold 32px sans-serif";
  ctx.textAlign = "center";

  if (tmplKey === "box_brand") {
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(w * 0.1, y - 40, w * 0.8, 60);
    ctx.fillStyle = "#000000";
    ctx.fillText(text.substring(0, 30), w / 2, y);
  } else if (tmplKey === "bold_neon") {
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 8;
    ctx.strokeText(text.substring(0, 30), w / 2, y);
    ctx.fillStyle = "#06b6d4";
    ctx.fillText(text.substring(0, 30), w / 2, y);
  } else {
    ctx.fillStyle = "rgba(20,20,24,0.8)";
    ctx.fillRect(w * 0.1, y - 40, w * 0.8, 60);
    ctx.fillStyle = "#00e6ff";
    ctx.fillText(text.substring(0, 30), w / 2, y);
  }
}

async function loadAssetBrowser() {
  if (!state.selectedCharId) return;
  const pGrid = document.getElementById("asset-pose-grid");
  if (!pGrid) return;

  try {
    const res = await fetch(`/api/characters/${state.selectedCharId}/actions`);
    const data = await res.json();
    pGrid.innerHTML = "";
    Object.keys(data.actions || {}).forEach(pose => {
      const card = document.createElement("div");
      card.className = "asset-card";
      card.innerHTML = `
        <img src="/assets/characters/${state.selectedCharId}/actions/${pose}.png" class="asset-thumb" alt="${pose}">
        <div class="asset-label">${pose.toUpperCase()}</div>
      `;
      card.addEventListener("click", () => {
        if (!state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
        pushState();
        state.plan.scenes[state.selectedSceneIdx].pose = pose;
        renderStudioEditor();
      });
      pGrid.appendChild(card);
    });
  } catch (e) {
    console.error(e);
  }
}

async function fetchSceneImage() {
  if (!state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
  const sc = state.plan.scenes[state.selectedSceneIdx];
  const qInput = document.getElementById("prop-img-query");
  const q = qInput?.value || sc.hook || "concept";

  try {
    const res = await fetch("/api/images/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: sc.image?.source || "web",
        query_or_prompt: q,
        char_id: state.selectedCharId,
        pose: sc.pose || "idle",
      }),
    });
    const data = await res.json();
    if (res.ok) {
      pushState();
      if (!sc.image) sc.image = {};
      sc.image.url = data.url;
      sc.image.needed = true;
      renderStudioEditor();
    }
  } catch (e) {
    console.error(e);
  }
}

async function generateCurrentPose() {
  if (!state.selectedCharId || !state.plan || !state.plan.scenes[state.selectedSceneIdx]) return;
  const sc = state.plan.scenes[state.selectedSceneIdx];
  const pose = sc.pose || "explain";

  const fd = new FormData();
  fd.append("pose", pose);

  try {
    const res = await fetch(`/api/characters/${state.selectedCharId}/actions`, { method: "POST", body: fd });
    if (res.ok) {
      loadAssetBrowser();
      renderCanvasPreview();
    }
  } catch (e) {
    console.error(e);
  }
}

function addScene() {
  if (!state.plan) return;
  pushState();
  state.plan.scenes.push({
    hook: `Scene ${state.plan.scenes.length + 1}`,
    script: "New scene narration text",
    sfx: "none",
    sfx_time: 0.3,
    animation: "pop-in",
    transition: "fade",
    background: "gradient-violet",
    pose: "explain",
    image: { needed: state.mode === "explain", source: "web", query_or_prompt: "concept", position: "top" },
    duration: 4.0,
  });
  state.selectedSceneIdx = state.plan.scenes.length - 1;
  renderStudioEditor();
}

function deleteSelectedScene() {
  if (!state.plan || state.plan.scenes.length <= 1) return;
  pushState();
  state.plan.scenes.splice(state.selectedSceneIdx, 1);
  state.selectedSceneIdx = Math.max(0, state.selectedSceneIdx - 1);
  renderStudioEditor();
}

async function savePlan() {
  if (!state.plan || !state.plan.id) return;
  try {
    await fetch(`/api/plans/${state.plan.id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: state.plan.title,
        logline: state.plan.logline,
        mode: state.plan.mode,
        content_goal: state.plan.content_goal,
        sfx_enabled: state.plan.sfx_enabled,
        scenes: state.plan.scenes,
      }),
    });
  } catch (e) {
    console.error(e);
  }
}

async function startRender() {
  if (!state.plan || !state.plan.id) return;
  const resSelect = document.getElementById("res-select");
  const voiceSelect = document.getElementById("voice-select");
  const kokoroSelect = document.getElementById("kokoro-select");
  const msg = document.getElementById("render-msg");
  const progBox = document.getElementById("render-progress");

  const [w, h] = (resSelect?.value || "720x1280").split("x").map(Number);

  if (msg) msg.innerText = "Queuing render job...";
  if (progBox) progBox.classList.remove("hidden");

  try {
    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plan_id: state.plan.id,
        voice_id: voiceSelect?.value || null,
        kokoro_voice: kokoroSelect?.value || "af_bella",
        subtitle_template: state.subtitleTemplate,
        width: w,
        height: h,
      }),
    });

    const data = await res.json();
    if (res.ok) {
      pollJob(data.job_id);
    } else {
      if (msg) msg.innerText = data.detail || "Error starting render.";
    }
  } catch (e) {
    if (msg) msg.innerText = e.message;
  }
}

function pollJob(jobId) {
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const data = await res.json();
      const pStage = document.getElementById("progress-stage");
      const pBar = document.getElementById("progress-bar");
      const pPct = document.getElementById("progress-pct");

      if (pStage) pStage.innerText = data.stage;
      if (pBar) pBar.style.width = `${data.progress}%`;
      if (pPct) pPct.innerText = `${data.progress}%`;

      if (data.status === "completed") {
        clearInterval(timer);
        showRenderResult(data.result);
      } else if (data.status === "failed") {
        clearInterval(timer);
        alert(`Render failed: ${data.error}`);
      }
    } catch (e) {
      console.error(e);
    }
  }, 1000);
}

function showRenderResult(res) {
  const resultBox = document.getElementById("render-result");
  const video = document.getElementById("result-video");
  const dlMp4 = document.getElementById("dl-mp4");
  const dlSrt = document.getElementById("dl-srt");

  if (resultBox) resultBox.classList.remove("hidden");
  if (video) video.src = res.download_url;
  if (dlMp4) dlMp4.href = res.download_url;
  if (dlSrt) dlSrt.href = res.srt_url;
}

function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.ctrlKey && e.key.toLowerCase() === "z") {
      e.preventDefault();
      undo();
    } else if (e.ctrlKey && e.key.toLowerCase() === "y") {
      e.preventDefault();
      redo();
    }
  });
}
