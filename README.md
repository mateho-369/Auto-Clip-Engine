# ✦ AI Content Creator — Local AI + Bot Video Studio

Turn **one idea + one photo** into a finished vertical video where **your own character**
explains everything — planned by a **team of local Ollama AIs that you personally assign**,
narrated in **your cloned voice**, with animations, transitions, sound effects and karaoke
captions. **100% local, 100% private.**

This repo now contains three local studios:

| Project | Port | What it does |
|---|---|---|
| **Khmer AI Content Studio** (`ai_studio/`) — *new, flagship* | `8000` | Director-led multi-agent **video** pipeline: your Khmer script (or a topic) → scenes → Khmer voice → your RVC timbre → Wan 480p footage → MMAudio ambience → QA → final `.mp4`. **→ [README-STUDIO.md](README-STUDIO.md)** |
| **AI Content Creator** (`ai_creator/`) | `8000` | AI-team planned character videos (this page) — run it on `--port 8002` if the studio above is up |
| **Auto-Clip Engine v3.0** (`src/`) — legacy | `8001` | Clips viral highlights out of long videos (see [below](#-legacy-auto-clip-engine-v30)) |

### 🎬 Khmer AI Content Studio (new)

```bash
./setup-studio.sh              # venv + deps + folders + a readiness report that tells you what to install next
python -m ai_studio --check    # just the report
python -m ai_studio --demo     # http://localhost:8000 — live per-stage stepper, all local, no cloud
```

Two entry modes: **A** = you paste the script and it is treated as inviolable ground truth
(only mechanical scene segmentation); **B** = the Controller writes the Khmer script for a
topic and waits for you to approve it. Full setup for Ollama (`sailor2:8b`), the
sherpa-onnx Khmer voice, RVC voice training, ComfyUI + Wan + MMAudio, and the 8 GB VRAM
safety rules: **[README-STUDIO.md](README-STUDIO.md)**. Works on Machine B too — script,
voice and assembly run there, the GPU stages defer to Machine A. Windows: `setup-studio.ps1`.

---

## 🚀 One-Command Setup (Linux / macOS)

```bash
./setup-creator.sh
```

This verifies ffmpeg, checks Ollama (pulls `llama3.2:3b`), downloads the local
Kokoro-82M TTS weights, creates the venv, and offers two optional extras:

* **rembg** — real background removal for cleaner character cutouts
* **TTS (XTTS v2)** — *voice cloning*: speaks in **your** voice from a ~1-minute recording (~2.5 GB, PyTorch)

Then start the studio:

```bash
./venv/bin/python -m uvicorn ai_creator.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**.

> **Windows:** install ffmpeg (scoop/choco), Ollama from ollama.com, create the venv,
> `pip install -r requirements-dev.txt`, then run the same uvicorn command.

---

## 🎬 How It Works — 4 Steps

### Step 1 · Your Character (face & body memory)
Upload one or more photos of your character (person, drawing, avatar, mascot).
The studio extracts and **remembers** on disk (`characters/<id>/`):

* the **face region** + a perceptual face hash
* the dominant **color palette**
* the **body/full figure** (feathered cutout asset — `rembg` if installed, soft mask otherwise)

Every future video re-uses the exact same character. Add more photos any time —
the UI shows a "looks same / maybe / different" similarity verdict (face hash + palette distance).

### Step 2 · AI Team (you choose who does what)
A deliberately small team of **local Ollama models** — you assign **any installed model to each role**,
and any role can be switched off:

| Role | Job |
|---|---|
| 👑 **Planner / Director (Controller)** | Receives your idea, breaks it into scenes, **delegates tasks** to the other AIs |
| ✍️ **Scriptwriter** | Writes/polishes the narration each scene speaks |
| 🔊 **SFX Director** | Chooses which sound effect plays per scene and exactly when |
| 🎞️ **Animator** | Picks character animations + scene transitions |
| ✅ **QA Reviewer** *(optional)* | Reviews the plan for empty scripts, bad timing, repeated elements |

With Ollama offline (or a role off), built-in deterministic fallbacks take over —
the studio never dead-ends.

### Step 3 · Create
Type your idea, pick length (10–90 s) and style → **Run AI team**. You see the
"AI team at work" feed (which model did what), then a fully **editable scene board**:
script, SFX + timing, animation, transition, background, duration per scene.

### Step 4 · Render
One click composes everything locally: animated character (entry/exit/idle +
**talk-pulse synced to the voice audio**), scene transitions (fade / slide / zoom / wipe),
karaoke captions, the SFX mix, narration, then final H.264 MP4 + SRT.
Formats: 720×1280, 1080×1920, 1280×720.

## 🎙️ Voice (human-like, yours)

1. **Voice cloning (best):** record/upload **30–60 s** of your voice in Step 1.
   With the optional XTTS v2 engine installed, every narration speaks in **your** voice.
2. **Kokoro-82M (local, default):** very human-like, weights included by the setup script,
   pick a voice (Bella, Skye, Nicole, Michael, …).
3. **gTTS (online fallback)** when neither is available.

## 🔊 Sound Effects
Nine SFX are **synthesized on your machine with numpy** (whoosh, pop, ding, click,
riser, boom, applause, sparkle, typing) — zero downloads, works air-gapped.
Drop your own `.wav` files into `sfx_library/` and they appear in the picker.

## 🧪 Tests & CI
```bash
PYTHONPATH=. ./venv/bin/pytest -v
```
123 tests cover all three projects: API endpoints, Ollama JSON extraction, team config,
fallback planning, character memory, animation/transition math, SFX synthesis,
full render integration (real MP4 + SRT produced), and graceful-degradation paths — plus
47 for the Khmer studio (Khmer text handling, the stage DAG, resume/regeneration, a real
7-stage run producing a playable MP4, the VRAM clamps, and the whole HTTP surface).
GitHub Actions runs the whole suite on every push.

## 📁 Layout
```
ai_studio/             # KHMER AI CONTENT STUDIO (flagship — see README-STUDIO.md)
  app.py, api.py       # FastAPI app + the whole REST/WS surface
  config.py            # machine profiles, role→model map, VRAM safety, settings.json
  db.py                # SQLite history: projects, scenes, runs, stage rows, assets, prompts, events
  pipeline/            # spec.py (stage DAG) · scheduler.py (async queue, retry, resume) · stages.py · context.py
  agents/              # controller.py (scene breakdown) · auto_idea.py (Mode B script) · qa.py
  engines/             # tts.py (sherpa-onnx) · rvc.py · video.py (ComfyUI/Wan) · sfx.py (MMAudio) · assembly.py
  previz.py, ambience.py, media.py, vram.py   # CPU fallback renderers, ffmpeg helpers, VRAM guard
  workflows/           # ComfyUI API-format JSON templates (bring your own)
  static/              # the studio UI (dashboard + live stepper + per-scene inspector)
  demo.py              # sample projects + a service-free end-to-end smoke run
ai_creator/            # earlier studio
  app.py               # FastAPI server + API
  planner.py           # controller-AI pipeline (plan -> delegate -> validate)
  team.py              # role -> model assignment
  ollama_client.py     # local Ollama client + robust JSON extraction
  character.py         # face/body memory, palette, similarity, cutout assets
  voice.py             # voice store + Kokoro / XTTS clone / gTTS
  sfx.py               # offline SFX synthesizer + library
  animation.py         # entry/exit/idle/talk-pulse transforms
  transitions.py       # fade/slide/zoom/wipe blending
  renderer.py          # scene composer + audio mixer + final encode
  templates/, static/  # the studio UI
src/                   # legacy Auto-Clip Engine (unchanged)
tests/                 # pytest suite (all three projects)
```

**Privacy:** everything runs on your machine — Ollama LLMs, Kokoro/XTTS TTS,
numpy SFX, OpenCV/moviepy rendering. Nothing is uploaded anywhere
(gTTS is only used if you enable it and it needs the internet).

---

# ⚡ Legacy: Auto-Clip Engine v3.0

The original **Global Highlights** clipping engine: long 16:9 videos → viral 9:16
shorts with face tracking, local Whisper transcription, LLM re-ranking, Kokoro
narration and animated captions.

## Run it
```bash
./setup.sh    # one-command setup (ffmpeg check, Ollama model, MediaPipe/Kokoro weights, venv)
./venv/bin/python -m uvicorn src.app:app --host 0.0.0.0 --port 8001 --reload
```

## Manual installation (Windows / fallback)
1. **ffmpeg**: `scoop install ffmpeg` (Windows) / `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Debian)
2. **Ollama**: install from [ollama.com](https://ollama.com), then `ollama pull llama3.2:3b`
3. **Local model files**:
   ```bash
   curl -L -o blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
   curl -L -o kokoro-v0_19.onnx https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/kokoro-v0_19.onnx
   curl -L -o voices.bin https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/voices.bin
   ```
4. **Python env**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows   (Linux/macOS: source venv/bin/activate)
   pip install -r requirements.txt
   ```

## Feature stack
* **Semantic highlight detection** — multi-modal peak signal analysis + Ollama Llama 3.2 re-ranking (40/60 blend), graceful heuristic fallback
* **Local transcription** — `faster-whisper` (tiny/base, GPU auto-fallback to CPU)
* **Premium voiceover** — Kokoro-82M ONNX, audio ducking to 25% under narration
* **Face tracking** — MediaPipe `blaze_face_short_range` + EMA smoothing, Haar-cascade fallback with cinematic center-drift

## Compliance guardrails
The app includes an in-app checklist: ownership/licensing declaration, transformative
value (animated captions, smart crops, AI voiceover), no unlicensed scraping.
