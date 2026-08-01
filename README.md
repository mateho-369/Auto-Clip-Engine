# ⚡ Global Highlights: Local Auto-Clip Engine (v3.0)

Global Highlights is an advanced, workstation-grade local clipping engine that transforms long-form landscape videos (16:9) into viral, professional 9:16 vertical shorts optimized for YouTube Shorts, Instagram Reels, and Facebook Reels.

This is **v3.0**, upgraded with an automated single-command setup, full CI verification pipelines, dynamic face-tracking, local speech-to-text, and local premium narration.

---

## 🚀 One-Command Local Setup (Linux / macOS)

We have built a single, automated setup script that verifies your local system dependencies, installs Ollama model weight sets, fetches pre-trained face-tracking and TTS files, and prepares an isolated virtual environment.

To setup and run the application instantly, execute:
```bash
./setup.sh
```

*(This command verifies ffmpeg, pulls the Llama 3.2 model, downloads MediaPipe/Kokoro files, creates `venv`, installs all pinned packages, and prints the startup instructions!)*

---

## 💻 Manual Step-by-Step Installation (Fallback / Windows)

### 1. Install System Dependencies
Make sure you have `ffmpeg` and `ffprobe` on your system.
```bash
# On Debian/Ubuntu
sudo apt update && sudo apt install -y ffmpeg

# On macOS (using Homebrew)
brew install ffmpeg

# On Windows
scoop install ffmpeg  # or using chocolatey: choco install ffmpeg
```

### 2. Set Up Local Ollama (Required for Semantic AI Pass)
1. Download and run Ollama from [ollama.com](https://ollama.com).
2. Pull the default 3B model in your terminal:
   ```bash
   ollama pull llama3.2:3b
   ```

### 3. Download Local Model Files
To use local Kokoro TTS and MediaPipe face tracking:
```bash
# Download MediaPipe face detector model
curl -L -o blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# Download Kokoro ONNX model and voices
curl -L -o kokoro-v0_19.onnx https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/kokoro-v0_19.onnx
curl -L -o voices.bin https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/voices.bin
```

### 4. Setup Python Environment & Packages
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5. Start the Web App
```bash
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to **`http://localhost:8000`** in your browser.

---

## 🧪 Automated Testing & CI Verification

v3.0 is reinforced with a robust `pytest` test suite, verified by GitHub Actions.

### Running Tests Locally
To execute the automated test suite on your local workstation, run:
```bash
PYTHONPATH=. ./venv/bin/pytest -v
```

This test suite covers:
- **FastAPI Endpoints:** Health checks, response shapes, and state transitions using `TestClient`.
- **Numpy Scalar Guard:** Verifies all public scoring values return native Python floats (never numpy types) to prevent API serialization crashes.
- **Cascade Classifier Assertions:** Validates that frontal face cascades load correctly without silent failures.
- **Graceful Degradation:** Mocks Ollama and Kokoro failure modes to prove the pipeline continues running correctly even when offline.
- **Integration Pipeline:** Generates a 2-second synthetic video with audio, performs vertical face crop and audio merge, and asserts the final vertical video has non-zero-duration audio.

---

## 🛠️ Workstation-Grade System Stack

### 1. Semantic Highlight Detection (Local LLM Re-Ranking)
- **Our Selection:** Multi-Modal Peak Signal Analysis + **Ollama Llama 3.2 (3B)**.
- **How it works:** Candidates are generated via heuristic signal peaks, then the top performing segments are evaluated semantically by Llama 3.2 running on your local machine to score humor and story structure. Blended score combines physical audio peaks (40%) and semantic wisdom (60%).
- **Graceful Fallback:** If Ollama is offline or uninstalled, the app automatically switches to local heuristic evaluation without crashing.

### 2. High-Accuracy Transcription (Local Offline Whisper)
- **Our Selection:** `faster-whisper` (utilizing a local `tiny` or `base` model on CPU/CUDA).
- **Why:** 100% offline, zero cloud rate-limits, and near-perfect segment/word alignment. Automatically prefers GPU (CUDA) if available.

### 3. Premium Voiceover (Local Kokoro-82M TTS)
- **Our Selection:** `kokoro-onnx` (small, Apache-2.0, CPU-friendly) using premium model voices (`af_bella`, `am_adam`, `bf_emma`, `bm_george`).
- **Why:** Incredible human-like tone, 100% private. Ducking engine dims background video audio to 25% while the narrator speaks.

### 4. Advanced Crop Tracking (MediaPipe Local Face Detection)
- **Our Selection:** MediaPipe Tasks FaceDetector (`blaze_face_short_range.tflite`) + Exponential Moving Average (EMA) Motion Smoothing.
- **Why:** High precision face tracking. If face tracking is momentarily obscured, the system falls back to Haar Cascades, and drifts back gracefully toward center (Cinematic Glide) rather than jumping.

---

## 🛡️ Monetization & Compliance (Section 3 Guardrails)

YouTube and Facebook enforce strict rules against raw compilation channels.
This engine features an **in-app Compliance Checklist** that remains locked until the creator verifies:
1. **Ownership/Licensing:** User represents they own the raw footage or hold a clear commercial license.
2. **Transformative Value:** The engine applies animated dynamic captions, smart vertical crops, and customizable AI voiceovers, ensuring editorial/format transformation that complies with monetization policies.
3. **No Unlicensed Scraping:** Built-in workflows are gated, preventing illegal re-hosting or platform scraper violations.
