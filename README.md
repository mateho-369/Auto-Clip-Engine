# ⚡ Global Highlights: Local Auto-Clip Engine (v2.0)

Global Highlights is an advanced, workstation-grade local clipping engine that transforms long-form landscape videos (16:9) into viral, professional 9:16 vertical shorts optimized for YouTube Shorts, Instagram Reels, and Facebook Reels.

This is **v2.0**, upgraded for private, high-performance offline environments with local transcription, deep semantic AI analysis, advanced face tracking, and premium voice synthesis.

---

## 🛠️ Workstation-Grade System Stack

### 1. Semantic Highlight Detection (Local LLM Re-Ranking)
- **Our Selection:** Multi-Modal Peak Signal Analysis + **Ollama Llama 3.2 (3B)** Semantic Evaluation.
- **How it works:** Candidates are generated via heuristic signal peaks, then the top performing segments are evaluated semantically by Llama 3.2 running on your local machine to score humor, insights, and structural hooks. Blended score combines physical audio peaks (40%) and semantic wisdom (60%).
- **Graceful Fallback:** If Ollama is offline or uninstalled, the app automatically switches to high-fidelity local heuristic evaluation without crashing.

### 2. High-Accuracy Transcription (Local Offline Whisper)
- **Our Selection:** `faster-whisper` (utilizing a local `tiny` or `base` CTranslate2 model on CPU).
- **Why:** 100% offline, zero cloud rate-limits, and near-perfect segment/word alignment. Degrades gracefully to Google SpeechRecognition if model files are corrupted or unavailable.

### 3. Premium Voiceover (Local Kokoro-82M TTS)
- **Our Selection:** `kokoro-onnx` (small, Apache-2.0, CPU-friendly) using `blaze_face` or premium model voices (`af_bella`, `am_adam`, `bf_emma`, `bm_george`).
- **Why:** Incredible human-like tone, 100% private. Ducking engine dims background video audio to 25% while the narrator speaks. Fallback defaults to `gTTS`.

### 4. Advanced Crop Tracking (MediaPipe Local Face Detection)
- **Our Selection:** MediaPipe Tasks FaceDetector (`blaze_face_short_range.tflite`) + Exponential Moving Average (EMA) Motion Smoothing.
- **Why:** High precision face tracking. If face tracking is momentarily obscured, the system falls back to Haar Cascades, and drifts back gracefully toward center (Cinematic Glide) rather than jumping.

---

## ⚙️ How the Blended Highlight Ranking Works

The virality index is calculated dynamically for 20-second sliding windows:

$$\text{Heuristic Score} = 0.35 \times A_e + 0.40 \times L_h + 0.15 \times V_m + 0.10 \times H_s$$

If Ollama is active, the final rating is computed as:

$$\text{Final Blended Score} = 0.40 \times \text{Heuristic Score} + 0.60 \times \text{Ollama Semantic Score}$$

- **Ollama Semantic Score ($S_{ollama}$):** Evaluation of humor, irony, insight, and storytelling flow.

---

## 🚀 Step-by-Step Local Setup & Running Guide

### 1. Install System Dependencies
Make sure you have `ffmpeg` and `ffprobe` on your system.
```bash
# On Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# On macOS (using Homebrew)
brew install ffmpeg
```

### 2. Set Up Local Ollama (Required for Semantic AI Pass)
1. Download Ollama from [ollama.com](https://ollama.com).
2. Pull the default 3B model in your terminal:
   ```bash
   ollama pull llama3.2:3b
   ```
3. Make sure the Ollama server is running in the background (default: `http://localhost:11434`).

### 3. Download Offline Model Files (Optional, for Premium Offline TTS)
To use local Kokoro TTS and MediaPipe face tracking:
```bash
# Download the MediaPipe face detector model
curl -L -o blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# Download Kokoro ONNX model and voices (if you want premium Kokoro TTS)
curl -L -o kokoro-v0_19.onnx https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/kokoro-v0_19.onnx
curl -L -o voices.bin https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/voices.bin
```

### 4. Install Python Packages
```bash
pip install -r requirements.txt
```

### 5. Start the Web App
```bash
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Once started, navigate to **`http://localhost:8000`** in your browser.

---

## 🖥️ Live Polling & Concurrency

The export pipeline runs on an asynchronous background threadpool. When you click **Render & Crop Clip**, the frontend receives a `job_id` and polls `/export/status/{job_id}` every second. This gives you **real-time precise progress indicators** showing exactly what frame and rendering stage the CPU is processing.

---

## 🛡️ Monetization & Compliance (Section 3 Guardrails)

YouTube and Facebook enforce strict rules against raw compilation channels.
This engine features an **in-app Compliance Checklist** that remains locked until the creator verifies:
1. **Ownership/Licensing:** User represents they own the raw footage or hold a clear commercial license.
2. **Transformative Value:** The engine applies animated dynamic captions, smart vertical crops, and customizable AI voiceovers, ensuring editorial/format transformation that complies with monetization policies.
3. **No Unlicensed Scraping:** Built-in workflows are gated, preventing illegal re-hosting or platform scraper violations.
