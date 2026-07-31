# ⚡ Global Highlights: Auto-Clip Engine

Global Highlights is an advanced, autonomous AI clipping engine that transforms long-form landscape videos (16:9) into viral, professional 9:16 vertical shorts optimized for YouTube Shorts, Instagram Reels, and Facebook Reels.

It runs completely locally, featuring a beautiful Tailwind CSS-powered dashboard, automatic face-tracking, dual-layer animated captions, and high-quality narrator voiceover overlay with intelligent audio ducking.

---

## 🛠️ System Stack & Final Decision

### 1. Highlight/Virality Detection
- **Our Selection:** Multi-Modal Sound Peak & Motion Analysis + Lexical Scoring Framework.
- **Why:** Full local execution without GPU/VRAM limits, identifying laughing spikes, exclamation triggers, and high visual action density.

### 2. Speech-to-Text / Captioning
- **Our Selection:** `SpeechRecognition` (Google Web Speech API / PocketSphinx fallback) + Dynamic Word Duration Interpolator + OpenCV Dual-Layer Caption Burn-In.
- **Why:** Outperforms heavy PyTorch models (Whisper) that trigger out-of-disk/memory crashes in lightweight cloud or sandbox environments, while delivering a robust, responsive karaoke-style subtitle engine.

### 3. Open-Source Text-to-Speech (TTS)
- **Our Selection:** `gTTS` (Google Text-to-Speech) with multi-accent support (US, UK, CA).
- **Why:** Light, Apache-2.0 licensed, highly reliable, and provides professional-grade commentary tracks without VRAM requirements.

### 4. Auto Vertical-Crop / Face-Tracking
- **Our Selection:** OpenCV Haar Cascade Frontal Face Detector + Exponential Moving Average (EMA) Motion Smoothing.
- **Why:** Eliminates the jerky camera jumps of simple box-cropping, offering a buttery smooth "cinematic slide/glide" that continuously follows the speaker.

---

## ⚙️ How the Highlight Ranking Engine Works

The virality index is calculated dynamically for 20-second sliding windows using a **multi-signal weighted average**:

$$\text{Virality Score} = 0.35 \times A_e + 0.40 \times L_h + 0.15 \times V_m + 0.10 \times H_s$$

Where:
- **Audio Energy ($A_e$ - 35%):** Measures Root Mean Square (RMS) spikes indicating punchlines, jokes, shouts, or dramatic commentary.
- **Lexical/Speech Hook ($L_h$ - 40%):** Scores word density and searches for high-impact hook terms (`wow`, `crazy`, `unbelievable`, `omg`, `secrets`, etc.).
- **Visual Motion ($V_m$ - 15%):** Analyzes frame-to-frame grid differentials to prioritize active sequences over static, silent slides.
- **Hook Strength ($H_s$ - 10%):** Assesses audio intensity in the first 3 seconds of the clip to ensure a high-retention thumbnail/start.

Overlapping candidate intervals are deduplicated using a greedy algorithm: whenever two intervals overlap by more than 30%, only the higher-scoring interval is retained.

---

## 🚀 Step-by-Step Setup & Running Guide

### 1. Install System Dependencies
Make sure you have `ffmpeg` and `ffprobe` installed on your machine.
```bash
# On Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# On macOS (using Homebrew)
brew install ffmpeg
```

### 2. Install Python Packages
```bash
pip install fastapi uvicorn moviepy gTTS SpeechRecognition opencv-python pydub jinja2 numpy pillow
```

### 3. Start the Web App
```bash
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Once started, navigate to `http://localhost:8000` in your web browser.

---

## 🔀 Swapping Components Later

### To Swap in `faster-whisper` (Local Whisper):
1. Install faster-whisper: `pip install faster-whisper`
2. Update `src/highlight_engine.py` to initialize the model:
```python
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")
```
3. Use `model.transcribe()` to retrieve precise word-by-word timestamps instead of the duration estimator.

### To Swap in `Kokoro-82M` or `XTTS-v2` for voice cloning:
1. Install Kokoro: `pip install kokoro-onnx` or install XTTS-v2 via `tts`.
2. Update the `generate_voiceover_mp3` method in `src/voiceover_engine.py` to point to the local Kokoro ONNX model and generate realistic cloned voiceover tracks!

---

## 🛡️ Monetization & Compliance (Section 3 Guardrails)

YouTube and Facebook enforce strict rules against raw compilation channels (e.g. funny fail channels made from re-uploaded clips).
This engine features an **in-app Compliance Checklist** that remains locked until the creator verifies:
1. **Ownership/Licensing:** User represents they own the raw footage or hold a clear commercial license.
2. **Transformative Value:** The engine applies animated dynamic captions, smart vertical crops, and customizable AI voiceovers, ensuring editorial/format transformation that complies with monetization policies.
3. **No Unlicensed Scraping:** Built-in workflows are gated, preventing illegal re-hosting or platform scraper violations.
