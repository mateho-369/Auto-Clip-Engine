#!/usr/bin/env bash

# AI Content Creator — one-command local setup (Linux / macOS)
# Verifies ffmpeg, checks Ollama + models, downloads Kokoro TTS weights,
# sets up the Python environment and optional extras (rembg, XTTS cloning).

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  ✦ AI Content Creator: Local Studio Setup ✦          ${NC}"
echo -e "${BLUE}======================================================${NC}\n"

# 1. FFmpeg
echo -e "${BLUE}[1/5] Checking FFmpeg...${NC}"
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✔ FFmpeg found on PATH.${NC}\n"
else
    echo -e "${YELLOW}⚠ FFmpeg not found on PATH.${NC}"
    echo -e "   Debian/Ubuntu:  ${GREEN}sudo apt-get update && sudo apt-get install -y ffmpeg${NC}"
    echo -e "   macOS:          ${GREEN}brew install ffmpeg${NC}"
    echo -e "   Windows:      ${GREEN}scoop install ffmpeg${NC} or ${GREEN}choco install ffmpeg${NC}"
    echo -e "   (The studio can also use the ffmpeg bundled with the Python 'imageio-ffmpeg' package.)\n"
fi

# 2. Ollama — the AI team brain
echo -e "${BLUE}[2/5] Checking Ollama (local LLM for the AI team)...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✔ Ollama installed! Pulling the default controller model (llama3.2:3b)...${NC}"
    (ollama pull llama3.2:3b && echo -e "${GREEN}✔ llama3.2:3b ready.${NC}") \
        || echo -e "${YELLOW}⚠ Could not pull llama3.2:3b — make sure 'ollama serve' is running, then: ollama pull llama3.2:3b${NC}"
    echo -e "${YELLOW}Optional — bigger models for better scripts (assign per-role in the UI):${NC}"
    echo -e "    ${GREEN}ollama pull qwen2.5:7b${NC}  (great all-rounder)   ${GREEN}ollama pull llama3.1:8b${NC}  (strong writing)\n"
else
    echo -e "${YELLOW}⚠ Ollama not found. Install it for the full AI team experience:${NC}"
    echo -e "    1) Download from ${GREEN}https://ollama.com/${NC}"
    echo -e "    2) ${GREEN}ollama pull llama3.2:3b${NC}"
    echo -e "   Without Ollama the studio still works in template-fallback mode.\n"
fi

# 3. Kokoro TTS weights (local, very human-like voice)
echo -e "${BLUE}[3/5] Verifying local Kokoro TTS weights...${NC}"
if [ ! -f "kokoro-v0_19.onnx" ]; then
    echo -e "${YELLOW}⬇ Downloading Kokoro-82M ONNX weights (~80 MB)...${NC}"
    curl -L -o "kokoro-v0_19.onnx" "https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/kokoro-v0_19.onnx"
    echo -e "${GREEN}✔ Kokoro ONNX model downloaded.${NC}"
else
    echo -e "${GREEN}✔ Kokoro ONNX model already present.${NC}"
fi
if [ ! -f "voices.bin" ]; then
    echo -e "${YELLOW}⬇ Downloading Kokoro voices catalog (~20 MB)...${NC}"
    curl -L -o "voices.bin" "https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/voices.bin"
    echo -e "${GREEN}✔ Kokoro voices downloaded.${NC}"
else
    echo -e "${GREEN}✔ Kokoro voices already present.${NC}"
fi

# 4. Python environment
echo -e "${BLUE}[4/5] Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment (venv)...${NC}"
    python3 -m venv venv
fi
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements-dev.txt
echo -e "${GREEN}✔ Core dependencies installed.${NC}\n"

# 5. Optional extras (interactive)
echo -e "${BLUE}[5/5] Optional extras...${NC}"
read -r -p "Install rembg (real background removal for character cutouts)? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    ./venv/bin/pip install rembg onnxruntime
    echo -e "${GREEN}✔ rembg installed (downloads its U2-Net model on first use).${NC}"
fi
read -r -p "Install TTS/XTTS v2 (voice CLONING from your 1-min recording, large ~2.5GB)? [y/N] " ans2
if [[ "$ans2" =~ ^[Yy]$ ]]; then
    ./venv/bin/pip install -r requirements-creator-optional.txt
    echo -e "${YELLOW}⬇ Downloading the XTTS v2 clone model (first time only, ~1.9GB)...${NC}"
    ./venv/bin/python - <<'EOF'
try:
    from TTS.api import TTS
    TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("✔ XTTS v2 model downloaded — voice cloning is READY.")
except Exception as e:
    print(f"⚠ XTTS model download failed ({e}). You can retry later with:")
    print("  ./venv/bin/python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')\"")
EOF
fi
echo ""

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  🎉 AI Content Creator setup complete! 🎉            ${NC}"
echo -e "${GREEN}======================================================${NC}\n"
echo -e "Start the studio:"
echo -e "👉 ${YELLOW}./venv/bin/python -m uvicorn ai_creator.app:app --host 0.0.0.0 --port 8000${NC}"
echo -e "Open ${GREEN}http://localhost:8000${NC} — step 1: create your character, step 2: assign your AI team."
echo -e "\n(Legacy Auto-Clip Engine still runs on port 8001:"
echo -e " ${YELLOW}./venv/bin/python -m uvicorn src.app:app --host 0.0.0.0 --port 8001${NC})\n"
