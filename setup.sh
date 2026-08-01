#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Setup colorful output constants
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  ⚡ Global Highlights Auto-Clip Engine: Setup Wizard ⚡ ${NC}"
echo -e "${BLUE}======================================================${NC}\n"

# 1. Check for FFmpeg on PATH
echo -e "${BLUE}[Step 1/5] Checking for FFmpeg...${NC}"
if command -v ffmpeg &> /dev/null && command -v ffprobe &> /dev/null; then
    echo -e "${GREEN}✔ FFmpeg and FFprobe are already installed and available on PATH!${NC}\n"
else
    echo -e "${RED}✘ FFmpeg or FFprobe was not found on your system PATH!${NC}"
    echo -e "${YELLOW}Please install FFmpeg to continue:${NC}"
    echo -e "  - On Debian/Ubuntu:  ${GREEN}sudo apt-get update && sudo apt-get install -y ffmpeg${NC}"
    echo -e "  - On macOS (Homebrew): ${GREEN}brew install ffmpeg${NC}"
    echo -e "  - On Windows (Scoop/Choco): ${GREEN}scoop install ffmpeg ${NC}or${GREEN} choco install ffmpeg${NC}"
    echo -e "Please install it, restart your terminal, and run this setup script again.\n"
    exit 1
fi

# 2. Check for Ollama
echo -e "${BLUE}[Step 2/5] Checking for Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✔ Ollama is installed!${NC}"
    echo -e "${BLUE}Pulling default local LLM (llama3.2:3b)...${NC}"
    # Non-blocking, print progress
    ollama pull llama3.2:3b || echo -e "${YELLOW}⚠️ Could not pull llama3.2:3b automatically. Make sure Ollama app is running and try 'ollama pull llama3.2:3b' manually.${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️ Ollama was not found on your system PATH.${NC}"
    echo -e "If you want to use the ${YELLOW}Semantic AI Re-Ranking pass${NC}, please install Ollama:"
    echo -e "  - Download from: ${GREEN}https://ollama.com/${NC}"
    echo -e "  - Run: ${GREEN}ollama pull llama3.2:3b${NC} once installed.\n"
fi

# 3. Download Local AI model weights
echo -e "${BLUE}[Step 3/5] Verifying local AI model files...${NC}"

# MediaPipe blaze_face
MP_MODEL="blaze_face_short_range.tflite"
if [ ! -f "$MP_MODEL" ]; then
    echo -e "${YELLOW}⬇ Downloading MediaPipe Face Tracking Model (225 KB)...${NC}"
    curl -L -o "$MP_MODEL" "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    echo -e "${GREEN}✔ MediaPipe model downloaded!${NC}"
else
    echo -e "${GREEN}✔ MediaPipe Face Tracking model already present.${NC}"
fi

# Kokoro model files
KOKORO_MODEL="kokoro-v0_19.onnx"
KOKORO_VOICES="voices.bin"

if [ ! -f "$KOKORO_MODEL" ]; then
    echo -e "${YELLOW}⬇ Downloading local Kokoro TTS ONNX weights (~80 MB)...${NC}"
    curl -L -o "$KOKORO_MODEL" "https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/kokoro-v0_19.onnx"
    echo -e "${GREEN}✔ Kokoro ONNX model downloaded!${NC}"
else
    echo -e "${GREEN}✔ Kokoro ONNX model already present.${NC}"
fi

if [ ! -f "$KOKORO_VOICES" ]; then
    echo -e "${YELLOW}⬇ Downloading Kokoro local voices catalog (~20 MB)...${NC}"
    curl -L -o "$KOKORO_VOICES" "https://github.com/thewhpoly/kokoro-onnx/releases/download/v0.2.0/voices.bin"
    echo -e "${GREEN}✔ Kokoro voices downloaded!${NC}"
else
    echo -e "${GREEN}✔ Kokoro voices already present.${NC}"
fi
echo ""

# 4. Create virtual environment & install requirements
echo -e "${BLUE}[Step 4/5] Setting up isolated Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment (venv)...${NC}"
    python3 -m venv venv
fi

echo -e "${BLUE}Activating venv and installing pinned dependencies...${NC}"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo -e "${GREEN}✔ Dependencies installed successfully in virtual environment!${NC}\n"

# 5. Finished setup!
echo -e "${BLUE}[Step 5/5] Finishing Setup...${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  🎉 Global Highlights v3.0 setup is complete! 🎉 ${NC}"
echo -e "${GREEN}======================================================${NC}\n"

echo -e "To start the local web application, run the following command:"
echo -e "👉 ${YELLOW}./venv/bin/python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload${NC}\n"
echo -e "Navigate to ${GREEN}http://localhost:8000${NC} in your browser."
echo -e "To run the automated test suite, execute: ${GREEN}PYTHONPATH=. ./venv/bin/pytest${NC}\n"
