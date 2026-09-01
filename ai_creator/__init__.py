"""AI Content Creator — a local AI + bot video studio.

Modules:
    ollama_client  — local Ollama API client + robust JSON extraction
    team           — the AI team: role -> model assignment config
    planner        — controller-AI pipeline that plans & delegates work
    character      — your persistent character (face/body memory from photos)
    voice          — voice store + TTS engines (Kokoro / XTTS clone / gTTS)
    sfx            — 100% offline synthesized sound-effect library
    animation      — character entry/exit/idle/talk transforms
    transitions    — scene transition blending (fade, slide, zoom, wipe)
    renderer       — composes scenes -> final MP4 + SRT + mixed audio
    app            — FastAPI server + UI
"""

__version__ = "1.0.0"
