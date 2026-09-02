"""Khmer AI Content Studio — director-led, multi-agent local video pipeline.

This is the next major version of the ``ai_creator`` "AI Team" idea: instead of
one planner role doing everything, content flows through a *directed pipeline of
specialised local agents* where every stage's output is the next stage's input,
scheduled on a real asyncio task queue with live per-stage status.

    Mode A (Director: paste a locked script)  ┐
    Mode B (Controller: auto-generate Khmer)  ┘-> [1] Scene breakdown (Ollama)
                                                     ├──> [3a] Voice  sherpa-onnx vits-mms-khm
                                                     │      └──> [3b] Timbre  RVC
                                                     └──> [4] Video  ComfyUI Wan 1.3B/5B (480p)
                                                     -> [video fit] duration-match to voice
                                                     -> [5] SFX   MMAudio (video-to-audio)
                                                     -> [6] QA    reviewer (LLM + deterministic)
                                                     -> [7] Assembly ffmpeg -> final .mp4

Hardware budget (hard constraint everywhere): one RTX 5070 8GB, 16GB RAM, no
cloud.  Everything degrades gracefully: if a model/service is missing the stage
falls back to a deterministic local implementation (Khmer-aware segmentation,
procedural previz video, procedurally synthesised ambience) and the UI says so
loudly instead of stalling.

Nothing here phones out — every engine is a localhost call or a subprocess.
"""

__version__ = "1.0.0"
STUDIO_NAME = "Khmer AI Content Studio"
STUDIO_TAGLINE = "Director-led multi-agent video pipeline · 100% local"

__all__ = ["__version__", "STUDIO_NAME", "STUDIO_TAGLINE"]
