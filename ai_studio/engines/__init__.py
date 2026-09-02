"""engines package — the model/media executors behind each pipeline stage.

3a voice        → :mod:`ai_studio.engines.tts`       (sherpa-onnx VITS, Khmer)
3b timbre       → :mod:`ai_studio.engines.rvc`       (RVC / Seed-VC / bypass)
4 video         → :mod:`ai_studio.engines.video`      (ComfyUI Wan 1.3B/5B → previz)
5 sfx           → :mod:`ai_studio.engines.sfx`        (MMAudio → procedural ambience)
7 assembly      → :mod:`ai_studio.engines.assembly`   (ffmpeg)

Each engine exposes a *sync* function (the scheduler runs them in threads) and a
`probe()` the settings page uses. They must never raise on missing hardware —
they return ``{"ok": False, "reason": ...}`` so the stage can fall back.
"""
