# Patch Notes — verified against the actual pushed repo

I cloned `mateho-369/Auto-Clip-Engine`, installed the real dependencies, and ran it.
The claim that the code was "tested and working flawlessly" did not hold up — the
server crashed on startup, and two of the five core features never worked at all.
Everything below was reproduced, not guessed.

## 1. FATAL — server could not start at all
`src/app.py` imported `HTMLFileResponse` from `fastapi.responses`. That class does
not exist in FastAPI/Starlette (confirmed: `ImportError: cannot import name
'HTMLFileResponse' from 'fastapi.responses'`). The app crashed before the first
request could ever be served — the README's own `uvicorn` command would fail
immediately.
**Fix:** use the `FileResponse` that was already imported, with `media_type="text/html"`.

## 2. Auto-crop silently lost all audio
`src/video_cropper.py` called `full_video.audio.subclip(...)`. MoviePy 2.x (the
version this code otherwise targets, since it uses `with_audio()` elsewhere)
renamed `.subclip()` to `.subclipped()` — `.subclip()` no longer exists (confirmed
against the installed `moviepy==2.1.2`). The call raised `AttributeError`, which was
caught by a broad `except`, silently falling back to renaming the silent
(audio-less) crop as the "final" clip — and still reporting success. Every exported
clip would have had no sound.
**Fix:** `.subclip()` → `.subclipped()`.

## 3. Voiceover feature never worked, at all
`src/voiceover_engine.py` called `.subclip()` five more times — same bug, same
root cause. The voiceover-overlay function would always throw, always get caught,
and always return `False`, so the AI-narration feature silently no-ops on every
single export despite being advertised as implemented and tested. I also removed
~15 lines of dead code that computed the same composite audio twice under
different variable names before using only the second copy.
**Fix:** `.subclip()` → `.subclipped()`, plus removed the redundant computation.

## 4. Highlight ranking was working off misaligned transcripts
`src/highlight_engine.py`'s `transcribe_audio_segments()` called
`recognizer.record(source, offset=start, duration=segment_duration)` in a loop,
passing an increasing `offset=start` each iteration. I checked the
`speech_recognition` library source directly: `offset` skips forward from
**wherever the stream cursor currently is**, not from the start of the file. Since
each call already advances the cursor, adding `offset=start` on top compounds —
each successive chunk drifts further from its labeled `start`/`end`, and the
stream hits EOF well before `total_duration` is reached, silently turning later
chunks empty. This corrupts the "Lexical & Speech" signal, which is 40% of the
final virality score — i.e. ranking quality degrades the further into the video
you go.
**Fix:** drop `offset` entirely and read sequentially with `duration` only, since
`AudioFile` already keeps its own cursor between calls.

## 5. Missing `requirements.txt`
The repo had no `requirements.txt` despite the README instructing `pip install
...` — added one with the actual dependencies used in the code.

## Verified
`from src import app` now imports successfully with no errors (previously crashed
on line 1 of execution). I have not been able to render an actual full video in
this environment (no GPU, and this sandbox can't reach the model-weight hosts
gTTS/Whisper-class tools need), so treat this as "the code now runs" rather than
"the whole video pipeline was rendered end-to-end" — worth a real test run on your
side with an actual video file before you trust it for production posting.
