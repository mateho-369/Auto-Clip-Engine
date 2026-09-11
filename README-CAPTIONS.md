# Khmer Caption Typography — guide & Windows notes

The Studio burns captions into the exported MP4 with **one renderer**
(ffmpeg + libass, HarfBuzz shaping) for previews, tests and final exports.
What you see in the Typography & Captions panel is the exporter's own
output — not a CSS mock-up.

## Fonts

Five OFL-licensed Khmer families ship inside `ai_studio/assets/fonts/`
(license texts in the same folder, see `assets/fonts/README.md`):

| id | family | weights | role |
|----|--------|---------|------|
| `noto_sans_khmer` | Noto Sans Khmer | regular, bold | default UI/body |
| `noto_serif_khmer` | Noto Serif Khmer | regular, bold | cinema / editorial |
| `battambang` | Battambang | regular, bold | familiar body face |
| `kantumruy_pro` | Kantumruy Pro | regular, bold | bold social look |
| `moul` | Moul | regular | display/title |

Font selection is explicit — there is no silent system fallback. Requesting
a weight that does not exist for a family (e.g. Moul + bold) is a hard
error, never a synthetic substitution. No CDN is contacted; the web UI
serves the same TTFs from `/static/fonts/…`.

## Presets

`clean`, `cinema`, `bold social`, `soft card`, `editorial` — each expands
**server-side** to real export parameters (font, weight, size %, colors,
outline, shadow, panel, position, margins, spacing, max width/lines).
A preset chip + tweaks is the intended flow; `reset` returns every field to
the selected preset. Validation clamps out-of-range values and reports what
it adjusted.

## Style precedence

1. built-in preset (baked into the named preset),
2. global default (`settings.caption_style`, machine-wide),
3. per-project override (`project.settings.captions`).

Projects saved before captions existed load unchanged: a legacy
`settings.subtitle_style` is mapped into the modern schema (map itself is
unit-tested). The UI's own theme (light/dark) is unrelated to caption
render settings.

## Rendering guarantees

- Breaks happen **between dictionary words** (khmercut); clusters, coeng
  stack and marks are never split; trailing signs (។ ៕ ៖ ៗ …) never start
  a line; author spaces are preserved.
- Width is measured in **shaped pixels** (uharfbuzz), so sizing is exact at
  any output resolution (480×854 … 1080×1920, portrait or landscape).
- Overlong captions shrink per-block (bounded); if even the minimum size
  needs more lines than allowed, the text is still kept in full and a
  warning is attached to the manifest — never truncated.
- Karaoke `\k` tags use the same pixel budget; timings are a proportional
  estimate (labelled `estimated-proportional`), sentence windows are exact.
- If burned captions are requested but the burn fails, the pipeline **fails
  loudly** — it never silently hands you an uncaptioned MP4. The manifest
  records font file, renderer, timing and warnings for every export.

## Windows

- Nothing in the caption path shells out through a shell: FFmpeg is invoked
  with argument lists and escaped filter graphs, so project paths with
  spaces (`C:\Users\... \My Videos\...`) are safe.
- Fonts load from the installed package directory via `importlib.resources`
  — no absolute POSIX paths, no registry, no font-install step.
- Install: `python -m pip install -r requirements-studio.txt` (adds
  `uharfbuzz` + `khmercut`; both have wheels for Windows). `khmercut` is an
  optional accelerator — without it the studio falls back to cluster-based
  word hints, with slightly less clever line breaks.
- The preview cache lives under your data root
  (`data/studio/caption_previews`); it is bounded and safe to delete.

## Tests

```
python -m pip install -r requirements-studio.txt pytest httpx
PYTHONPATH=. pytest tests/test_studio_captions.py tests/test_studio_caption_wrap.py -v
PYTHONPATH=. pytest -q                # whole suite
```

Shaping-dependent tests skip automatically when `uharfbuzz`/`khmercut` are
absent (e.g. minimal CI), so the suite stays green everywhere.
