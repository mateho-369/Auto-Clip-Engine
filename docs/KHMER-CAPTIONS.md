# Khmer captions, end to end

*Why the studio burned empty boxes into Khmer captions, what replaced that, and how to
verify it yourself.*

---

## 1 · The defect (reproduced, then fixed)

Symptom: a finished vertical video whose narration was correct Khmer, whose `.srt` was
correct Khmer, and whose burned-in subtitles were **rows of empty rectangles** (tofu).

Root cause — two independent faults, both in the burn step:

1. **The named font did not exist.** The old subtitle library carried one hardcoded
   libass `force_style` string per style key, with `FontName=Khmer OS Battambang`
   written into it. That font is not shipped with the studio and is not installed on
   most machines. With no fontconfig family to resolve, libass fell back to a font
   without Khmer glyphs and drew `.notdef` boxes for every cluster.
2. **Nothing wrapped the text, and libass could not.** The `subtitles` filter in the
   static ffmpeg build used here is compiled **without** `ASS_FEATURE_WRAP_UNICODE`
   (see `captions.capabilities()`), so libass broke long Khmer lines at Latin
   word-spaces — and Khmer sentences frequently contain none. Lines either ran off the
   frame or were broken mid-cluster, which is visually wrong even when the font is right.

Evidence of the "before" state is kept as a frame of the old render
(`crop_A_default.png`, a row of tofu) next to the frames produced by the rewrite.

Three more behaviours made the fault hard to see, and are now impossible:

* the burn step was reached through a `try/except` that could continue without captions;
* the studio called the *burn* path only for export, while the UI's style gallery used a
  different code path, so a preview could look fine while the export was wrong;
* nothing compared the font actually used with the font that was requested.

## 2 · What the fix guarantees

| Guarantee | Enforced by |
|---|---|
| Every caption is drawn by a **bundled, OFL-licensed** font | `caption_style.FONT_FAMILIES`; `captions.font_dir_for_style()` builds a per-render font directory |
| libass is given the family that the file actually contains, plus `fontsdir` | `captions.ass_style_line()` / `build_ass()`; static ffmpeg has **no fontconfig**, so the `FontName` must match the TTF's real family |
| Khmer is wrapped **by us**, never by libass | `captions.fit_text()` → cluster-safe segmentation (`khmer.segment_clusters`), `WrapStyle: 2`, explicit `\N` |
| Preview and export are the **same renderer** | `POST /api/caption-preview` runs `captions.preview_frame()`, which writes and burns the same ASS the export writes |
| A requested caption burn **fails loudly** instead of disappearing | `captions.require_libass()` raises; `engines/assembly.assemble()` no longer swallows the error |
| The UI can always tell you the **font, size, style and warnings** that were used | `X-Caption-Style` / `-Font` / `-Warnings` / `-Lines` headers, `captions.render_metadata()`, and the run manifest |

## 3 · Architecture

```
khmer.py            script-aware text: clusters, sentence split, [[silent: …]], validate_script
   │
caption_style.py    the style schema: validation, presets, bundled-font registry, precedence
   │
captions.py         the single renderer: measure → fit → layout → ASS → burn / preview → metadata
   │
media.py            thin compatibility wrappers (old call sites keep working)
   │
engines/assembly.py stage 7: scenes → cues (+ word timings) → SRT / ASS / burn → manifest
   │
api.py + UI         Typography & Captions inspector, previews, per-project overrides
```

Design rule: **one style dict, one cue list, one renderer.** The ASS file that the
preview burns is byte-for-byte the same kind of file the export burns, and the manifest
records the font file that produced it.

### Files

| File | Responsibility |
|---|---|
| `ai_studio/fonts/` | 4 bundled families (Regular + Bold, Light for Battambang) with their `OFL.txt` licences |
| `ai_studio/caption_style.py` | style dict, field validation, 5 presets, font registry, `safe_font_path()`, `normalize_patch()`, `merge_styles()` |
| `ai_studio/captions.py` | measurement (uharfbuzz when available), fit/wrap, layout + panel geometry, ASS writer, burn, preview frame, `coverage_gaps()`, `capabilities()` |
| `ai_studio/khmer.py` | cluster segmentation, danda-aware sentence splitting, `validate_script()` |
| `scripts/verify_khmer_captions.py` | burns a matrix of videos and **measures the pixels** (see §8) |

## 4 · The style dict

One flat dict; every field validated server-side. Defaults live in
`caption_style.DEFAULT_STYLE`, presets in `caption_style.PRESETS`.

| Field | Range / values | Default | Meaning |
|---|---|---|---|
| `font` | `noto-sans-khmer`, `noto-serif-khmer`, `kantumruy-pro`, `battambang` | `noto-sans-khmer` | bundled family id |
| `weight` | shipped weights per family (`400`, `700`, Battambang also `300`) | `400` | static instance |
| `font_size_px` | 14 – 220 | `72` | px on the **1080×1920 reference**, rescaled per frame height |
| `color` | `#rrggbb` / `#aarrggbb` / named | `#FFFFFF` | text colour |
| `outline_width` / `outline_color` | 0 – 10 | `3` / `#000000` | ASS outline |
| `shadow` / `shadow_color` / `shadow_opacity` | 0 – 12 / colour / 0 – 1 | `0.35` / `#000000` / `0.65` | soft shadow |
| `background`, `background_color`, `background_opacity`, `background_padding`, `background_radius` | bool, colour, 0 – 1, 0 – 120, 0 – 120 | `False`, `#0B0F14`, `0.55`, `26`, `18` | the rounded panel, drawn as a real ASS vector shape |
| `position` | `bottom` \| `center` \| `top` | `bottom` | vertical anchor |
| `alignment` | `center` \| `left` \| `right` | `center` | horizontal |
| `margin_h` / `margin_v` | 0 – 30 / 0 – 40 (% of frame) | `8` / `9` | safe margins |
| `line_spacing` | 0.8 – 2.4 | `1.25` | line box multiple |
| `max_lines` / `max_width_pct` | 1 – 5 / 40 – 100 | `3` / `88` | wrap budget |
| `letter_spacing` | 0 – 6 | `0` | tracking |
| `preset` | a preset key or `custom` | `custom` | what the UI highlights |

### Presets

A preset **is** a style dict, so "apply Cinema" and "export" can never disagree.

| Key | Look | Font / size |
|---|---|---|
| `clean` | white, restrained outline + soft shadow, bottom | Noto Sans Khmer 72 |
| `cinema` | warm ivory serif, wide spacing, quiet | Noto Serif Khmer 68 |
| `bold-social` | high-contrast yellow, thick outline, centred | Kantumruy Pro 700 · 88 |
| `soft-card` | white on a dark translucent rounded panel | Noto Sans Khmer 68 |
| `editorial` | calm serif, generous margins, essay feel | Noto Serif Khmer 66 |

Legacy keys still resolve: `bold_yellow → bold-social`, `minimal_top → editorial`,
anything else → `clean`.

### Precedence

```
built-in default  →  studio settings (cfg["caption_style"])  →  project override  →  request patch
```

* A project that never saved a style simply inherits — **nothing is rewritten for old projects.**
* A request body is a **patch** (`caption_style.normalize_patch`): `{"color": "#FF0000"}`
  changes one field and leaves the rest alone, while a bad value is still a `400`, never a
  silently repaired default.
* Naming a preset in a patch (e.g. `{"preset": "cinema"}`) restarts from that preset, which
  is what "apply this preset" has to mean.
* A style that no longer equals the preset it names is relabelled **`custom`** on the way in
  and out, so the UI highlight, the API payload and the exported pixels agree. Numeric
  rounding or a different case in a hex colour never triggers the relabel; a real edit does.

## 5 · Bundled fonts

| id | Family | Weights | Licence | Notes |
|---|---|---|---|---|
| `noto-sans-khmer` | Noto Sans Khmer | 400, 700 | OFL-1.1 | default; cleanest on small screens |
| `noto-serif-khmer` | Noto Serif Khmer | 400, 700 | OFL-1.1 | editorial / cinema |
| `kantumruy-pro` | Kantumruy Pro | 400, 700 | OFL-1.1 | geometric, high-contrast |
| `battambang` | Battambang | 300, 400, 700 | OFL-1.1 | the classic Khmer UI face; **no `–` / `—` glyph** (see limits) |

`ai_studio/fonts/LICENSE-OFL-*.txt` ships the OFL text for each family.

Serving: `GET /api/fonts/{font_id}/{filename}` returns a bundled TTF for the UI's
`@font-face`. The route never accepts a path — `caption_style.safe_font_path()` matches
the basename against the family's declared files only, so `../` can never escape
`ai_studio/fonts/` (covered by a test).

## 6 · Text handling rules

* **Measurement**: uharfbuzz shaping when available, else a documented estimate;
  `render_metadata()["measurement"]` says which one was used.
* **Wrapping** happens at cluster boundaries. A line never starts or ends with
  U+17D2 (coeng); a coeng always keeps its subscript consonant. Khmer runs with no
  spaces are split at cluster boundaries and **reported** as `unbreakable run split…`.
* **Explicit newlines** in the script survive as line breaks.
* **`[[silent: …]]`** is *displayed* in captions but never spoken (and never sent to TTS).
* **Markup safety**: unbalanced `[[`, stray ASS control characters and braces are escaped;
  `khmer.validate_script()` lists them instead of editing your text.
* **Fit**: the font size is reduced (down to 60 % of the request, and the reduction is
  reported) before anything is clipped; `bounds_problems()` reports ink that would leave
  the frame or the safe margins.
* **Language mix**: Khmer, Latin and digits are measured separately and reported by
  `coverage_gaps()`.

### Karaoke — honest about what it is

`karaoke.enabled` colours words as they are spoken using ASS `\k` tags. The per-word
windows are **proportional estimates** inside each cue (from the scene's word/syllable
timing estimates), not forced alignment against the audio. Every surface says so:
`render_metadata()["karaoke"]["timing"]` reads *"proportional estimate … not forced
alignment"*, and the UI tooltip repeats it.

## 7 · HTTP surface

| Route | Purpose |
|---|---|
| `GET /api/caption-style` | bootstrap: bundled fonts, presets, palettes, current global style, karaoke, capabilities, reference size, precedence |
| `POST /api/caption-style` | save the studio-wide style (+ optional `burn_captions`, `emit_srt`, `karaoke`); validated → `400` with the reason |
| `POST /api/caption-style/reset` | back to a named preset |
| `GET /api/projects/{id}/caption-style` | global + project style + effective style + `is_default` |
| `PUT|POST /api/projects/{id}/caption-style` | save/clear the project override (partial patches inherit) |
| `POST /api/caption/validate` | `khmer.validate_script()` for one text (`ok`, `errors`, `warnings`, `stats`) |
| `POST /api/caption-preview` | **real** PNG rendered by the exporter's renderer; headers `X-Caption-Style`, `X-Caption-Font`, `X-Caption-Warnings`, `X-Caption-Lines`, `X-Caption-Url`; optional `project_id` / `scene_idx` composites onto a real frame of the project |
| `GET /api/caption-style/previews` | preset × font sample PNGs (cached on disk) |
| `POST /api/projects/{id}/render-captions` | re-burn captions onto the existing cut (no pipeline re-run), publishes the `final_captions` asset |
| `GET /api/assets/{id}/download?cap=1` | download the newest captioned cut of the project (default when the asset is a final cut) |
| `GET /api/fonts/{font_id}/{filename}` | bundled TTF for `@font-face` |
| `GET /api/style-previews` | legacy gallery, now rendered through this same pipeline |

Config keys: `caption_style`, `karaoke.enabled`, `karaoke.color`,
`assembly.burn_captions`, `assembly.emit_srt`, `assembly.caption_seconds_per_cue`
(`3.4` s ≈ reading time for one cue), `tts.allow_online` (unrelated to captions but
documented in §9).

## 8 · How to verify (do this, don't trust the doc)

```bash
# 1. the fast matrix: burns a real video per preset and measures the pixels
PYTHONPATH=. python scripts/verify_khmer_captions.py --out /tmp/verify --quick

# 2. the full matrix (several sizes, tones and presets + contact sheets + report.json)
PYTHONPATH=. python scripts/verify_khmer_captions.py --out /tmp/verify

# 3. the test suite (renderer, style validation, HTTP surface, end-to-end export)
PYTHONPATH=. pytest tests/test_studio_captions.py -q
PYTHONPATH=. pytest tests/test_studio_text.py tests/test_studio_pipeline.py -q
```

What the harness does that a screenshot cannot:

* renders every test string over a **real** previz backdrop at several frame sizes;
* takes a caption-free frame from the same video and the caption frame, diffs them, and
  requires **ink inside the caption band** — a caption that renders nothing fails;
* computes the ink bounding box and warns when it leaves the frame or the safe area;
  on the bundled fonts the em-dash/en-dash gap of Battambang is reported here;
* counts `.notdef` glyphs **per character** with shaping, so a missing glyph is found
  even when the surrounding text renders fine.

Known-good result (this machine, static ffmpeg + uharfbuzz, no fontconfig):

```
portrait_1080 mid clean: 12/12 frames with ink        (× 5 presets)
font noto-sans-khmer: notdef 0 · noto-serif-khmer: notdef 0 · kantumruy-pro: notdef 0
font battambang: notdef 2 → ['–', '—']                (expected, see limits)
bounds/overflow warnings: none
```

## 9 · Honest limits

* **Word timing is an estimate.** No ASR/forced alignment in this repo — karaoke is
  proportional, and labelled as such everywhere.
* **Battambang has no `–` or `—`.** libass substitutes another bundled font for those
  glyphs, so the dash still appears, but the mismatch is real: it is reported by
  `coverage_gaps()` / `coverage_warnings()` and in the preview warnings.
* **Four families only.** Anything else is refused rather than silently substituted —
  that refusal is the whole point of the fix.
* **libass cannot wrap Khmer here.** Keep wrapping in `khmer`/`captions`; never rely on
  `ASS_FEATURE_WRAP_UNICODE` being compiled in.
* **Glyph fallback is libass' choice**, not a per-style guarantee: a *bundled* font is
  always the primary face, but for uncovered characters libass may pick another installed
  font. That is why coverage gaps are reported instead of hidden.
* **`edge-tts` is opt-in and online** (`tts.allow_online`): it sends the script text to
  Microsoft and is never enabled by default. It is unrelated to the caption renderer, but
  it is the one place where real Khmer speech becomes available without a local model.
* **RVC bypass is labelled.** When no voice model is available the base voice passes
  through and the stage says so, naming the engine that actually produced the audio.
