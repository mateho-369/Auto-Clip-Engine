# ComfyUI workflow templates

The studio never builds a node graph in code — it **injects values into a
ComfyUI API-format workflow** at `{{PLACEHOLDER}}` markers, then submits it to
`/prompt`. Two consequences:

1. These files are *starting points*. If your ComfyUI / custom-node version names
   things differently, export your own working workflow (ComfyUI → `Save (API Format)`,
   or Dev-mode → *Export (API)*), drop it in this folder
   (`ai_studio/workflows/`) or in `<data_dir>/workflows/`, put the markers where the
   values belong, and pick it in **Settings → Video / SFX**.
2. If a marker is missing, the run says so
   (`unresolved placeholders: …`) rather than quietly rendering the wrong thing.

## Available placeholders

| Marker | Meaning |
|---|---|
| `{{PROMPT}}` | positive visual prompt for the scene (from Stage 1) |
| `{{NEGATIVE}}` | negative prompt (studio default blocks text/watermark/flicker/…) |
| `{{WIDTH}}` `{{HEIGHT}}` | output size, kept ≤ 480×854 by the VRAM guard |
| `{{FRAMES}}` | frame count = clip seconds × `{{FPS}}`, capped by `video.max_frames` |
| `{{FPS}}` | frames per second of the generated clip |
| `{{DURATION}}` | clip length in seconds (float) — MMAudio uses this |
| `{{STEPS}}` `{{CFG}}` `{{SHIFT}}` `{{SEED}}` `{{MOTION}}` | sampler settings |
| `{{TEXT}}` `{{MOOD}}` | the scene's Khmer line and its mood tag |
| `{{SFX_PROMPT}}` | natural-ambience description produced from the mood tag |
| `{{VIDEO_PATH}}` | input video already uploaded into ComfyUI (MMAudio) |
| `{{AUDIO_PATH}}` | input audio inside ComfyUI |
| `{{START_IMAGE}}` | uploaded start frame (Wan2.2 TI2V) |
| `{{OUT_PREFIX}}` | output filename prefix |
| `{{SAMPLE_RATE}}` | audio sample rate |

A placeholder that sits as the *whole* string value is replaced with a typed
value (so `"length": "{{FRAMES}}"` becomes the integer `49`, not `"49"`).

## Wan model files expected by these templates

* `wan2.1_t2v_1.3b_480p.json` — `models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors`,
  `models/text_encoders/umt5_xxl_fp16.safetensors`, `models/vae/wan2.1_vae.safetensors`.
  This is the reliable choice on 8GB VRAM (≈480p, 17–81 frames).
* `wan2.2_ti2v_5b_480p.json` — `wan2.2_ti2v_5B_fp16.safetensors` loaded as
  `fp8_e4m3fn_scaled`, `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `wan2.2_vae.safetensors`.
  Only switch to this if `Settings → Video` reports enough free VRAM; start ComfyUI with
  `--lowvram` (or `--cache-none`) so its native offloading keeps you under 8GB.
* `mmaudio_small_480p.json` — needs the [`kijai/ComfyUI-MMAudio`](https://github.com/kijai/ComfyUI-MMAudio)
  node pack and the **small** model set in `ComfyUI/models/mmaudio/` (the small
  checkpoint is the "below 8GB VRAM" flavour; `mmaudio_large_44k_v2` is not).

> If a video/SFX engine is missing, the pipeline falls back to the CPU previz
> renderer and procedural ambience so you still get a complete draft — and the
> stage card in the UI tells you which engine actually produced it.
