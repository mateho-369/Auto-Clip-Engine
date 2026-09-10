"""Stage 7 — Final Assembly (ffmpeg).

Inputs per scene: duration-matched silent video, converted voice wav, ambience
wav. Output: one MP4 (H.264 yuv420p + AAC, faststart), plus an SRT, a poster
frame and a `manifest.json` that records every scene, engine, prompt and file
that produced this cut — the "memory" the brief wants, exported with the video.

Assembly is deliberately forgiving: a scene with a missing clip gets the previz
renderer or a black slate with the voice over it, and the manifest says which one
happened, so the Director always gets a watchable cut instead of a stack trace.
"""
import os

from .. import khmer, media, previz
from ..util import (ensure_dir, jdump, media_duration, write_json)


def _caption_windows(scenes, starts, k_end):
    """One caption per SENTENCE (professional subtitle rhythm).

    A scene's narration often holds two sentences; showing the whole scene
    text at once produced stacked multi-line boxes with mid-sentence breaks.
    Each scene window [start_i, k_end_i] is split at sentence boundaries and
    the time distributed proportionally to sentence length (clusters) —
    scene boundaries stay exact, in-scene sentence times are an estimate.
    """
    windows = []
    for i, s in enumerate(scenes):
        t0 = float(starts[i])
        t1 = max(t0 + 0.6, float(k_end[i]))
        disp = khmer.display_text(s.get("text", "")).strip()
        if not disp:
            continue
        sents = [x.strip() for x in khmer.split_sentences(disp) if x.strip()]
        if len(sents) <= 1:
            windows.append((t0, t1, disp))
            continue
        weights = [max(4.0, float(khmer.cluster_len(x))) for x in sents]
        total = sum(weights)
        cur = t0
        for j, (w, sent) in enumerate(zip(weights, sents)):
            if j == len(sents) - 1:
                windows.append((cur, t1, sent))
            else:
                end = cur + (t1 - t0) * (w / total)
                windows.append((cur, end, sent))
                cur = end
    return windows


def assemble(project, scenes, stage_assets, cfg, out_dir, run_id="", progress=None,
             allow_previz=True):
    """scenes: [{idx, text, ...}], stage_assets: {kind: {idx: {path, duration, meta}}}"""
    asm = cfg.get("assembly", {})
    v = cfg.get("video", {})
    sfx_cfg = cfg.get("sfx", {})
    width, height = int(v.get("width", 480)), int(v.get("height", 854))
    fps = int(asm.get("fps", 24))
    # deterministic gap between lines/scenes (tts.line_gap_sec, default 1.0s)
    gap = max(0.0, float((cfg.get("tts", {}) or {}).get("line_gap_sec", 1.0)))
    ensure_dir(out_dir)
    work = ensure_dir(os.path.join(out_dir, f".assembly_{run_id or 'x'}"))

    seg_videos, voice_tracks, amb_tracks, starts, notes = [], [], [], [], []

    # ---- caption style: the validated, effective look for THIS project
    # (global settings.captions ← project override / legacy subtitle_style key)
    from .. import captions as cap_mod
    cap_style = cap_mod.effective_style(project, cfg)
    cap_warnings: list[str] = []

    # optional rendered title card (assembly.title_style) — a silent intro clip
    title_dur = 0.0
    title_style = str(asm.get("title_style") or "")

    # ---- AV-sync note on transitions: an xfade crossfade OVERLAPS clips, so
    # the joined picture is fade*(n-1) shorter than the sum of the parts.
    # Narration delays, scene starts and the SRT are computed with the SAME
    # overlap, so audio and captions stay locked to the picture. A crossfade
    # is only used when explicitly configured; the default is a hard cut
    # (zero drift by construction).
    transition = str(asm.get("transition") or "cut")
    fade = float(asm.get("fade_sec", 0.0) or 0.0)
    n_seg_est = len(scenes) + (1 if title_style else 0)
    eff_fade = fade if (transition == "crossfade" and fade > 0.02 and n_seg_est > 1) else 0.0
    if transition == "crossfade" and eff_fade == 0.0 and fade > 0.02:
        notes.append("crossfade requested but not applicable — using hard cuts "
                     "(audio/caption sync stays exact)")
    if title_style:
        try:
            card = os.path.join(work, "title_card.mp4")
            title_text = (asm.get("title_text") or project.get("title") or "")[:120]
            media.render_title_card(card, title_text, title_style, width, height,
                                    min(24, fps), duration=2.6)
            title_dur = media_duration(card, 0.0) or 2.6
            seg_videos.append(card)
            notes.append(f"title card '{title_style}' rendered at the start")
        except Exception as e:
            notes.append(f"title card skipped: {str(e)[:140]}")
    cursor = title_dur
    total = len(scenes) or 1
    if title_dur and eff_fade:
        # the title card also overlaps the first scene under xfade
        cursor = max(0.0, title_dur - eff_fade)
    for i, scene in enumerate(scenes):
        idx = int(scene.get("idx", i))
        if progress:
            progress(100.0 * i / total, f"scene {idx + 1}/{total}: normalising picture")
        video = (stage_assets.get("video_fit", {}).get(idx) or
                 stage_assets.get("video", {}).get(idx) or
                 stage_assets.get("talking_head", {}).get(idx))
        voice = (stage_assets.get("voice_final", {}).get(idx) or
                 stage_assets.get("voice", {}).get(idx))
        ambient = stage_assets.get("ambient", {}).get(idx)
        v_dur = media_duration(voice["path"], 0.0) if voice and voice.get("path") else 0.0
        dur = max(0.8, v_dur or float(scene.get("estimated_duration_sec") or 4.0))
        tail = gap if i < total - 1 else 0.0

        clip = None
        if video and video.get("path") and os.path.exists(video["path"]):
            clip = os.path.join(work, f"seg{i:02d}.mp4")
            try:
                media.normalize_clip(video["path"], clip, width, height, fps, duration=dur,
                                     tail_pad=tail)
            except Exception as e:
                notes.append(f"scene {idx + 1}: could not normalise clip ({str(e)[:110]}); using previz")
                clip = None
        if clip is None and allow_previz:
            clip = os.path.join(work, f"seg{i:02d}.previz.mp4")
            try:
                previz.render_clip(clip, duration=dur, width=width, height=height, fps=fps,
                                   mood_tag=scene.get("mood_tag") or "",
                                   visual_prompt=scene.get("visual_prompt") or "", seed=idx)
                notes.append(f"scene {idx + 1}: picture is a CPU previz draft")
            except Exception as e:
                notes.append(f"scene {idx + 1}: previz failed ({str(e)[:90]}) — black slate")
                clip = _black_slate(work, i, dur, width, height, fps)
        if clip is None:
            clip = _black_slate(work, i, dur, width, height, fps)
        if tail > 0.02 and media_duration(clip, dur) < dur + tail - 0.05:
            clip = _pad_tail(work, clip, i, tail)
        real_dur = media_duration(clip, dur)
        seg_videos.append(clip)
        if voice and voice.get("path") and os.path.exists(voice["path"]):
            voice_tracks.append({"path": voice["path"], "gain": 1.0, "delay": cursor,
                                 "is_voice": True, "fade_in": 0.01, "fade_out": 0.08})
        else:
            notes.append(f"scene {idx + 1}: no voice track — silent picture")
        if ambient and ambient.get("path") and os.path.exists(ambient["path"]):
            amb_tracks.append({"path": ambient["path"], "gain": float(sfx_cfg.get("voice_duck_gain", 0.32)),
                              "delay": cursor, "fade_in": 0.25, "fade_out": 0.5})
        starts.append(cursor)
        cursor += real_dur - (eff_fade if i < total - 1 else 0.0)
        if progress:
            progress(100.0 * (i + 1) / total, f"scene {idx + 1}/{total}: {real_dur:.1f}s")

    if not seg_videos:
        raise RuntimeError("nothing to assemble — no scene has a picture")

    if progress:
        progress(72.0, "concatenating scenes")
    silent = os.path.join(out_dir, f"{project.get('id', 'project')}_{run_id or 'run'}.silent.mp4")
    media.concat_clips(seg_videos, silent, fps=fps,
                       transition="crossfade" if eff_fade > 0.02 else "cut",
                       fade=eff_fade if eff_fade > 0.02 else 0.0,
                       work_dir=work)

    if progress:
        progress(84.0, "mixing narration + ambience")
    total_dur = media_duration(silent, cursor) or cursor
    mix = os.path.join(work, "mix.wav")
    tracks = voice_tracks + amb_tracks
    info = media.mix_audio(tracks, mix, total_sec=total_dur,
                           duck={"gain": float(sfx_cfg.get("voice_duck_gain", 0.32)),
                                 "threshold": 0.02, "pad": 0.22} if (amb_tracks and voice_tracks) else None)
    final_audio = os.path.join(work, "mix.norm.wav")
    if asm.get("loudnorm", True):
        try:
            media.loudnorm(mix, final_audio, target_lufs=float(asm.get("loudnorm_target_lufs", -16.0)))
        except Exception:
            final_audio = mix
    else:
        final_audio = mix

    if progress:
        progress(90.0, "encoding final MP4")
    final = os.path.join(out_dir, f"{project.get('id', 'project')}_{run_id or 'run'}.mp4")
    media.mux(silent, final_audio, final, crf=int(asm.get("crf", 23)),
              preset=str(asm.get("preset", "veryfast")), audio_kbps=int(asm.get("audio_kbps", 160)),
              video_codec=str(asm.get("video_codec", "libx264")), max_dur=total_dur)
    if not os.path.exists(final) or os.path.getsize(final) < 1024:
        raise RuntimeError("final encode failed (no readable MP4 produced)")

    out = {"path": final, "duration": media_duration(final, total_dur),
           "size_bytes": os.path.getsize(final), "width": width, "height": height,
           "fps": fps, "scenes": len(scenes), "notes": notes,
           "audio_peak_info": info, "line_gap_sec": gap, "title_style": title_style,
           "subtitle_style": cap_style.get("preset", "clean"),
           "transition": ("crossfade" if eff_fade > 0.02 else "cut"),
           "transition_fade_sec": eff_fade}

    # ---- SRT sidecar: sentence timing across each scene's real window; the
    # last scene ends at the video's actual length (no fake +3 s tail)
    srt = ""
    if asm.get("emit_srt", True):
        srt = os.path.join(out_dir, os.path.splitext(os.path.basename(final))[0] + ".srt")
        # display_text: [[silent: …]] words stay on screen even though not spoken
        media.write_srt([khmer.display_text(s.get("text", "")) for s in scenes], starts, srt,
                        total_duration=media_duration(final, total_dur))
        out["srt"] = srt

    # ---- burned captions: the SAME builder + libass path the preview uses.
    # A requested burn that cannot render is a HARD error — we never hand back
    # an uncaptioned MP4 as if it satisfied the request (the stage records the
    # exact reason and the Director can retry or explicitly turn captions off).
    burn_requested = bool(asm.get("burn_captions"))
    burned, ass_path = "", ""
    if burn_requested:
        try:
            if not media._has_filter("subtitles"):
                raise RuntimeError("this ffmpeg build has no libass 'subtitles' filter — "
                                   "cannot burn captions (keep the SRT sidecar, or install "
                                   "a full ffmpeg build)")
            burned = final.replace(".mp4", ".captions.mp4")
            ass_path = os.path.join(out_dir, os.path.splitext(os.path.basename(final))[0] + ".ass")
            k_end = [starts[i + 1] if i + 1 < len(starts) else out["duration"]
                     for i in range(len(scenes))]
            windows = _caption_windows(scenes, starts, k_end)
            info_ass = cap_mod.build_ass(windows, cap_style, width, height, ass_path)
            cap_warnings.extend(info_ass.get("warnings") or [])
            media.burn_ass(final, ass_path, burned)
            if not os.path.exists(burned) or os.path.getsize(burned) < 1024:
                raise RuntimeError("libass burn produced no readable MP4")
            out["with_captions"] = burned
            out["ass"] = ass_path
            timing_mode = info_ass.get("timing", "sentence-window")
            notes.append(f"captions burned · preset '{cap_style.get('preset')}' · "
                         f"font {info_ass.get('font_family')} "
                         f"({cap_style.get('weight')}) · timing: {timing_mode}"
                         + (" (estimated, not forced alignment)" if info_ass.get("karaoke") else ""))
        except Exception as e:
            notes.append(f"caption burn-in FAILED: {str(e)[:220]}")
            raise RuntimeError(f"caption burn-in failed: {str(e)[:220]}") from e

    out["captions"] = {
        "requested": burn_requested,
        "burned": bool(burned and os.path.exists(burned)),
        "asset": burned or "",
        "primary": burned if (burned and os.path.exists(burned)) else final,
        "preset": cap_style.get("preset"),
        "font": cap_mod.font_family_name(cap_style["font"]),
        "font_file": os.path.basename(cap_mod.font_file(cap_style["font"], cap_style["weight"])),
        "renderer": "ffmpeg/libass (HarfBuzz shaping) — same path as the preview",
        "timing": ("estimated-proportional (not forced alignment)" if cap_style.get("karaoke")
                   else "sentence-window"),
        "karaoke": bool(cap_style.get("karaoke")),
        "warnings": cap_warnings,
    }
    poster = media.thumbnail(final, os.path.join(out_dir, os.path.splitext(
        os.path.basename(final))[0] + ".poster.png"), at_sec=0.4, width=min(360, width))
    if poster:
        out["poster"] = poster
    if asm.get("emit_manifest", True):
        manifest = _manifest(project, scenes, stage_assets, starts, out, cfg, run_id, notes)
        out["manifest"] = write_json(os.path.join(out_dir, os.path.splitext(
            os.path.basename(final))[0] + ".manifest.json"), manifest)
    return out


def _pad_tail(work, src, i, tail):
    """Freeze the clip's last frame for `tail` seconds (the line gap pause)."""
    dst = os.path.join(work, f"seg{i:02d}.pad.mp4")
    from ..util import run_ffmpeg
    from .. import media as media_mod

    vf = f"tpad=stop_mode=clone:stop_duration={float(tail):.3f},format=yuv420p"
    run_ffmpeg(["-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "22", "-pix_fmt", "yuv420p", "-an", dst], timeout=1800)
    return dst


def _black_slate(work, i, dur, width, height, fps):
    """Last-resort picture so a broken scene never loses the whole cut."""
    dst = os.path.join(work, f"seg{i:02d}.black.mp4")
    from ..util import run_ffmpeg

    run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={dur:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", dst],
               timeout=600)
    return dst


def _manifest(project, scenes, stage_assets, starts, out, cfg, run_id, notes):
    """Everything a re-run or an audit needs: prompts, engines, files, timings."""
    def slot(kind, idx):
        a = stage_assets.get(kind, {}).get(idx)
        if not a:
            return None
        return {"path": os.path.basename(a.get("path", "")), "duration": a.get("duration"),
                "engine": a.get("engine"), "meta": a.get("meta") or {}}

    return {
        "studio": "khmer-ai-content-studio",
        "version": 2,
        "generated_at": out.get("generated_at"),
        "run_id": run_id,
        "project": {"id": project.get("id"), "title": project.get("title"), "mode": project.get("mode"),
                    "script_origin": project.get("script_origin"), "language": project.get("language"),
                    "voice_profile_id": project.get("voice_profile_id"),
                    "content_type": project.get("content_type") or "explainer",
                    "character_id": project.get("character_id") or "",
                    "style_notes": project.get("style_notes"),
                    "target_duration": project.get("target_duration")},
        "pacing": {"line_gap_sec": out.get("line_gap_sec"),
                   "title_style": out.get("title_style"),
                   "subtitle_style": out.get("subtitle_style"),
                   "transition": out.get("transition"),
                   "transition_fade_sec": out.get("transition_fade_sec")},
        "captions": out.get("captions") or {},
        "video": {"path": os.path.basename(out.get("captions", {}).get("primary")
                                           or out.get("path", "")),
                  "uncaptioned_path": os.path.basename(out.get("path", "")),
                  "duration": out.get("duration"),
                  "width": out.get("width"), "height": out.get("height"), "fps": out.get("fps"),
                  "size_bytes": out.get("size_bytes")},
        "engines": {"voice": cfg.get("tts", {}).get("engine"), "timbre": cfg.get("rvc", {}).get("engine"),
                    "video": cfg.get("video", {}).get("engine"), "sfx": cfg.get("sfx", {}).get("engine")},
        "scenes": [{
            "idx": s.get("idx"), "start": round(starts[i], 3), "text": s.get("text"),
            "visual_prompt": s.get("visual_prompt"), "mood_tag": s.get("mood_tag"),
            "sfx_prompt": s.get("sfx_prompt"),
            "estimated_duration_sec": s.get("estimated_duration_sec"),
            "voice": slot("voice", i), "voice_final": slot("voice_final", i),
            "video": slot("video", i), "video_fit": slot("video_fit", i),
            "ambient": slot("ambient", i), "qa": slot("qa", i),
        } for i, s in enumerate(scenes)],
        "notes": notes,
    }
