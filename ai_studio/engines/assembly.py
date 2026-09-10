"""Stage 7 — Final Assembly (ffmpeg).

Inputs per scene: duration-matched silent video, converted voice wav, ambience
wav. Output: one MP4 (H.264 yuv420p + AAC, faststart), plus an SRT, a poster
frame and a `manifest.json` that records every scene, engine, prompt and file
that produced this cut — the "memory" the brief wants, exported with the video.

Assembly is deliberately forgiving about *pictures*: a scene with a missing clip
gets the previz renderer or a black slate with the voice over it, and the
manifest says which one happened, so the Director always gets a watchable cut
instead of a stack trace.

It is NOT forgiving about captions. If ``assembly.burn_captions`` is on and the
burn cannot be done correctly (no libass in ffmpeg, the bundled font is missing,
a scene has no timing), the stage fails with an actionable message — a request
for captions is never answered with a silent uncaptioned MP4.
"""
import os

from .. import khmer, media, previz
from ..util import (ensure_dir, jdump, media_duration, write_json)


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

    # optional rendered title card (assembly.title_style) — a silent intro clip
    title_dur = 0.0
    title_style = str(asm.get("title_style") or "")
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
        cursor += real_dur
        if progress:
            progress(100.0 * (i + 1) / total, f"scene {idx + 1}/{total}: {real_dur:.1f}s")

    if not seg_videos:
        raise RuntimeError("nothing to assemble — no scene has a picture")

    if progress:
        progress(72.0, "concatenating scenes")
    silent = os.path.join(out_dir, f"{project.get('id', 'project')}_{run_id or 'run'}.silent.mp4")
    transition = str(asm.get("transition", "crossfade"))
    fade = float(asm.get("fade_sec", 0.35)) if len(seg_videos) > 1 else 0.0
    media.concat_clips(seg_videos, silent, fps=fps, transition=transition, fade=fade,
                       work_dir=work)

    # ---- crossfade pulls the picture timeline forward; the audio must follow ----
    # xfade overlaps two clips by `fade` seconds, so an N-clip picture is
    # (N-1)*fade shorter than the sum of the per-scene durations the narration was
    # planned against. Leaving the voice delays at their pre-fade values is how
    # captions and narration drift apart on a crossfaded cut — so the scene starts
    # are recomputed from the real concatenated timeline whenever the transitions
    # actually shortened the picture. On a hard cut nothing moves.
    total_dur = media_duration(silent, cursor) or cursor
    shortened = max(0.0, cursor - total_dur)
    if shortened > 0.02 and starts:
        notes.append(
            f"picture is {shortened:.2f}s shorter than the scene sum because transitions "
            f"overlap ({transition}, {fade:.2f}s) — narration offsets shifted to stay in sync")
        new_starts = [max(0.0, starts[i] - i * (shortened / max(1, len(starts) - 1)))
                      for i in range(len(starts))]
        shift = [new_starts[i] - starts[i] for i in range(len(starts))]
        for tr in voice_tracks + amb_tracks:
            for i, sh in enumerate(shift):
                if abs(tr["delay"] - starts[i]) < 0.02:
                    tr["delay"] = round(max(0.0, tr["delay"] + sh), 4)
                    break
        starts = [round(x, 4) for x in new_starts]

    if progress:
        progress(84.0, "mixing narration + ambience")
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

    # ---------------------------------------------------------------- captions
    # One cue list feeds the SRT, the ASS, the burn and the manifest, so the
    # sidecar subtitle, the burned captions and the reported timings cannot drift
    # apart. `ends` are the real per-scene end times (starts[i+1] / total_dur) —
    # the previous writer ended the last cue at `start + 3s`, which cut the final
    # line short or left it on screen after the audio had stopped.
    from .. import caption_style as caption_style_mod
    from .. import captions as captions_mod

    cap_style = caption_style_mod.merge_styles(
        cfg.get("caption_style") or caption_style_mod.DEFAULT_STYLE,
        project.get("settings", {}).get("caption_style")
        if isinstance(project.get("settings"), dict) else None,
        asm.get("caption_style") if isinstance(asm.get("caption_style"), dict) else None)
    karaoke = karaoke_for(cfg, project)
    cap_texts = [khmer.display_text(s.get("text", "")) for s in scenes]
    cap_ends = [starts[i + 1] if i + 1 < len(starts) else total_dur
                for i in range(len(scenes))]
    cues = captions_mod.cues_from_scenes(cap_texts, starts, cap_ends, style=cap_style)
    if karaoke.get("enabled"):
        cues = _with_word_timing(cues, cap_texts, starts, cap_ends)
    render_meta = captions_mod.render_metadata(cues, cap_style, width, height, karaoke=karaoke)
    for w in render_meta["warnings"]:
        notes.append(f"caption note: {w}")
    for w in render_meta["bounds_warnings"]:
        notes.append(f"caption bounds: {w}")

    sub_style = str(asm.get("subtitle_style") or cap_style.get("preset") or "clean")
    out = {"path": final, "duration": media_duration(final, total_dur),
           "size_bytes": os.path.getsize(final), "width": width, "height": height,
           "fps": fps, "scenes": len(scenes), "notes": notes,
           "audio_peak_info": info, "line_gap_sec": gap, "title_style": title_style,
           "subtitle_style": sub_style, "caption_style": cap_style,
           "caption_metadata": render_meta}

    if asm.get("emit_srt", True):
        srt = os.path.join(out_dir, os.path.splitext(os.path.basename(final))[0] + ".srt")
        captions_mod.write_srt(cues, srt, style=cap_style, width=width, height=height)
        out["srt"] = srt
    if asm.get("burn_captions"):
        # No try/except here on purpose: a requested caption burn that cannot be
        # done must fail loudly (see the module docstring). The scheduler records
        # the error and the UI shows it; the uncaptioned MP4 stays available.
        burned = final.replace(".mp4", ".captions.mp4")
        ass = os.path.join(out_dir, os.path.splitext(os.path.basename(final))[0] + ".ass")
        captions_mod.write_ass(cues, cap_style, width, height, ass, karaoke=karaoke,
                               title=project.get("title") or "")
        captions_mod.burn(final, ass, burned, cap_style)
        out["ass"] = ass
        out["with_captions"] = burned
        out["with_captions_duration"] = media_duration(burned, total_dur)
        notes.append(
            f"captions burned: {caption_style_mod.FONT_FAMILIES[cap_style['font']]['family']} "
            f"{cap_style['weight']} · {cap_style['preset']} · "
            f"{captions_mod.probe_size(burned)['width']}x{captions_mod.probe_size(burned)['height']}"
            + (" · karaoke (proportional word timing, not forced alignment)"
               if karaoke.get("enabled") else ""))
    if out.get("ass"):
        out["ass"] = out["ass"]
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
                   "subtitle_style": out.get("subtitle_style")},
        "captions": out.get("caption_metadata"),
        "video": {"path": os.path.basename(out.get("path", "")), "duration": out.get("duration"),
                  "width": out.get("width"), "height": out.get("height"), "fps": out.get("fps"),
                  "size_bytes": out.get("size_bytes")},
        "engines": {"voice": cfg.get("tts", {}).get("engine"),
                    "voice_resolved": (cfg.get("tts", {}) or {}).get("_resolved_engine"),
                    "timbre": cfg.get("rvc", {}).get("engine"), "video": cfg.get("video", {}).get("engine"),
                    "sfx": cfg.get("sfx", {}).get("engine")},
        "assets": {"final": os.path.basename(out.get("path", "")),
                   "captioned": os.path.basename(out.get("with_captions", "")) or None,
                   "srt": os.path.basename(out.get("srt", "")) or None,
                   "ass": os.path.basename(out.get("ass", "")) or None},
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


def karaoke_for(cfg, project=None):
    """Is the karaoke highlight on, and in what colour? (off unless configured)

    Karaoke timing is a *proportional estimate* derived from the scene's own
    audio window and the syllable weights — the pipeline has no word-level
    timestamps. It is reported as such in the render metadata and the manifest;
    it is never presented as forced alignment.
    """
    src = {}
    if isinstance(project, dict) and isinstance(project.get("settings"), dict):
        src = {**(project["settings"].get("karaoke") or {})}
    src = {**(cfg.get("karaoke") or {}), **src}
    return {"enabled": bool(src.get("enabled", False)),
            "color": str(src.get("color") or "#FFD84D"),
            "timing": "proportional estimate from scene audio windows (not forced alignment)"}


def _with_word_timing(cues, texts, starts, ends):
    """Attach ``[(word, start, end)]`` to each cue — karaoke style, whole-line text.

    The word list is only used to place ``\k`` tags *inside* an already wrapped,
    cluster-safe line; the displayed text always comes from the original string,
    so punctuation, repeated-word signs (ៗ) and coeng stacks survive untouched.
    """
    windows = []
    for i, text in enumerate(texts):
        windows.append((float(starts[i]), float(ends[i]), text))
    out = []
    for cue in cues:
        words = media.words_for_timing(cue.get("text", ""))
        s0, s1 = float(cue.get("start", 0)), float(cue.get("end", 0))
        span = max(0.2, s1 - s0)
        total = sum(w for _t, w in words) or 1.0
        acc, timed = 0.0, []
        for word, weight in words:
            a = s0 + span * acc / total
            acc += weight
            b = s0 + span * acc / total
            timed.append((word, round(a, 3), round(b, 3)))
        out.append({**cue, "words": timed})
    return out
