"""The renderer: composes scenes (background + animated character + captions),
blends transitions, mixes narration + SFX audio, and writes the final MP4 + SRT.

Everything runs locally: numpy/cv2 for frames, moviepy+ffmpeg for the final
encode. Each stage reports progress via the callback (the UI polls it).
"""
import math
import os
import shutil
import wave
import numpy as np

import cv2

from .animation import (ENTRY_DUR, EXIT_DUR, composite_rgba, entry_state,
                        exit_state, idle_bob, load_character_rgba, talk_pulse)
from .sfx import load_sfx
from .transitions import blend
from .planner import BG_COLORS
from .voice import audio_envelope, wav_duration
from .khmer_subtitles import render_caption_frame

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------- captions -------------------------------
def estimate_word_timings(text, clip_duration):
    words = (text or "").strip().split()
    if not words or clip_duration <= 0:
        return []
    total_chars = sum(len(w) for w in words)
    timings = []
    cur = 0.0
    for w in words:
        weight = len(w) / total_chars if total_chars else 1.0 / len(words)
        dur = max(0.15, min(1.4, clip_duration * weight))
        timings.append({"word": w, "start": cur, "end": cur + dur})
        cur += dur
    last = timings[-1]["end"]
    scale = clip_duration / last if last > 0 else 1.0
    for t in timings:
        t["start"] = round(t["start"] * scale, 3)
        t["end"] = round(t["end"] * scale, 3)
    return timings


def draw_captions(frame, words_timing, t, title=None, title_until=2.0, template_key="classic_yellow"):
    """Renders captions using selected subtitle template."""
    h, w = frame.shape[:2]
    if title and t < title_until:
        _draw_center_text(frame, title, int(h * 0.10), w, scale=1.4,
                          color=(255, 255, 255), thickness=4, shadow=True)
    return render_caption_frame(frame, words_timing, t, template_key=template_key, title=title)


def _draw_center_text(frame, text, y, w, scale=1.0, color=(255, 255, 255), thickness=3, shadow=True):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = int((w - tw) / 2)
    if shadow:
        cv2.putText(frame, text, (x + 3, y + 3), FONT, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _wrap(text, max_chars=26):
    words = text.split()
    lines, cur = [], ""
    for wd in words:
        if cur and len(cur) + 1 + len(wd) > max_chars:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines


def write_srt(scene_word_timings, scene_starts, path):
    """scene_word_timings: list of per-scene word timing lists (scene-local)."""
    def fmt(sec):
        hh = int(sec // 3600)
        mm = int((sec % 3600) // 60)
        ss = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    idx = 1
    with open(path, "w", encoding="utf-8") as f:
        for si, wt in enumerate(scene_word_timings):
            base = scene_starts[si]
            # group into 3-word phrases
            for j in range(0, len(wt), 3):
                group = wt[j:j + 3]
                s = base + group[0]["start"]
                e = base + group[-1]["end"]
                text = " ".join(g["word"] for g in group)
                f.write(f"{idx}\n{fmt(s)} --> {fmt(e)}\n{text}\n\n")
                idx += 1


# ------------------------------ backgrounds ------------------------------
def make_background(bg_name, w, h, seed=0):
    top, bottom = BG_COLORS.get(bg_name, BG_COLORS["gradient-violet"])
    if top != bottom:
        grad = np.zeros((h, 1, 3), dtype=np.float32)
        for y in range(h):
            p = y / max(1, h - 1)
            grad[y, 0] = [top[i] * (1 - p) + bottom[i] * p for i in range(3)]
        bg = np.repeat(grad, w, axis=1)
    else:
        bg = np.full((h, w, 3), top, dtype=np.float32)
    if bg_name in ("pattern-dots", "pattern-grid"):
        rng = np.random.default_rng(seed)
        bg = bg.astype(np.uint8).copy()
        if bg_name == "pattern-dots":
            for _ in range(70):
                cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
                r = int(rng.integers(2, 5))
                cv2.circle(bg, (cx, cy), r, (70, 70, 90), -1)
        else:
            step = max(40, w // 16)
            for x in range(0, w, step):
                cv2.line(bg, (x, 0), (x, h), (60, 60, 84), 1)
            for y in range(0, h, step):
                cv2.line(bg, (0, y), (w, y), (60, 60, 84), 1)
        return bg
    return bg.astype(np.uint8)


def make_particles(w, h, seed=1, count=26):
    rng = np.random.default_rng(seed)
    return [{
        "x": float(rng.uniform(0, w)),
        "y": float(rng.uniform(0, h)),
        "r": float(rng.uniform(2, 6)),
        "v": float(rng.uniform(8, 30)),
        "a": int(rng.integers(40, 110)),
        "c": tuple(int(v) for v in rng.integers(120, 220, size=3)),
    } for _ in range(count)]


def _composite_supporting_image(frame, supp_img, position="top"):
    if supp_img is None:
        return frame
    h, w = frame.shape[:2]

    if position == "behind":
        scaled = cv2.resize(supp_img, (w, h), interpolation=cv2.INTER_AREA)
        dark = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.addWeighted(scaled, 0.45, dark, 0.55, 0, frame)
        return frame

    if position == "side":
        card_w, card_h = int(w * 0.52), int(h * 0.42)
        x0, y0 = int(w * 0.05), int(h * 0.16)
    elif position == "pip":
        card_w, card_h = int(w * 0.46), int(h * 0.32)
        x0, y0 = int(w * 0.48), int(h * 0.14)
    else:  # "top"
        card_w, card_h = int(w * 0.80), int(h * 0.36)
        x0, y0 = int((w - card_w) / 2), int(h * 0.12)

    x1, y1 = x0 + card_w, y0 + card_h
    resized = cv2.resize(supp_img, (card_w, card_h), interpolation=cv2.INTER_AREA)

    # Draw card border & place resized image
    cv2.rectangle(frame, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (20, 20, 30), -1)
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (255, 255, 255), 2)
    frame[y0:y1, x0:x1] = resized

    return frame


def draw_particles(bg, particles, t):
    for p in particles:
        y = (p["y"] - t * p["v"]) % (bg.shape[0] + 20) - 10
        cv2.circle(bg, (int(p["x"]), int(y)), int(p["r"]),
                   (p["c"][0], p["c"][1], p["c"][2]), -1, lineType=cv2.LINE_AA)
    return bg


# ------------------------------ audio mixing ------------------------------
def _mix_audio(scene_audios, total_dur, sr=44100):
    """scene_audios: [{ 'narration': (samples, sr) or None, 'sfx': (samples, sr) or None, 'start': sec }]"""
    mix = np.zeros(int(total_dur * sr), dtype=np.float32)
    for sa in scene_audios:
        start = max(0.0, sa["start"])
        if sa.get("narration") is not None:
            samples, nsr = sa["narration"]
            samples = _resample_to(samples, nsr, sr)
            off = int(start * sr)
            end = min(len(mix), off + len(samples))
            if off < len(mix):
                mix[off:end] += samples[: end - off] * 1.0
        if sa.get("sfx") is not None:
            samples, nsr = sa["sfx"]
            samples = _resample_to(samples, nsr, sr)
            off = int((start + sa.get("sfx_time", 0.0)) * sr)
            end = min(len(mix), off + len(samples))
            if off < len(mix):
                mix[off:end] += samples[: end - off] * 0.85
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix = mix / peak * 0.95
    return mix


def _resample_to(samples, src_sr, dst_sr):
    if src_sr == dst_sr:
        return np.asarray(samples, dtype=np.float32)
    x = np.asarray(samples, dtype=np.float32)
    n_out = int(len(x) * dst_sr / src_sr)
    if n_out <= 0:
        return x[:0]
    pos = np.linspace(0, len(x) - 1, n_out)
    idx = np.floor(pos).astype(int)
    frac = pos - idx
    idx2 = np.minimum(idx + 1, len(x) - 1)
    return (x[idx] * (1 - frac) + x[idx2] * frac).astype(np.float32)


def _write_wav(path, samples, sr):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# -------------------------------- renderer --------------------------------
class Renderer:
    def __init__(self, sfx_dir, work_dir):
        self.sfx_dir = sfx_dir
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def render(self, plan, character_dir, tts, voice_cfg, width=720, height=1280,
               fps=24, out_dir="outputs", progress=None, subtitle_template="classic_yellow"):
        """Runs the full render. Returns result dict with file names."""
        def prog(stage, pct):
            if progress:
                progress(stage, pct)

        os.makedirs(out_dir, exist_ok=True)
        char_rgba_default = load_character_rgba(os.path.join(character_dir, "avatar.png"))
        scenes = plan["scenes"]
        mode = plan.get("mode", "standard")
        sfx_enabled = plan.get("sfx_enabled", True)

        # ---------- stage 1: narration (TTS) ----------
        prog("voice", 3)
        clone_wav = None
        if voice_cfg and voice_cfg.get("voice_id"):
            clone_wav = os.path.join(voice_cfg.get("voices_root", "voices"),
                                     voice_cfg["voice_id"], "recording.wav")
            if not os.path.exists(clone_wav):
                clone_wav = None
        scene_audios = []
        word_timings_all = []
        durations = []
        for i, sc in enumerate(scenes):
            prog(f"voice scene {i + 1}/{len(scenes)}", 3 + int(17 * i / len(scenes)))
            narr_wav = os.path.join(self.work_dir, f"narr_{i}.wav")
            ok, engine = tts.speak(sc["script"], narr_wav, clone_voice_wav=clone_wav,
                                   kokoro_voice=voice_cfg.get("kokoro_voice", "af_bella") if voice_cfg else "af_bella")
            from .voice import audio_envelope, wav_duration
            if ok and os.path.exists(narr_wav):
                narr_dur = wav_duration(narr_wav)
                env = audio_envelope(narr_wav, fps)
            else:
                narr_dur = 0.0
                env = None
                if engine == "none":
                    print(f"Scene {i + 1}: no TTS engine available — scene renders without narration.")
            dur = max(3.0, min(12.5, max(narr_dur + 0.5, sc.get("duration", 4.0))))
            durations.append(round(dur, 3))
            wt = estimate_word_timings(sc["script"], dur if not narr_dur else min(dur, narr_dur + 0.3))
            word_timings_all.append(wt)
            sfx_samples, sfx_sr = (None, 0)
            if sfx_enabled and sc.get("sfx", "none") != "none":
                sfx_samples, sfx_sr = load_sfx(self.sfx_dir, sc["sfx"])
            scene_audios.append({"narration": None, "sfx": None, "start": 0.0,
                                 "_narr_wav": narr_wav if ok else None,
                                 "_sfx": (sfx_samples, sfx_sr) if sfx_samples is not None else None})
            if ok and os.path.exists(narr_wav):
                with wave.open(narr_wav, "rb") as wf:
                    nsr = wf.getframerate()
                    n = wf.getnframes()
                    raw = wf.readframes(n)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if wf.getnchannels() > 1:
                    data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
                scene_audios[i]["narration"] = (data, nsr)
                scene_audios[i]["_env"] = env
            else:
                scene_audios[i]["_env"] = None
            if os.path.exists(narr_wav) and not ok:
                os.remove(narr_wav)

        # scene start times accounting for transition overlap
        overlap = 0.5
        starts = []
        acc = 0.0
        for i, d in enumerate(durations):
            starts.append(acc)
            acc += d - (overlap if i < len(durations) - 1 else 0.0)
        total_dur = acc + 0.4
        for i in range(len(scene_audios)):
            scene_audios[i]["start"] = starts[i]
            scene_audios[i]["sfx"] = scene_audios[i].pop("_sfx")
            scene_audios[i].pop("_narr_wav", None)

        # ---------- stage 2: frames per scene ----------
        scene_files = []
        for i, sc in enumerate(scenes):
            prog(f"render scene {i + 1}/{len(scenes)}", 20 + int(55 * i / len(scenes)))
            n_frames = max(8, int(durations[i] * fps))
            temp = os.path.join(self.work_dir, f"scene_{i}.mp4")
            scene_files.append(temp)
            writer = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            bg = make_background(sc["background"], width, height, seed=i + 1)
            particles = make_particles(width, height, seed=i * 7 + 3)
            env = scene_audios[i].get("_env")

            # Load pose image or default character
            pose_name = sc.get("pose", "idle")
            actions_dir = os.path.join(character_dir, "actions")
            pose_path = os.path.join(actions_dir, f"{pose_name}.png")
            if os.path.exists(pose_path):
                char_rgba = load_character_rgba(pose_path)
            else:
                char_rgba = char_rgba_default

            # Supporting image loading
            supp_img = None
            img_cfg = sc.get("image", {})
            if img_cfg and img_cfg.get("needed"):
                img_path = img_cfg.get("local_path")
                if not img_path or not os.path.exists(img_path):
                    url = img_cfg.get("url", "")
                    if url.startswith("/assets/cache/"):
                        img_path = os.path.join(self.work_dir, url.replace("/assets/cache/", ""))
                    elif url.startswith("/assets/characters/"):
                        parts = url.split("/")
                        if len(parts) >= 4:
                            img_path = os.path.join(os.path.dirname(character_dir), *parts[3:])
                if img_path and os.path.exists(img_path):
                    raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if raw is not None:
                        supp_img = raw

            # Character size & anchor position
            if mode == "explain":
                if supp_img is not None:
                    char_h = int(height * 0.46)
                    cx = int(width * 0.72)
                    cy = int(height * 0.68)
                else:
                    char_h = int(height * 0.52)
                    cx = width // 2
                    cy = int(height * 0.62)
            else:
                char_h = int(height * 0.52)
                cx = width // 2
                cy = int(height * 0.60)

            char_w = int(char_h * char_rgba.shape[1] / max(1, char_rgba.shape[0]))
            char_base = cv2.resize(char_rgba, (char_w, char_h), interpolation=cv2.INTER_AREA)

            for f in range(n_frames):
                t = f / fps
                frame = bg.copy()
                draw_particles(frame, particles, t)

                # Draw supporting image in Explain Mode
                if mode == "explain" and supp_img is not None:
                    pos = img_cfg.get("position", "top")
                    frame = _composite_supporting_image(frame, supp_img, pos)

                # Transform & character compositing
                if mode == "explain":
                    # Explain mode presenter stays planted, subtle talk-pulse and idle bob
                    scale, ox, oy, alpha = 1.0, 0.0, idle_bob(t + i) * 0.3, 1.0
                else:
                    if t < ENTRY_DUR:
                        scale, ox, oy, alpha = entry_state(sc["animation"], t)
                    elif t > durations[i] - EXIT_DUR and i == len(scenes) - 1:
                        scale, ox, oy, alpha = exit_state("fade-out", t - (durations[i] - EXIT_DUR))
                    else:
                        scale, ox, oy, alpha = 1.0, 0.0, idle_bob(t + i), 1.0

                if env is not None:
                    ei = min(int(t * fps), len(env) - 1)
                    scale *= talk_pulse(float(env[ei]))

                frame = composite_rgba(frame, char_base, cx, cy, scale, ox, oy, alpha)

                # Captions with chosen template
                frame = draw_captions(
                    frame, word_timings_all[i], t,
                    title=plan.get("title") if i == 0 else None, title_until=2.0,
                    template_key=subtitle_template
                )
                writer.write(frame)
            writer.release()

        # ---------- stage 3: transitions + silent master ----------
        prog("transitions", 78)
        master = os.path.join(self.work_dir, "master_silent.mp4")
        master_writer = cv2.VideoWriter(master, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        readers = [cv2.VideoCapture(p) for p in scene_files]
        for i, cap in enumerate(readers):
            first = None
            ret, fr = cap.read()
            if ret:
                first = fr
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if i > 0 and scenes[i - 1].get("transition", "fade") != "cut":
                # previous scene's last frame already written; blend window
                kind = scenes[i - 1]["transition"]
                n_blend = int(overlap * fps)
                prev_cap = readers[i - 1]
                prev_count = int(prev_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prev_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, prev_count - 1 - n_blend))
                blend_a = None
                ret2, fr2 = prev_cap.read()
                if ret2:
                    blend_a = fr2
                idx = 0
                while True:
                    ret3, inc = cap.read()
                    if not ret3:
                        break
                    if blend_a is not None and first is not None:
                        p = (idx + 1) / (n_blend + 1)
                        master_writer.write(blend(blend_a, inc, p, kind))
                    else:
                        master_writer.write(inc)
                    idx += 1
                    if idx >= n_blend and blend_a is not None:
                        break
                # continue writing the rest of this scene
                while True:
                    ret4, inc = cap.read()
                    if not ret4:
                        break
                    master_writer.write(inc)
            else:
                while True:
                    ret5, fr = cap.read()
                    if not ret5:
                        break
                    master_writer.write(fr)
        for cap in readers:
            cap.release()
        master_writer.release()

        # ---------- stage 4: audio mix + final encode ----------
        prog("audio mix", 86)
        mix = _mix_audio(scene_audios, total_dur)
        mix_wav = os.path.join(self.work_dir, "mix.wav")
        _write_wav(mix_wav, mix, 44100)

        prog("final encode", 90)
        from moviepy import VideoFileClip, AudioFileClip
        final_mp4 = os.path.join(out_dir, "video.mp4")
        final_srt = os.path.join(out_dir, "subtitles.srt")
        write_srt(word_timings_all, starts, final_srt)
        with VideoFileClip(master) as mc:
            mc = mc.with_audio(AudioFileClip(mix_wav))
            mc.write_videofile(final_mp4, codec="libx264", audio_codec="aac",
                               ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)
        for p in scene_files + [master]:
            try:
                os.remove(p)
            except OSError:
                pass
        prog("done", 100)
        return {
            "mp4": os.path.basename(final_mp4),
            "srt": os.path.basename(final_srt),
            "duration": round(total_dur, 2),
            "size": [width, height],
            "scenes": len(scenes),
        }
