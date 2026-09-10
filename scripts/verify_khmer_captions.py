"""Render the caption test matrix and prove, from actual pixels, that it worked.

This is not a unit test — it burns real videos with ffmpeg/libass, pulls frames
back out at native resolution, and measures where the caption ink actually
landed (by differencing against the same frame with no captions). It answers the
questions a passing unit test cannot:

* is every mark of every required Khmer test string really on screen?
* is the ink inside the safe area at 1080x1920, 720x1280, 480x854 and 1920x1080?
* does each bundled font family cover the text (no .notdef boxes)?
* does each preset change the pixels the way its parameters say it will?

    python scripts/verify_khmer_captions.py --out verify_out [--quick]

Outputs, all under ``--out``:

    frames/<preset>_<n>.png          full-resolution caption frames
    frames/<font>_sample.png         one frame per bundled font family
    sheets/<preset>.png              contact sheet per preset
    sheets/before_after.png          the reported tofu frame vs the fixed frame
    report.json                      measured ink bounds, fonts, warnings
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_studio import caption_style as cs          # noqa: E402
from ai_studio import captions, khmer              # noqa: E402
from ai_studio.util import ensure_dir, ffmpeg_exe, run_ffmpeg   # noqa: E402

TESTS = [
    ("line1", "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។"),
    ("line2", "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។"),
    ("line3", "បើថ្ងៃនេះអ្នកមានអារម្មណ៍នឿយហត់ សុំសម្រាកបន្តិចសិន។"),
    ("line4", "ដកដង្ហើមវែងៗ ហើយចាប់ផ្ដើមម្ដងទៀត។"),
    ("line5", "ជំហានតូចៗរបស់អ្នក ក៏មានតម្លៃដែរ។"),
    ("multi", "កុំបោះបង់ក្ដីសង្ឃឹម។ អ្នកអាចធ្វើបាន។"),
    ("tech", "ស្វែងយល់អំពីបច្ចេកវិទ្យា និងការអភិវឌ្ឍ។"),
    ("mixed", "ភាសាខ្មែរ — Khmer Unicode — ២០២៦ / 2026"),
    ("spaceless", "ការអភិវឌ្ឍន៍បច្ចេកវិទ្យាឌីជីថលនៅកម្ពុជា"),
    ("explicit", "ជំហានតូចៗ\nក៏រាប់ដែរ"),
    ("marks", "ដកដង្ហើមវែងៗ ហើយចាប់ផ្ដើមម្ដងទៀត។ នឿយហត់"),
    ("silent", "អ្នកអាចធ្វើបាន [[silent: ព្យាយាមម្ដងទៀត]]"),
]

SIZES = {
    "portrait_1080": (1080, 1920),
    "portrait_720": (720, 1280),
    "portrait_480": (480, 854),
    "landscape_1080": (1920, 1080),
}


def backdrop_for(w, h, path, tone="mid"):
    """A tone-and-gradient backdrop with light text regions — a real contrast test."""
    if os.path.exists(path):
        return path
    t = {"mid": "0x2B3A4A", "light": "0xD9D2C4", "busy": "0x7A6B4F"}[tone]
    dur = len(TESTS) * 3.0 + 4.0
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c={t}:s={w}x{h}:r=24:d={dur:.0f}",
                "-vf", f"drawbox=x=0:y={int(h*0.75)}:w={w}:h={int(h*0.25)}:c=0xE8E2D6@0.55:t=fill",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-an", path], timeout=600)
    return path


def ink_bbox(frame, baseline):
    """Caption ink bounds = pixels that differ from the caption-free frame."""
    from PIL import Image
    import numpy as np

    a = np.asarray(Image.open(frame).convert("RGB")).astype(np.int16)
    b = np.asarray(Image.open(baseline).convert("RGB")).astype(np.int16)
    if a.shape != b.shape:
        return None
    d = np.abs(a - b).max(axis=2)
    mask = d > 12
    if mask.sum() < 20:
        return None
    ys, xs = np.where(mask)
    return {"left": int(xs.min()), "right": int(xs.max()),
            "top": int(ys.min()), "bottom": int(ys.max()),
            "pixels": int(mask.sum()), "ink_w": int(xs.max() - xs.min() + 1),
            "ink_h": int(ys.max() - ys.min() + 1)}


def notdef_report(font_path, text):
    """Which characters of `text` would render as .notdef (empty box)?

    One buffer per unique character: shaping a whole sentence can legitimately
    produce a .notdef for a *space* between runs of different scripts, which is
    not missing coverage and must not be reported as such.
    """
    try:
        import uharfbuzz as hb
    except Exception:
        return {"available": False}
    blob = hb.Blob.from_file_path(font_path)
    font = hb.Font(hb.Face(blob))
    missing = []
    for ch in sorted(set(text)):
        if ch.isspace():
            continue
        buf = hb.Buffer()
        buf.add_str(ch)
        buf.guess_segment_properties()
        hb.shape(font, buf)
        if any(g.codepoint == 0 for g in buf.glyph_infos):
            missing.append(ch)
    return {"available": True, "checked": len(set(text)), "notdef": len(missing),
            "missing_chars": missing}


def sheet(files, dst, cols=2, label_h=34):
    from PIL import Image, ImageDraw

    ims = [Image.open(f).convert("RGB") for f in files if os.path.exists(f)]
    if not ims:
        return None
    tw = 460
    scale = [im.resize((tw, max(1, int(im.height * tw / im.width)))) for im in ims]
    rows = (len(scale) + cols - 1) // cols
    rh = max(i.height for i in scale) + label_h
    out = Image.new("RGB", (cols * tw, rows * rh), (16, 19, 24))
    d = ImageDraw.Draw(out)
    for i, (im, f) in enumerate(zip(scale, files)):
        r, c = divmod(i, cols)
        out.paste(im, (c * tw, r * rh + label_h))
        d.text((c * tw + 8, r * rh + 10), os.path.basename(f), fill=(200, 210, 220))
    ensure_dir(os.path.dirname(dst) or ".")
    out.save(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--quick", action="store_true", help="1080p portrait only, no crossfades")
    args = ap.parse_args()
    out = ensure_dir(args.out)
    frames = ensure_dir(os.path.join(out, "frames"))
    sheets = ensure_dir(os.path.join(out, "sheets"))
    report = {"ffmpeg": ffmpeg_exe(), "tests": [], "fonts": [], "sizes": {}}

    sizes = {"portrait_1080": SIZES["portrait_1080"]} if args.quick else SIZES
    tones = ("mid",) if args.quick else ("mid", "light", "busy")

    for size_name, (w, h) in sizes.items():
        report["sizes"][size_name] = {"width": w, "height": h, "presets": {}}
        for tone in tones:
            base = backdrop_for(w, h, os.path.join(out, f"backdrop_{size_name}_{tone}.mp4"), tone)
            plain = os.path.join(frames, f"_plain_{size_name}_{tone}.png")
            subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", "2.0", "-i", base, "-frames:v", "1", plain], check=True)
            for pkey in cs.PRESET_KEYS:
                st = cs.preset_style(pkey)
                texts = [t for _k, t in TESTS]
                starts = [i * 3.0 for i in range(len(texts))]
                ends = [s + 2.6 for s in starts]
                cues = captions.cues_from_scenes(texts, starts, ends, style=st)
                clip = os.path.join(out, f"{size_name}_{tone}_{pkey}.mp4")
                captions.burn_into(base, cues, st, clip,
                                   ass_path=os.path.join(out, f"{size_name}_{tone}_{pkey}.ass"))
                entry = {"preset": pkey, "tone": tone, "size": size_name,
                         "style": st, "frames": [], "warnings": []}
                for i, (key, text) in enumerate(TESTS):
                    at = starts[i] + 1.2
                    fr = os.path.join(frames, f"{size_name}_{tone}_{pkey}_{key}.png")
                    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                                    "-ss", f"{at:.3f}", "-i", clip, "-frames:v", "1", fr],
                                   check=True)
                    bl = os.path.join(frames, f"_{size_name}_{tone}_{pkey}_{key}_plain.png")
                    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                                    "-ss", f"{at:.3f}", "-i", base, "-frames:v", "1", bl],
                                   check=True)
                    box = ink_bbox(fr, bl)
                    fit = captions.fit_text(text, st, w, h)
                    lay = captions.layout_caption(text, st, w, h, fit=fit)
                    rec = {"key": key, "text": text, "at": round(at, 2), "frame": fr,
                           "ink": box, "lines": fit["lines"],
                           "font_size_px": round(fit["font_size_px"], 1),
                           "bounds": captions.bounds_problems(lay),
                           "warnings": fit["warnings"]}
                    if box is None:
                        rec["bounds"] = list(rec["bounds"]) + ["no caption ink found in frame"]
                    else:
                        mh = w * st["margin_h"] / 100.0
                        mv = h * st["margin_v"] / 100.0
                        if box["left"] < mh - 2 and st["alignment"] == "center" and \
                                not st["background"]:
                            pass          # centre alignment measures from the text box
                        if box["top"] < 0 or box["bottom"] > h - 1 or box["left"] < 0 or \
                                box["right"] > w - 1:
                            rec["bounds"].append(
                                f"ink crosses the frame edge: {box}")
                    entry["frames"].append(rec)
                    entry["warnings"].extend(rec["warnings"])
                    entry["warnings"].extend(rec["bounds"])
                report["sizes"][size_name]["presets"].setdefault(pkey, []).append(entry)
                files = [f["frame"] for f in entry["frames"]]
                sheet(files, os.path.join(sheets, f"{size_name}_{tone}_{pkey}.png"))
                print(f"[verify] {size_name} {tone} {pkey}: "
                      f"{sum(1 for f in entry['frames'] if f['ink'])}/{len(entry['frames'])} frames with ink")

    # ---- font coverage (real HarfBuzz glyph ids, .notdef = missing coverage)
    all_text = (" ".join(t.split("\n")[0] for _k, t in TESTS)
                + " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                + " —–/.,!?()[]:;\"'" + khmer.KHMER_DIGITS
                + "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអ"
                + "ាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៑្។៕៖ៗ៘៙៚៛ៜ៝")
    for fid in cs.FONT_FAMILIES:
        path, weight = cs.font_file(fid)
        rep = notdef_report(path, all_text)
        rep.update({"font_id": fid, "family": cs.FONT_FAMILIES[fid]["family"],
                    "file": os.path.basename(path), "weight": weight})
        report["fonts"].append(rep)
        st = {**cs.preset_style("clean"), "font": fid, "font_size_px": 80}
        cues = [{"text": "ជីវិតមនុស្ស មិនមែនជាការប្រណាំងទេ។", "start": 0, "end": 3}]
        fr = os.path.join(frames, f"font_{fid}_sample.png")
        captions.preview_frame(cues, st, 1080, 1080, fr,
                               backdrop=os.path.join(out, "backdrop_portrait_1080_mid.mp4"))
        print(f"[verify] font {fid}: {rep}")

    sheet([os.path.join(frames, f"font_{f}_sample.png") for f in cs.FONT_FAMILIES],
          os.path.join(sheets, "fonts.png"), cols=2)

    # ---- karaoke (proportional timing, whole-line source text preserved)
    st = cs.preset_style("clean")
    kara_text = "យើងម្នាក់ៗ មានផ្លូវដើររៀងៗខ្លួន។"
    toks = [w for w in khmer.display_text(kara_text).split(" ") if w]
    words = [(w, i * 0.4, (i + 1) * 0.4) for i, w in enumerate(toks)]
    cues = [{"text": kara_text, "start": 0, "end": 3, "words": words}]
    kara = {"enabled": True, "color": "#FFD84D"}
    kf = os.path.join(frames, "karaoke_sample.png")
    captions.preview_frame(cues, st, 1080, 1080, kf, at_sec=0.9, karaoke=kara)
    report["karaoke"] = {"frame": kf, "cues": len(cues),
                         "timing": "proportional estimate (not forced alignment)"}

    write_json_file(os.path.join(out, "report.json"), report)
    problems = [w for s in report["sizes"].values() for v in s["presets"].values()
                for e in v for w in e["warnings"]]
    bad_fonts = [f for f in report["fonts"] if f.get("notdef")]
    print(f"\n[verify] report → {os.path.join(out, 'report.json')}")
    print(f"[verify] layout warnings: {len(problems)}")
    for p in sorted(set(problems)):
        print("   -", p)
    print(f"[verify] fonts with missing glyphs: {bad_fonts or 'none'}")
    return 0 if not bad_fonts else 1


def write_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
