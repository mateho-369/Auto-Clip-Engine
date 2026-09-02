"""The fixed content style guideline (Section 5 of the brief).

Single source of truth: Mode B script generation, Stage 1 scene tagging, the SFX
prompt writer and the QA reviewer all quote *this* text, so the house voice can
never drift between agents. The Director can extend (not replace) it per project
with ``project.style_notes``.
"""

HOUSE_VOICE = (
    "the voice of a caring older sibling: calm, warm, gentle, unhurried"
)

STYLE_GUIDELINE = """KHMER CONTENT STYLE GUIDELINE (fixed house voice)
- Language: Khmer (ភាសាខ្មែរ), natural spoken register — not textbook/literary.
- Tone: calm, warm, positive, life-affirming. Explain one idea kindly, as if a
  caring older sibling were talking to a younger sibling at 6am on a porch.
- Message: do not give up. Struggle is temporary and instructive; self-blame is
  not required to improve. Never shame, never fear-monger, never diagnose, never
  promise money/cures, never mention politics or religion in a divisive way.
- Register: simple words, short sentences, concrete images. 1 idea per sentence.
  Use ។ as the sentence finisher. No emoji, no exclamation chains, no hashtags,
  no sales language, no "subscribe/follow" begging (a gentle closing wish is enough).
- Structure for a short: (1) a soft, curious opening, (2) name the feeling/struggle,
  (3) reframe it with one simple image from nature or daily life, (4) one small
  practical step, (5) a warm blessing-like closing line.
- Imagery: peaceful and natural — soft light, morning mist, slow water, rice fields,
  birds, lotus, a candle, a lantern, a winding path, leaves moving in the wind,
  distant mountains, gentle rain on a tin roof, a warm kitchen.
- Sound: quiet ambience (birds, water, wind in leaves), no stingers, no whooshes,
  no dramatic risers, no music that pushes emotion.
"""

# Scene budget defaults for mechanical segmentation (seconds of speech per scene)
SCENE_TARGET_SECONDS = 6.0
SCENE_MIN_SECONDS = 3.5
SCENE_MAX_SECONDS = 11.0
SCENE_MAX_CHARS = 190          # per-scene character budget (Khmer is dense)

# --------------------------------------------------------------- imagery map
# Keyword (Khmer or latin) -> (visual prompt fragment, mood tag, ambience text).
# Stage 1 asks the LLM for these; when Ollama is offline the deterministic
# segmenter uses this table so every scene still gets a sane visual prompt.
KEYWORD_IMAGERY = [
    (("ទឹក", "river", "ទន្លេ", "ស្ទឹង", "water"),
     "slow river at sunrise, soft golden light on calm water, mist drifting",
     "water-calm", "gentle flowing water, distant birds"),
    (("ភ្លៀង", "rain", "ភ្លៀងធ្លាក់"),
     "gentle rain on a tin roof, green leaves wet, grey soft light, cozy window view",
     "rain-soft", "soft steady rain, quiet drops on leaves"),
    (("បក្សី", "bird", "សត្វបក្សី"),
     "small birds over a rice field at dawn, wide sky, warm low sun",
     "birds-dawn", "gentle birds chirping, light breeze"),
    (("ផ្កា", "lotus", "ផ្កាឈូក", "flower"),
     "lotus flowers opening on still pond water, pastel morning colours",
     "flowers-still", "light wind, soft water lapping"),
    (("ភ្នំ", "ព្រៃ", "mountain", "forest", "ដើមឈើ", "tree"),
     "misty mountains at first light, tall trees, slow drifting fog",
     "forest-mist", "forest ambience, birds far away, leaves rustling"),
    (("ថ្ងៃ", "ព្រឹក", "sunrise", "morning", "ព្រះអាទិត្យ", "sun"),
     "sunrise over rice fields, warm haze, long soft shadows",
     "sunrise-warm", "morning birds, calm breeze"),
    (("យប់", "ផ្កាយ", "night", "star", "ព្រះចន្ទ", "moon"),
     "quiet village night, lantern glow, stars above a dark field",
     "night-quiet", "crickets, very soft night air"),
    (("ផ្លូវ", "journey", "path", "ដំណើរ"),
     "a long earthen path through green fields, walking pace, soft dust in sunlight",
     "path-walking", "footsteps on dry earth, light wind"),
    (("គ្រួសារ", "មាតា", "បិតា", "family", "កូន", "mother", "កុមារ"),
     "warm kitchen light, hands pouring tea, simple home, soft focus",
     "home-warm", "quiet room, distant kettle, soft chair creak"),
    (("សាលា", "សិស្ស", "school", "study", "សៀវភៅ", "book", "ការរៀន"),
     "an open notebook on a wooden desk, morning light through a window",
     "study-calm", "page turning, quiet room tone"),
    (("ក្ដីសុខ", "សុខ", "peace", "សន្តិភាព", "ស្ងប់"),
     "still lake reflecting clouds, one small boat, enormous quiet",
     "still-lake", "water lapping softly, far-off birds"),
    (("កម្លាំង", "ការព្យាយាម", "strength", "effort", "ប្រឹង"),
     "hands tying a cloth, someone climbing a gentle slope at dawn, effort and breath",
     "effort-dawn", "breath, wind, footsteps on earth"),
    (("អំពើល្អ", "ចិត្ត", "kind", "សេចក្ដីស្រឡាញ់", "love", "មិត្តភាព"),
     "two people sharing an umbrella, warm light, soft smiles, slow motion",
     "kind-warm", "rain lightly, quiet street"),
]

DEFAULT_VISUAL = ("soft natural light over calm tropical scenery, slow gentle camera motion, "
                  "warm peaceful mood, no text, no faces in close-up")
DEFAULT_MOOD = "calm-warm"
DEFAULT_AMBIENCE = "gentle birds chirping, soft flowing water, peaceful morning ambience"
DEFAULT_NEGATIVE = ("text, watermark, subtitles, letters, logos, hands with six fingers, "
                    "extra limbs, deformed face, flicker, fast cuts, camera shake, "
                    "violence, horror, dark gloomy tone, low quality, oversaturated")

MOOD_AMBIENCE = {
    "water-calm": "gentle flowing water, occasional bird call, no music",
    "rain-soft": "soft steady rain, distant thunder very quiet, no music",
    "birds-dawn": "small birds chirping, light breeze in leaves",
    "flowers-still": "light wind, soft water lapping, quiet",
    "forest-mist": "forest ambience, far birds, rustling leaves",
    "sunrise-warm": "morning birds, calm breeze, distant village waking",
    "night-quiet": "crickets at night, very soft breeze",
    "path-walking": "slow footsteps on dry earth, wind",
    "home-warm": "quiet room tone, kettle far away",
    "study-calm": "page turning, quiet room",
    "still-lake": "water lapping softly, occasional bird",
    "effort-dawn": "steady breathing, wind, footsteps",
    "kind-warm": "light rain, hushed street ambience",
    "calm-warm": DEFAULT_AMBIENCE,
}


def ambience_for(mood_tag, visual_prompt=""):
    """SFX Director's text prompt: mood first, then anything the script demands."""
    mood = str(mood_tag or "").strip().lower()
    if mood in MOOD_AMBIENCE:
        return MOOD_AMBIENCE[mood]
    hay = f"{mood} {visual_prompt}".lower()
    for (keys, _v, m, amb) in KEYWORD_IMAGERY:
        if m == mood:
            return amb
        for k in keys:
            if k in hay:
                return MOOD_AMBIENCE.get(m, amb)
    return DEFAULT_AMBIENCE


def imagery_for(text):
    """Deterministic (visual_prompt, mood_tag) for a scene — no LLM needed."""
    hay = (text or "").lower()
    for keys, visual, mood, _amb in KEYWORD_IMAGERY:
        for k in keys:
            if k and k in hay:
                return visual, mood
    return DEFAULT_VISUAL, DEFAULT_MOOD


def project_prompt(user_notes=""):
    notes = (user_notes or "").strip()
    if not notes:
        return STYLE_GUIDELINE
    return (STYLE_GUIDELINE + "\nEXTRA DIRECTOR NOTES for this project (may refine, "
            "but never contradict, the guideline above):\n" + notes + "\n")
