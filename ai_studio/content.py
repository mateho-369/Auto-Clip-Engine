"""Content types — the creative framing of a project.

Every project used to be implicitly "explain something calmly and positively".
Now the Director picks a content type (or lets Auto mode choose one) and the
whole pipeline — script prompt, scene breakdown, visual tags, deterministic
fallbacks — is shaped accordingly.

This is a *definition* table, not opinions scattered through the stages: the
Controller prompt, the deterministic breakdown, the QA reviewer, the video/SFX
stage and the frontend all read the same ``CONTENT_TYPES`` dict, so adding a
new type later is one entry + (optionally) a deterministic rule, not a hunt
through six modules.

Research basis: these are proven short-form educational/creator formats
(explainer, hypothetical "what if", comparison, decision helper, word-nuance,
myth-vs-fact correction, quick tip).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContentTypeSpec:
    key: str
    label: str
    one_liner: str                      # what it is (frontend card description)
    instruction: str                    # injected into Controller / Scriptwriter prompts
    scene_rule: str                     # deterministic/LLM structural rule for the breakdown
    default_visual_source: str = "generated_video"   # generated_video | illustration | character_demo
    default_render_mode: str = "broll"               # broll | talking_head
    mood_bias: str = ""                 # preferred mood slug ('' = follow the words)
    target_duration_factor: float = 1.0              # quick_tip → shorter by default
    tags: dict = field(default_factory=dict)         # extra meta fields for scenes/scripts
    emoji: str = "🎬"


CONTENT_TYPES = {
    "explainer": ContentTypeSpec(
        key="explainer",
        label="Explainer",
        one_liner="One concept, explained simply and warmly — today's calm house style.",
        instruction=(
            "CONTENT TYPE: EXPLAINER. Explain ONE concept simply and warmly. One idea per "
            "sentence, concrete images, no jargon. The existing calm/positive house voice "
            "applies unchanged."
        ),
        scene_rule="one idea per scene; each scene introduces exactly one sub-point",
        emoji="🧠",
    ),
    "what_if": ContentTypeSpec(
        key="what_if",
        label="What If",
        one_liner="A hypothetical scenario explored — imaginative, speculative.",
        instruction=(
            "CONTENT TYPE: WHAT IF. Open the script with a hypothetical hook ('What if…'). "
            "Explore ONE plausible scenario with concrete, sensory consequences. Stay "
            "imaginative and speculative, but keep the tone warm and grounded — no fear-mongering. "
            "Visuals may lean more imaginative/speculative than the calm-nature default "
            "(e.g. dreamlike light, unreal palette, gentle surrealism)."
        ),
        scene_rule="scene 1 states the hypothetical; later scenes each explore one consequence",
        mood_bias="night-quiet",
        tags={"fantasy": True},
        emoji="✨",
    ),
    "compare": ContentTypeSpec(
        key="compare",
        label="Compare",
        one_liner="Two things set side by side — A's trait, then B's equivalent.",
        instruction=(
            "CONTENT TYPE: COMPARE. Structure the script in TWO clearly parallel halves: "
            "the first half introduces side A's traits, the second repeats the SAME structure "
            "for side B (A's trait → B's equivalent trait, over and over). Tag every scene "
            "with which side it belongs to (`A` or `B`) and end with one balanced summary "
            "scene. The visual treatment should contrast the two sides (split framing, "
            "alternating colour treatment) so the comparison reads visually, not just verbally."
        ),
        scene_rule=("split scenes into side A then side B (half the scenes each), then one "
                    "balanced summary scene; tag each scene meta.side = 'A' | 'B' | 'summary'"),
        default_visual_source="illustration",
        tags={"paired": True},
        emoji="⚖️",
    ),
    "choose": ContentTypeSpec(
        key="choose",
        label="Choose (decision helper)",
        one_liner="Helps the viewer decide between options — trade-offs plus a clear takeaway.",
        instruction=(
            "CONTENT TYPE: CHOOSE / DECISION HELPER. Present the options with their real "
            "trade-offs (what each costs and gives), then END with a clear takeaway: either "
            "a recommendation, or an honest 'it depends on X' framing. Not neutral description — "
            "the viewer should leave able to decide."
        ),
        scene_rule=("one scene per option (tag meta.side = option id), then a final takeaway "
                    "scene tagged meta.side = 'takeaway'"),
        default_visual_source="illustration",
        tags={"decide": True},
        emoji="🧭",
    ),
    "word_nuance": ContentTypeSpec(
        key="word_nuance",
        label="Word nuance",
        one_liner="Same word, two meanings — or two words for a similar feeling, contrasted.",
        instruction=(
            "CONTENT TYPE: WORD NUANCE. Explicitly contrast the two meanings/words with a "
            "CONCRETE example sentence for EACH, so the distinction is heard, not just stated "
            "abstractly. Label the two halves (meaning-1 / meaning-2) and end with a sentence "
            "that uses both correctly."
        ),
        scene_rule=("scene for meaning 1 (tag meta.side='meaning-1'), scene for meaning 2 "
                    "(meta.side='meaning-2'), optional contrast scene (meta.side='contrast')"),
        default_visual_source="illustration",
        tags={"nuance": True},
        emoji="🔤",
    ),
    "myth_vs_fact": ContentTypeSpec(
        key="myth_vs_fact",
        label="Myth vs fact",
        one_liner="A common misconception, corrected — myth first, fact second.",
        instruction=(
            "CONTENT TYPE: MYTH VS FACT. State the myth plainly FIRST (label it the myth), "
            "then the fact, and briefly address WHY people believe the myth (it sounds "
            "reasonable / it used to be taught / it is repeated). Never mock the believer — "
            "the correction is warm and clarifying."
        ),
        scene_rule=("scene 1 tagged meta.side='myth', scene 2 meta.side='fact' with the "
                    "'why people believe it' inside the fact scene, optional 'why-it-matters' closing"),
        tags={"correct": True},
        emoji="✅",
    ),
    "quick_tip": ContentTypeSpec(
        key="quick_tip",
        label="Quick tip",
        one_liner="One fast, practical, actionable piece of advice — shorter by default.",
        instruction=(
            "CONTENT TYPE: QUICK TIP. ONE fast practical actionable tip. Imperative, "
            "action-oriented (tell the viewer what to DO), not descriptive. Keep it shorter "
            "than the usual run — target duration defaults lower. One tip, one scene, done."
        ),
        scene_rule="1-2 scenes; language imperative; no scene may exceed ~8s",
        target_duration_factor=0.6,
        tags={"action": True},
        emoji="⚡",
    ),
}

DEFAULT_CONTENT_TYPE = "explainer"
VALID_CONTENT_TYPES = tuple(CONTENT_TYPES)
EXPRESSION_SUGGESTIONS = [
    "neutral", "happy", "sad", "determined", "surprised", "calm", "curious",
    "proud", "thoughtful", "excited", "worried", "grateful",
]

# mood → nearest character expression label (the matching rule, documented for
# the API response and README). Simple string/synonym match, deliberately
# over-engineered-avoided: unknown moods fall back to "neutral".
MOOD_TO_EXPRESSION = {
    "calm-warm": "calm", "water-calm": "calm", "still-lake": "calm",
    "rain-soft": "calm", "flowers-still": "calm",
    "sunrise-warm": "happy", "birds-dawn": "happy", "kind-warm": "happy",
    "night-quiet": "neutral", "forest-mist": "neutral", "home-warm": "calm",
    "study-calm": "thoughtful", "path-walking": "determined",
    "effort-dawn": "determined", "sad": "sad", "sorrow": "sad",
    "surprised": "surprised", "curious": "curious", "thoughtful": "thoughtful",
    "happy": "happy", "determined": "determined", "neutral": "neutral",
}


def spec(key):
    return CONTENT_TYPES.get(key or DEFAULT_CONTENT_TYPE, CONTENT_TYPES[DEFAULT_CONTENT_TYPE])


def instruction_block(key):
    """The content-type instruction injected into agent prompts."""
    return spec(key).instruction


def valid(key):
    return (key or DEFAULT_CONTENT_TYPE) in CONTENT_TYPES


def normalize(key):
    return key if valid(key) else DEFAULT_CONTENT_TYPE


def expression_for_mood(mood_tag, labels=None):
    """Nearest available expression label for a scene's mood.

    Rule: synonym table first (sorrow → sad), then exact label match, then
    'calm' (the house mood), then 'neutral'. `labels` (the character's actual
    uploaded labels) constrains the answer when provided.
    """
    mood = str(mood_tag or "").strip().lower().replace("_", "-")
    labels = [str(l).strip().lower() for l in (labels or []) if str(l).strip()]
    # the mood itself is a candidate only when it already *is* a known
    # expression label/alias — unknown moods must not leak through as labels
    known_aliases = set(MOOD_TO_EXPRESSION) | set(EXPRESSION_SUGGESTIONS) | {"calm"}
    cands = ([MOOD_TO_EXPRESSION.get(mood, "")] if mood in MOOD_TO_EXPRESSION else []) \
        + ([mood] if mood in known_aliases else []) \
        + [MOOD_TO_EXPRESSION.get("calm-warm"), "neutral", "calm"]
    if labels:
        for c in cands:
            if c and c in labels:
                return c
        return labels[0]
    for c in cands:
        if c:
            return c
    return "neutral"


def content_type_payload():
    """API-visible catalog (frontend renders cards from this — not hardcoded)."""
    return [{
        "key": s.key, "label": s.label, "one_liner": s.one_liner, "emoji": s.emoji,
        "default_visual_source": s.default_visual_source,
        "default_render_mode": s.default_render_mode,
        "target_duration_factor": s.target_duration_factor,
    } for s in CONTENT_TYPES.values()]
