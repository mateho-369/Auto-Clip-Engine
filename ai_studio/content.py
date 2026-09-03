"""Content-type model: what kind of short the user is making.

Every project has a ``content_type``.  The value flows into the LLM prompts,
the deterministic (Ollama-offline) fallbacks and every scene's visual metadata,
so the same pipeline can make a calm explainer today and a split-frame compare
tomorrow without any new stage being added.

The types below are proven short-form educational/creator formats.  They are
deliberately independent of the house voice: an explainer must still be warm
and calm, but a ``what_if`` does not have to look like a nature documentary.
"""

# label / description are informational (the UI renders cards from these).
# prompt is the instruction injected into the Auto-Idea + Controller system.
# visual_tail is appended to every scene's visual_prompt for this type.
CONTENT_TYPES = {
    "explainer": {
        "label": "Explainer",
        "description": "One concept, explained simply and warmly.",
        "prompt": (
            "CONTENT TYPE: EXPLAINER. Pick one concept and explain it simply, "
            "warmly and without jargon. One idea per sentence; build from a soft "
            "opening to a small aha or a gentle takeaway."
        ),
        "visual_tail": "",
        "default_duration": 30.0,
    },
    "what_if": {
        "label": "What if…?",
        "description": "A hypothetical scenario explored in the calm house voice.",
        "prompt": (
            "CONTENT TYPE: WHAT IF. Open with a concrete hypothetical hook "
            "(e.g. \"What if we tried the opposite for one week?\"). Keep the "
            "house voice calm and curious — no doom, no shock. Explore one "
            "branch of the scenario, then land on a light reflection."
        ),
        "visual_tail": ("imaginative speculative visualization, surreal but soft, "
                        "soft dreamlike light, gentle slow camera drift") ,
        "default_duration": 30.0,
    },
    "compare": {
        "label": "Side-by-side",
        "description": "Two things set side by side in clearly parallel halves.",
        "prompt": (
            "CONTENT TYPE: COMPARE. Structure the script in two clearly parallel "
            "halves: A's trait, then B's equivalent trait, repeated. Every A point "
            "must have a matching B point. End with what each is best for, or the "
            "one difference that matters most."
        ),
        "visual_tail": ("split framing or alternating color treatment to contrast "
                        "side A vs side B, clean graphic contrast, minimal text"),
        "default_duration": 30.0,
    },
    "choose": {
        "label": "Decision helper",
        "description": "Helps the viewer decide between options.",
        "prompt": (
            "CONTENT TYPE: CHOOSE. Present two or three options with their real "
            "trade-offs, not as neutral description. Finish with a clear "
            "recommendation, or a crisp \"it depends on X\" framing. Speak like "
            "a helpful friend, never like a salesman."
        ),
        "visual_tail": ("clean options table or split visual, calm product-free "
                        "illustration, clear A/B/C emphasis"),
        "default_duration": 30.0,
    },
    "word_nuance": {
        "label": "Word nuance",
        "description": "Same word, two meanings — or two similar words that differ.",
        "prompt": (
            "CONTENT TYPE: WORD NUANCE. Contrast the two meanings or two words "
            "explicitly. Give one concrete example sentence for each so the "
            "distinction is heard, not just stated. End with when to use which."
        ),
        "visual_tail": ("paired word cards or mirrored typography, one concept per "
                        "side, soft contrastful light"),
        "default_duration": 30.0,
    },
    "myth_vs_fact": {
        "label": "Myth vs fact",
        "description": "A common misconception, then the correction.",
        "prompt": (
            "CONTENT TYPE: MYTH vs FACT. State the myth plainly first, then the "
            "fact. Briefly explain why people believe the myth (one line is enough), "
            "then give the fact with a concrete, simple reason. Stay calm and "
            "non-preachy."
        ),
        "visual_tail": ("myth side in muted washed tones, fact side in warm clear "
                        "light, simple correction graphic"),
        "default_duration": 30.0,
    },
    "quick_tip": {
        "label": "Quick tip",
        "description": "One fast, practical, actionable piece of advice.",
        "prompt": (
            "CONTENT TYPE: QUICK TIP. Give ONE fast, practical, actionable piece "
            "of advice. Use imperative, action-oriented language. Keep it shorter "
            "than the other types — one tip, why it works, and the simplest way to "
            "do it today."
        ),
        "visual_tail": ("practical close-up shot, simple diagram or hands "
                        "demonstrating, bright clear light"),
        "default_duration": 20.0,
    },
}

DEFAULT = "explainer"


def normalize_content_type(value, default=DEFAULT):
    """Return a known content type, or ``default`` when blank/unknown."""
    ct = str(value or default).strip().lower()
    return ct if ct in CONTENT_TYPES else default


def list_content_types():
    return [{"key": k, **{kk: vv for kk, vv in v.items() if kk != "prompt"}}
            for k, v in CONTENT_TYPES.items()]


def default_duration(content_type):
    return float(CONTENT_TYPES.get(normalize_content_type(content_type), {}).get(
        "default_duration", 30.0))


def content_type_prompt(content_type):
    """Instruction block injected into Auto-Idea and Controller system prompts."""
    ct = normalize_content_type(content_type)
    info = CONTENT_TYPES[ct]
    base = info.get("prompt", "")
    if ct == "quick_tip":
        base += "\nTarget runtime: shorter than the default (around 15-20s unless directed otherwise)."
    return f"\n{base}\n"


def visual_tail(content_type):
    """Tail appended to each scene's ``visual_prompt`` for this type."""
    ct = normalize_content_type(content_type)
    return CONTENT_TYPES[ct].get("visual_tail", "").strip()


def scene_tag(content_type, index, total):
    """The side/tag for a scene, assigned deterministically by content structure.

    ``compare`` and ``word_nuance`` use the first/second halves; ``myth_vs_fact``
    uses myth then fact; ``choose`` uses options then a takeaway.  Other types get
    a stable tag that downstream stages can consume the same way.
    """
    ct = normalize_content_type(content_type)
    idx = max(0, int(index))
    total = max(1, int(total))
    if ct == "compare":
        return "A" if idx < total / 2 else "B"
    if ct == "word_nuance":
        return "meaning-1" if idx < total / 2 else "meaning-2"
    if ct == "myth_vs_fact":
        return "myth" if idx < total / 2 else "fact"
    if ct == "choose":
        return "takeaway" if idx == total - 1 else "option"
    if ct == "what_if":
        return "scenario"
    if ct == "quick_tip":
        return "tip"
    return "explain"
