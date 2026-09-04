"""Mood → pose/body-language phrases for character shots.

The video stage used to only append an atmosphere phrase to the prompt. When
a character is on screen (``character_demo`` / ``talking_head`` / character
I2V) the pose matters as much as the light, so each mood carries an actual
body-language description. Editable table (not inline constants) so more moods
can be added later in one place.

For each mood: ``pose`` = what the character should physically do;
``gaze``/``handling`` folded into the phrase for simplicity.
"""

MOOD_POSES = {
    "sad": "sitting slumped, head lowered, slow subtle movement",
    "sorrow": "sitting slumped, head lowered, slow subtle movement",
    "hopeful": "standing tall, gentle upward gaze, relaxed open posture",
    "happy": "standing tall, gentle upward gaze, relaxed open posture",
    "joy": "standing tall, gentle upward gaze, relaxed open posture",
    "sunrise-warm": "standing tall, gentle upward gaze, relaxed open posture",
    "kind-warm": "standing tall, gentle upward gaze, relaxed open posture",
    "birds-dawn": "standing tall, gentle upward gaze, relaxed open posture",
    "determined": "standing firm, steady gaze forward",
    "effort-dawn": "standing firm, steady gaze forward",
    "path-walking": "standing firm, steady gaze forward",
    "calm": "relaxed seated posture, soft slow breathing motion",
    "calm-warm": "relaxed seated posture, soft slow breathing motion",
    "water-calm": "relaxed seated posture, soft slow breathing motion",
    "still-lake": "relaxed seated posture, soft slow breathing motion",
    "rain-soft": "relaxed seated posture, soft slow breathing motion",
    "flowers-still": "relaxed seated posture, soft slow breathing motion",
    "home-warm": "relaxed seated posture, soft slow breathing motion",
    "study-calm": "leaning slightly forward, thoughtful expression, still hands",
    "thoughtful": "leaning slightly forward, thoughtful expression, still hands",
    "curious": "head tilted, one hand raised slightly, attentive posture",
    "surprised": "eyes wide, leaning back slightly, hands slightly raised",
    "night-quiet": "relaxed seated posture, soft slow breathing motion",
    "forest-mist": "standing still, arms relaxed, calm steady breath",
    "excited": "standing tall, hands gesturing openly, bright upward gaze",
    "proud": "standing tall, chin slightly raised, steady gaze forward",
    "worried": "standing with weight shifted, hands clasped, glancing down",
    "grateful": "hands at chest, gentle nod, soft closed-eye smile",
}

DEFAULT_POSE = "relaxed seated posture, soft slow breathing motion"


def pose_for(mood_tag):
    """Body-language phrase for a mood slug (never empty)."""
    mood = str(mood_tag or "").strip().lower().replace("_", "-")
    return MOOD_POSES.get(mood, DEFAULT_POSE)


def poses():
    """API-visible table (frontend can show what a mood will look like)."""
    return list(MOOD_POSES.items())
