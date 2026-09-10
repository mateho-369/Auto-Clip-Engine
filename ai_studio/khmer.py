"""Khmer-aware text mechanics: normalisation, sentence splitting, timing.

Why this module exists at all: Khmer (ភាសាខ្មែរ) is written *scriptio continua* —
words are not separated by spaces, so every Western heuristic ("count words",
"split at 12 words") is wrong for it. On top of that Khmer text is full of
zero-width joiners (U+200B word separators, U+200C/ZWJ coeng stacks) and the
sentence finisher is often a space or ។ (period) rather than '.'.

Everything the pipeline does with text — mechanical scene segmentation, spoken
duration estimates, chunking for the TTS model, QA silence checks — goes
through here so the numbers stay consistent between stages.
"""
import re
import unicodedata

# Khmer script block + the punctuation/zero-width characters we care about.
KHMER_RANGE = (0x1780, 0x17FF)
KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
ZWSP = "\u200b"
ZWJ = "\u200d"
LRM = "\u200e"

# ------------------------------------------------------ character-cluster units
# Verified against the Unicode Khmer chart (U+1780–U+17FF) via unicodedata's
# categories (Python 3.11, Unicode 15): the *base* characters of an abugida
# cluster are the letters (consonants U+1780–U+17A2 + independent vowels
# U+17A3–U+17B3), all category Lo. Everything else inside the block that the
# shaping engine glues onto a base is a combining mark (category Mn/Mc,
# U+17B4–U+17D3) — dependent vowels, signs, and the coeng U+17D2.
#  U+17DC (KHMER SIGN AVAKRAHASANYA) is category Lo but is a *sign*, not a
#  letter — it must never start a new cluster.
#  U+17D7 (KHMER SIGN LEK TOO) is category Lm and behaves like a letter.
COENG = "\u17d2"
COENG_CP = 0x17d2
# bases: Lo in U+1780..U+17B3 (independent vowels stop at U+17B3); U+17DC is
# deliberately outside this range.
KHMER_BASE_RANGE = (0x1780, 0x17B3)
# combining marks / dependent vowels / signs glued to the base (incl. coeng)
KHMER_MARK_RANGE = (0x17B4, 0x17D3)
KHMER_EXTRA_MARKS = (0x17DC,)          # AVAKRAHASANYA — a sign, not a base
KHMER_LEK_TOO = 0x17D7                  # letter-like Lm sign → starts a cluster
_KHMER_BASES = None
_KHMER_MARKS = None


def _build_masks():
    """Build the cluster masks once from unicodedata (verified sub-ranges)."""
    global _KHMER_BASES, _KHMER_MARKS
    if _KHMER_BASES is not None:
        return
    bases, marks = set(), set()
    for cp in range(KHMER_BASE_RANGE[0], KHMER_BASE_RANGE[1] + 1):
        if unicodedata.category(chr(cp)) == "Lo":
            bases.add(cp)
    for cp in range(KHMER_MARK_RANGE[0], KHMER_MARK_RANGE[1] + 1):
        if unicodedata.category(chr(cp))[0] == "M":
            marks.add(cp)
    marks.update(KHMER_EXTRA_MARKS)
    _KHMER_BASES, _KHMER_MARKS = bases, marks


def is_base(cp):
    """True when `cp` can start a Khmer orthographic cluster."""
    _build_masks()
    return cp in _KHMER_BASES or cp == KHMER_LEK_TOO


def is_mark(cp):
    """True when `cp` is a Khmer combining mark / dependent vowel / sign
    (including coeng U+17D2 and the AVAKRAHASANYA sign)."""
    _build_masks()
    return cp in _KHMER_MARKS


def split_clusters(text):
    """Split `text` into Khmer Character Clusters (KCCs).

    A Khmer cluster = one base (consonant / independent vowel / lek-too) plus
    anything the shaping engine glues on: every ``COENG + following char`` pair
    (subscript consonants) and every combining mark / dependent vowel / sign.
    COENG is the dangerous one: ``ស + ្ + វ`` must stay one string, or a viewer
    sees a stray space after a bare ``្``.

    Non-Khmer characters (Latin, digits, spaces, punctuation) are returned as
    their own single-character "cluster" so every *index* in the returned list
    is a safe truncation/wrapping point.

    >>> split_clusters("ស្វែង")
    ['ស្', 'វែ', 'ង']  # ្ + ស... grouping shown for illustration — actual units keep all codepoints
    """
    if not text:
        return []
    _build_masks()
    out, cur = [], []
    i, n = 0, len(text)
    while i < n:
        cp = ord(text[i])
        if is_base(cp):
            cur = [text[i]]
            i += 1
            # absorb: (COENG + any single following codepoint) or (combining mark)
            while i < n:
                c = text[i]
                if c == COENG and i + 1 < n:
                    cur.append(c)
                    cur.append(text[i + 1])
                    i += 2
                    continue
                cp2 = ord(c)
                if is_mark(cp2):
                    cur.append(c)
                    i += 1
                    continue
                break
            out.append("".join(cur))
            continue
        if text[i] == COENG and i + 1 < n:
            # stray coeng at a cluster boundary (shouldn't happen in valid text,
            # but never lose it): glue it to the following char.
            out.append(text[i] + text[i + 1])
            i += 2
            continue
        out.append(text[i])
        i += 1
    return out


def truncate_clusters(text, max_clusters, suffix="…"):
    """Truncate by cluster count — never mid-cluster.

    ``max_clusters`` is a number of KCCs, not codepoints, and the suffix is
    appended only when something was cut. ``max_clusters <= 0`` returns ``""``.
    """
    if text is None:
        return ""
    max_clusters = max(0, int(max_clusters))
    clusters = split_clusters(text)
    if len(clusters) <= max_clusters:
        return text
    if max_clusters == 0:
        return ""
    return "".join(clusters[:max_clusters]) + suffix


def cluster_len(text):
    """Length of `text` in character-cluster units (for Khmer-aware budgets)."""
    return len(split_clusters(text or ""))


def clip_clusters(text, max_clusters):
    """Truncate by cluster count with NO suffix (silent cut)."""
    return truncate_clusters(text, max_clusters, suffix="")


def words(text):
    """Khmer-aware *word* tokens for caption wrapping and karaoke timing.

    Uses the ``khmercut`` dictionary segmenter when it is installed, so a word
    like យឺត stays whole (cluster-safe wrapping alone still breaks words at
    cluster boundaries, which reads like hyphenating "bab/y" without the
    hyphen). khmercut ships its dictionary in the wheel and works offline.
    Falls back to whitespace splitting, then to 2-cluster pseudo-words for
    scriptio-continua text that has no spaces at all.
    """
    t = (text or "").strip()
    if not t:
        return []
    try:
        from khmercut import tokenize  # optional dependency, offline dictionary
        toks = [w for w in tokenize(t) if w.strip()]
        if toks:
            # line-break safety: a trailing sign (។ ៕ ៖ ៗ ! ? . ,) is part of
            # the word it follows — it must never start a caption line. Merging
            # here fixes wrapping AND keeps karaoke highlighting attached.
            merged: list[str] = []
            for tok in toks:
                if merged and len(tok) == 1 and tok in "។៕៖ៗ？！?!.,，":
                    merged[-1] += tok
                else:
                    merged.append(tok)
            return merged
    except Exception:
        pass
    if " " in t:
        toks = [w for w in re.split(r"\s+", t) if w]
        if toks:
            return toks
    units = split_clusters(t)
    words, cur = [], []
    for u in units:
        cur.append(u)
        if len(cur) >= 2:
            words.append("".join(cur))
            cur = []
    if cur:
        words.append("".join(cur))
    return words


def wrap_words(text, max_clusters=16, width_fn=None, budget_px=None):
    """Wrap `text` into lines that break BETWEEN words, never inside one.

    Break points come from `words()` — dictionary word boundaries when
    khmercut is available. Source spaces are PRESERVED: the spaces the author
    wrote stay in the rendered line, and a line break consumes the space it
    breaks at instead of deleting it.

    Width is measured with ``width_fn`` (e.g. shaped pixel widths from
    captions.Shaper) when given, otherwise in character clusters — the visual
    unit Khmer actually occupies. ``budget_px`` overrides the cluster budget
    when measuring in pixels.
    """
    t = (text or "").strip()
    if not t:
        return [""]
    toks: list[tuple[str, bool]] = []
    for pi, phrase in enumerate(t.split(" ")):
        if not phrase:
            continue
        for j, w in enumerate(words(phrase)):
            toks.append((w, pi > 0 and j == 0))
    measure = width_fn or (lambda s: cluster_len(s))
    budget = float(budget_px) if budget_px else max(1, int(max_clusters))
    space_w = measure(" ") if width_fn else 1.0

    # token list with widths (a lone trailing sign stays glued to its word)
    toks2: list[tuple[str, bool, float]] = []
    for word, had_space in toks:
        lone_punct = len(word) == 1 and word in "។៕៖？！?!.,，—–-/" and toks2
        if lone_punct:
            pw, _ps, _w = toks2[-1]
            sep = " " if had_space else ""
            toks2[-1] = (pw + sep + word, _ps, _w + (space_w if had_space else 0.0) + measure(word))
            continue
        toks2.append((word, had_space, measure(word)))
    n = len(toks2)
    if n == 0:
        return [""]

    # prefix widths: width(i,j) = shaped width of tokens i..j on one line
    pref = [0.0]
    for _w, _s, wd in toks2:
        pref.append(pref[-1] + (space_w if _s else 0.0) + wd)
    INF = float("inf")

    def line_width(i: int, j: int) -> float:
        return pref[j] - pref[i]

    def line_cost(i: int, j: int, is_last: bool) -> float:
        """Raggedness cost of tokens i..j as one (possibly last) line."""
        leftover = budget - line_width(i, j)
        if leftover >= 0:
            # a single word alone on the final line reads as a mistake —
            # balance instead (unless it is the only possible break)
            orphan = is_last and j - i == 1 and n > 1
            return (leftover * leftover) * (4.0 if orphan else 1.0)
        if j - i == 1:
            # one word wider than the whole line: allowed, flagged by caller
            return (budget * budget) * 8.0
        return INF

    # DP minimum-raggedness: dp[j] = best cost of the first j tokens
    dp = [INF] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            if dp[i] == INF:
                continue
            c = dp[i] + line_cost(i, j, j == n)
            if c < dp[j]:
                dp[j] = c
                back[j] = i
    # fall back to one-word-per-line if even that is impossible
    if dp[n] == INF:  # pragma: no cover (single tokens are always allowed)
        return [w for w, _s, _wd in toks2]
    bounds, j = [], n
    while j > 0:
        bounds.append((back[j], j))
        j = back[j]
    bounds.reverse()
    out = []
    for i, j in bounds:
        parts = [(w, s and k > i) for k, (w, s, _wd) in enumerate(toks2) if i <= k < j]
        out.append("".join((" " if s else "") + w for w, s in parts))
    return out or [""]


def wrap_clusters(text, max_clusters=16):
    """Wrap `text` at cluster boundaries: returns a list of lines, each at most
    ``max_clusters`` long (a single cluster longer than the budget is its own
    line — never split). Used by caption wrapping, where a raw codepoint cut
    is exactly what produces the ``្ + space`` corruption."""
    if not text:
        return [""] if text is not None else []
    max_clusters = max(1, int(max_clusters))
    lines, cur, count = [], [], 0
    for cl in split_clusters(text):
        # never break *inside* a cluster; a huge cluster gets its own line
        if cur and count + 1 > max_clusters:
            lines.append("".join(cur))
            cur, count = [], 0
        cur.append(cl)
        count += 1
    if cur:
        lines.append("".join(cur))
    return lines or [""]

# -------------------------------------------------------------- silent markup
# [[silent: ...]] — the Director marks words to appear on screen but not be
# spoken. Bracket content is removed entirely from the TTS string; the inner
# words are kept (brackets removed) for captions/on-screen text.
_SILENT_RE = re.compile(r"\[\[\s*silent\s*:\s*(.*?)\s*\]\]", re.DOTALL | re.IGNORECASE)


def split_silent(text):
    """Split a script into ``(display_piece, spoken_piece)`` blocks.

    Each block is a contiguous run of normal text (``display == spoken``) or a
    single ``[[silent: …]]`` span (``spoken == ""``, ``display == inner text``).
    Works on any text (Khmer or Latin); other ``[[...]]``/``[...]`` notations
    are left alone by this function (``strip_emoji_and_marks`` handles those).
    """
    if not text:
        return []
    out = []
    pos = 0
    for m in _SILENT_RE.finditer(text):
        if m.start() > pos:
            chunk = text[pos:m.start()]
            out.append((chunk, chunk))
        out.append((m.group(1) or "", ""))
        pos = m.end()
    if pos < len(text):
        chunk = text[pos:]
        out.append((chunk, chunk))
    return out


def spoken_text(text):
    """Everything that should reach the TTS engine: silent blocks removed."""
    return "".join(sp for _d, sp in split_silent(text or ""))


def display_text(text):
    """Everything that should appear on screen: brackets removed, words kept."""
    return _SILENT_RE.sub(lambda m: m.group(1) or "", text or "")


def has_silent_markup(text):
    return bool(_SILENT_RE.search(text or ""))

_SENT_END_KH = "។៕៙៚!?။"            # ។ khmer period, plus latin marks
_SOFT_BREAK = re.compile(r"[,;:\u1784\u179A]?")  # noqa: RUF001 (harmless)

_SENT_SPLIT = re.compile(
    r"(?<=[។៕៙៚!?])\s*"            # after a hard mark (+ optional space)
    r"|(?<=[।\.\?!])\s+"            # latin marks need a following space
    r"|\n+",                        # any newline (scripts are pasted with line breaks)
)

_WHITESPACE_RUN = re.compile(r"[ \t\r\f\v\u00a0\u1680\u2000-\u200f\u2028\u2029\u205f\u3000]+")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200e\u200f\ufeff]")


def is_khmer(text):
    """True when most of the letters are Khmer script."""
    letters = [c for c in text or "" if not c.isspace()]
    if not letters:
        return False
    km = sum(1 for c in letters if KHMER_RANGE[0] <= ord(c) <= KHMER_RANGE[1])
    return km / len(letters) > 0.5


def normalize(text):
    """Unicode-normalise + strip zero-width noise, keep meaningful spacing.

    NFKC would happily *destroy* Khmer stacking (it decomposes coeng forms into
    sequences some fonts render badly), so we use NFC and only remove the
    invisible joiners that TTS front-ends choke on.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _ZERO_WIDTH.sub("", t)
    t = t.replace("\u00ad", "")          # soft hyphen
    t = _WHITESPACE_RUN.sub(" ", t)
    # a Khmer space is also a word/sentence separator: squeeze runs
    t = re.sub(r" +", " ", t)
    return t.strip()


def normalize_block(text):
    """Paragraph-preserving normalisation (used for the Director's script)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _ZERO_WIDTH.sub("", t)
    lines = [_WHITESPACE_RUN.sub(" ", ln).strip() for ln in t.replace("\r\n", "\n").split("\n")]
    out, blank = [], 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank <= 1 and out:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def char_len(text):
    """'Letters' excluding spaces — the Khmer size unit."""
    return len(re.sub(r"\s+", "", text or ""))


def token_count(text):
    """Word-ish count: latin words, or Khmer syllable clusters (per space).

    For Khmer we count *clusters* between spaces; each is roughly 1-3 spoken
    syllables, so `syllable_estimate` refines it for timing.
    """
    t = normalize(spoken_text(text))
    if not t:
        return 0
    if is_khmer(t):
        return len([p for p in t.split(" ") if p])
    return len([p for p in re.split(r"[\s\-–—/]+", t) if p])


def syllable_estimate(text):
    """Spoken syllable count. Khmer: ~0.6 syllable per letter-cluster unit.

    ``[[silent: …]]`` words are not spoken, so they are excluded — duration
    estimates must reflect what the TTS will actually say.
    """
    t = normalize(spoken_text(text))
    if not t:
        return 0.0
    if is_khmer(t):
        letters = char_len(t)
        clusters = token_count(t)
        return max(1.0, letters / 4.1 + clusters * 0.35)
    words = t.split()
    vowels = sum(len(re.findall(r"[aeiouyàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụ]+", w, re.I)) for w in words)
    return max(float(len(words)), float(vowels))


def estimate_speech_seconds(text, wpm=None, calm=1.0):
    """Estimated spoken duration.

    Calm/meditative Khmer narration runs ~3.2-3.8 syllables/sec; `calm` (>1 =
    slower) is exposed so the studio's "pace" setting actually changes scene
    budgets and the QA tolerance band.
    """
    syl = syllable_estimate(text)
    if syl <= 0:
        return 0.0
    rate = (wpm / 60.0) if wpm else 3.5
    rate = max(1.0, rate / max(0.5, float(calm or 1.0)))
    return round(syl / rate + 0.35, 2)      # + tail ring-down


# -------------------------------------------------------------- segmentation
def split_sentences(text, max_chars=None):
    """Split a script into ordered sentences, honouring newlines as hard breaks.

    ``max_chars`` over-long runs (a Director pasting one 400-char paragraph) are
    secondarily split at Khmer spaces so no scene is unrenderable.
    """
    text = normalize_block(text)
    if not text:
        return []
    raw = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts = [p.strip() for p in _SENT_SPLIT.split(para) if p and p.strip()]
        raw.extend(parts)
    if max_chars:
        out = []
        for s in raw:
            if len(s) <= max_chars:
                out.append(s)
                continue
            out.extend(_hard_split(s, max_chars))
        return [s for s in out if s]
    return [s for s in raw if s]


def _hard_split(sentence, max_chars):
    """Split on Khmer spaces (then commas) to respect the char budget.

    ``max_chars`` is interpreted as *characters* for Latin text and as
    *character clusters* for Khmer — a raw codepoint cut can land between a
    coeng (U+17D2) and its subscript consonant, which renders as a broken
    syllable. The budget comparisons therefore use ``cluster_len()`` on Khmer
    and ``len()`` elsewhere.
    """
    is_kh = is_khmer(sentence)
    measure = (lambda t: cluster_len(t)) if is_kh else len
    chunks, cur = [], ""
    for piece in re.split(r"(?<= )", sentence):
        if measure(cur) + measure(piece) > max_chars and cur:
            chunks.append(cur.strip())
            cur = piece
        else:
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    # last resort: a single "word" longer than the budget — still cluster-cut
    final = []
    for c in chunks:
        while measure(c) > max_chars * 1.6:
            final.append(truncate_clusters(c, max_chars).rstrip("…").strip())
            c = _skip_clusters(c, max_chars).strip()
        if c:
            final.append(c)
    return [f for f in final if f]


def measure_from_start(text, n):
    """Codepoint length of the first `n` clusters (Latin: just `n`)."""
    if is_khmer(text):
        return len("".join(split_clusters(text)[:max(0, int(n))]))
    return max(0, int(n))


def _skip_clusters(text, n):
    """Everything after the first `n` clusters (Khmer) or `n` codepoints."""
    if is_khmer(text):
        return "".join(split_clusters(text)[max(0, int(n)):])
    return text[max(0, int(n)):]


def join_sentences(sentences):
    """Re-join scene texts the way they must read when spoken.

    Always a single space: in Khmer the space *is* the boundary marker when ។ is
    absent, the TTS front-end wants it, and `equal_text` ignores whitespace so the
    Director's wording is still provably unchanged.
    """
    return " ".join(s.strip() for s in (sentences or []) if s and s.strip())


def equal_text(a, b):
    """Compare two script texts for *creative equality* (mode-A guarantee).

    Ignores whitespace/zero-width/normalisation differences only — never word
    changes. ``[[silent: …]]`` markup is treated as intentional formatting (the
    words inside are part of the script, they are just not spoken), so it is
    normalised away on BOTH sides before the comparison — the Director's
    wording is still provably unchanged.
    """
    def key(t):
        t = unicodedata.normalize("NFC", str(t or ""))
        t = _ZERO_WIDTH.sub("", t)
        t = display_text(t)                      # brackets = formatting, not words
        return re.sub(r"[\s]+", "", t)
    return key(a) == key(b)


def to_khmer_digits(text):
    out = str(text or "")
    for i, d in enumerate(KHMER_DIGITS):
        out = out.replace(str(i), d)
    return out


def from_khmer_digits(text):
    out = str(text or "")
    for i, d in enumerate(KHMER_DIGITS):
        out = out.replace(d, str(i))
    return out


def tts_chunks(text, max_chars=180):
    """Sentence groups sized for the VITS model (it degrades past ~200 chars)."""
    sents = split_sentences(text, max_chars=int(max_chars * 1.4)) or [normalize(text)]
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [normalize(text) or "…"]


# ------------------------------------------------------------- misc guards
def strip_emoji_and_marks(text):
    """TTS-safe text: no emoji, no markdown, no [stage directions].

    ``[[silent: …]]`` is display-ONLY markup, not a stage direction — the
    spans are protected from the generic bracket stripper so the Director's
    caption wording survives (spoken_text() removes them when speaking).
    """
    t = normalize(text)
    protected = []
    def _hold(m):
        protected.append(m.group(0))
        return "\x00%d\x00" % (len(protected) - 1)
    t = _SILENT_RE.sub(_hold, t)
    t = re.sub(r"[\*\_`#>]+", " ", t)
    t = re.sub(r"\[[^\]]{0,60}\]|\([^)]{0,40}\)", " ", t)
    t = "".join(c for c in t if not (0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF))
    t = re.sub(r"\s+", " ", t).strip()
    for i, seg in enumerate(protected):
        t = t.replace("\x00%d\x00" % i, seg)
    return t


def looks_like_markdown_or_notes(text):
    """Used by QA: the Director may have pasted headings/notes we must not speak."""
    t = text or ""
    return bool(re.search(r"^#{1,6}\s|^\s*[-*]\s|\*\*|__|^\s*>\s", t, re.M))


def title_from(text, maxlen=64):
    """Short title for a project card: first sentence, trimmed by CLUSTERS.

    ``maxlen`` counts Khmer character clusters (never codepoints), so a title
    like ``ស្វែងយល់`` can never be cut between the coeng and its subscript —
    the bug where ``ស្វែងយល់`` rendered as ``ស្ វែងយល់``.
    """
    sents = split_sentences(text, max_chars=None)
    first = sents[0] if sents else normalize(text)
    first = strip_emoji_and_marks(first).rstrip(_SENT_END_KH)
    if cluster_len(first) > maxlen:
        return truncate_clusters(first, maxlen)
    return first or "Untitled"
