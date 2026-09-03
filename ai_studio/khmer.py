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
KHMER_BASE_RANGE = (0x1780, 0x17B3)   # verified against Unicode Khmer chart:
# consonants (U+1780-U+17A2) + independent vowels (U+17A3-U+17B3) are all
# Unicode category Lo.  U+17B4+ (vowel signs / vocalic signs / COENG) are marks.
KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
ZWSP = "\u200b"
ZWJ = "\u200d"
LRM = "\u200e"
COENG = "\u17d2"

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
    t = normalize(text)
    if not t:
        return 0
    if is_khmer(t):
        return len([p for p in t.split(" ") if p])
    return len([p for p in re.split(r"[\s\-–—/]+", t) if p])


def syllable_estimate(text):
    """Spoken syllable count. Khmer: ~0.6 syllable per letter-cluster unit."""
    t = normalize(text)
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


# ------------------------------------------------- Khmer character clusters
def _khmer_base(ch):
    """A Khmer consonant or independent vowel (Unicode category Lo in block)."""
    return bool(ch) and KHMER_RANGE[0] <= ord(ch) <= KHMER_RANGE[1] and unicodedata.category(ch) == "Lo"


def _khmer_mark(ch):
    """A Khmer combining/attached glyph (vowel sign, vocalic sign, COENG, ...)."""
    return bool(ch) and KHMER_RANGE[0] <= ord(ch) <= KHMER_RANGE[1] and unicodedata.category(ch).startswith("M")


def split_clusters(text):
    """Split a string into Khmer character clusters (KCC).

    Khmer is an abugida: ``COENG (U+17D2) + consonant`` is a subscript that is
    never allowed to be separated from the base it subscripts, and dependent
    vowels/signs are combining marks that must stay glued to the base.  Walking
    from a Khmer base this function consumes every following COENG pair and
    every following combining mark, so the returned strings are safe truncation
    and line-break units.  Non-Khmer characters become their own single-char
    unit (a following Unicode combining mark is kept with its base for Latin
    too, so general text is not damaged either).
    """
    out, i, n = [], 0, len(text or "")
    while i < n:
        ch = text[i]
        if _khmer_base(ch):
            j = i + 1
            while j < n:
                nxt = text[j]
                if nxt == COENG:
                    j += 1
                    if j < n:
                        j += 1                       # absorb the subscript consonant
                    continue
                if _khmer_mark(nxt):
                    j += 1
                    continue
                break
            out.append(text[i:j])
            i = j
            continue
        # non-base character
        if unicodedata.category(ch).startswith("M") and out:
            out[-1] = out[-1] + ch                   # Latin/other combining mark
        else:
            out.append(ch)
        i += 1
    return out


def cluster_count(text):
    """Number of Khmer character clusters (raw chars for plain Latin text)."""
    return len(split_clusters(text or ""))


def cluster_len(text):
    """Length in characters, but never counting a cluster boundary mid-grapheme."""
    return len(text or "")


def cluster_cut(text, count):
    """Character index after ``count`` clusters (used to continue a split)."""
    if count <= 0:
        return 0
    pos = 0
    for i, cl in enumerate(split_clusters(text or "")):
        if i >= int(count):
            break
        pos += len(cl)
    return pos


def cluster_boundary(text, index):
    """The largest cut at or before ``index`` that does not split a cluster.

    Used by caption/line wrappers that must slice by raw index for performance.
    """
    index = max(0, min(index, len(text or "")))
    if index == 0:
        return 0
    pos = 0
    for cl in split_clusters(text or ""):
        pos += len(cl)
        if pos >= index:
            return pos if index == pos else pos - len(cl)
    return pos


def truncate_clusters(text, max_clusters, suffix="…"):
    """Truncate by *cluster* count, never inside a Khmer cluster.

    ``suffix`` is appended only when truncation actually happened.  For plain
    Latin/mixed text a cluster is one character, so this is a drop-in safe
    replacement for ``text[:n]``.
    """
    if not text:
        return ""
    clusters = split_clusters(text)
    if len(clusters) <= max(0, int(max_clusters)):
        return text
    if max_clusters <= 0:
        return str(suffix or "")
    return "".join(clusters[:int(max_clusters)]).strip() + (suffix or "")


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
    """Split on Khmer spaces (then commas) to respect the cluster budget.

    ``max_chars`` is interpreted as *clusters* for Khmer text (the "n" unit used
    throughout this module), so a COENG subscript can never be sliced off its
    base when a long over-budget sentence is forced through ``split_sentences``.
    """
    use_clusters = is_khmer(sentence)

    def unit(text):
        return cluster_count(text) if use_clusters else len(text)

    chunks, cur = [], ""
    for piece in re.split(r"(?<= )", sentence):
        if unit(cur) + unit(piece) > max_chars and cur:
            chunks.append(cur.strip())
            cur = piece
        else:
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    # last resort: a single "word" longer than the budget
    final = []
    for c in chunks:
        while unit(c) > max_chars * 1.6:
            if use_clusters:
                cut = cluster_cut(c, max_chars)
                final.append(truncate_clusters(c, max_chars, suffix="").strip())
                c = c[cut:].strip()
            else:
                final.append(c[:max_chars].strip())
                c = c[max_chars:].strip()
        if c:
            final.append(c)
    return final


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
    changes. Used to prove no AI agent rewrote the Director's script.
    """
    def key(t):
        t = unicodedata.normalize("NFC", str(t or ""))
        t = _ZERO_WIDTH.sub("", t)
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
    """TTS-safe text: no emoji, no markdown, no [stage directions]."""
    t = normalize(text)
    t = re.sub(r"[\*\_`#>]+", " ", t)
    t = re.sub(r"\[[^\]]{0,60}\]|\([^)]{0,40}\)", " ", t)
    t = "".join(c for c in t if not (0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF))
    return re.sub(r"\s+", " ", t).strip()


def looks_like_markdown_or_notes(text):
    """Used by QA: the Director may have pasted headings/notes we must not speak."""
    t = text or ""
    return bool(re.search(r"^#{1,6}\s|^\s*[-*]\s|\*\*|__|^\s*>\s", t, re.M))


def title_from(text, maxlen=64):
    """Short title for a project card: first sentence, trimmed by cluster.

    ``maxlen`` is the old character budget; for Khmer it is treated as a cluster
    budget so truncation can never separate a COENG subscript from its base.
    """
    sents = split_sentences(text, max_chars=None)
    first = sents[0] if sents else normalize(text)
    first = strip_emoji_and_marks(first).rstrip(_SENT_END_KH)
    return truncate_clusters(first, maxlen) if first else "Untitled"
