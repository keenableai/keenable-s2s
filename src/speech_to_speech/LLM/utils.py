import base64
import io
import re
from typing import Optional

import requests  # type: ignore[import-untyped]
from PIL import Image

SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

SPEECHABLE_PATTERN = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–\n\r\t]",
    flags=re.UNICODE,
)


def remove_unspeechable(text: str) -> str:
    """Keep only speechable characters: letters, digits, punctuation, whitespace.
    support unicode characters (english, arabic, chinese, japanese, korean, etc.)
    """
    text = text.translate(SMART_PUNCT_TRANSLATION)
    return SPEECHABLE_PATTERN.sub("", text)


# Sentence segmentation for streaming TTS. Replaces nltk.sent_tokenize (the nltk
# dependency was removed to close CVE-2026-54293, a path traversal in
# nltk.data.load). Dependency-free and tuned to match nltk's English punkt on the
# boundaries this pipeline relies on: split on . ! ? … when followed by
# whitespace, without breaking on decimals ("3.14"), initials ("J. Smith"), or
# common abbreviations ("Mr.", "e.g."). Trailing text with no terminator is
# returned as the final element, so streaming callers can keep it buffered.
_SENTENCE_TERMINATORS = ".!?…"

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
        "inc", "ltd", "co", "corp", "no", "fig", "al", "gen", "rev", "hon",
        "e.g", "i.e", "a.m", "p.m", "u.s", "u.k",
    }
)

_PRECEDING_TOKEN = re.compile(r"[A-Za-z.]+$")


def _is_false_boundary(text: str, term_start: int, next_char: str) -> bool:
    """True when a . ! ? at ``term_start`` is not a real sentence boundary."""
    # A following lowercase letter is a strong signal we're mid-sentence
    # (e.g. "e.g. foo", "see fig. 2 below").
    if next_char and next_char.islower():
        return True
    match = _PRECEDING_TOKEN.search(text[:term_start])
    if not match:
        return False
    token = match.group(0)
    normalized = token.lower().rstrip(".")
    if not normalized:
        return False
    # Single-letter initials ("J. Smith", "U. S.").
    if len(normalized.replace(".", "")) == 1:
        return True
    return normalized in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences for streaming TTS.

    Drop-in replacement for ``nltk.sent_tokenize`` covering the cases this
    pipeline exercises. Returns stripped, non-empty sentences; any trailing
    fragment without a terminator is the last element.
    """
    if not text or not text.strip():
        return []
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in _SENTENCE_TERMINATORS:
            i += 1
            continue
        # Consume a run of terminators ("?!", "...").
        run_end = i
        while run_end < n and text[run_end] in _SENTENCE_TERMINATORS:
            run_end += 1
        # A boundary requires end-of-text or trailing whitespace (this alone
        # excludes decimals and mid-token dots like "3.14" / "a.m").
        if run_end < n and not text[run_end].isspace():
            i = run_end
            continue
        next_pos = run_end
        while next_pos < n and text[next_pos].isspace():
            next_pos += 1
        next_char = text[next_pos] if next_pos < n else ""
        if _is_false_boundary(text, i, next_char):
            i = run_end
            continue
        segment = text[start:run_end].strip()
        if segment:
            sentences.append(segment)
        start = run_end
        i = run_end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
}


def resolve_auto_language(language_code: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Strip the ``-auto`` suffix and resolve the human-readable language name.

    Returns ``(clean_code, language_name)``.  ``language_name`` is non-None
    when the code (with or without ``-auto``) maps to a known language.
    """
    if not language_code:
        return language_code, None
    if language_code.endswith("-auto"):
        language_code = language_code[:-5]
    if language_code not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE:
        return language_code, None
    return language_code, WHISPER_LANGUAGE_TO_LLM_LANGUAGE.get(language_code)


def image_url_to_pil(image_url: str) -> Image.Image:
    """Convert an image URL or base64 data URI to a PIL Image.

    Accepts:
    - 'data:image/...;base64,<b64>' data URIs
    - 'https://...`` or ``http://...' URLs (fetched with a 10s timeout)
    """
    if image_url.startswith("data:"):
        _, b64_data = image_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64_data)))
    resp = requests.get(image_url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))
