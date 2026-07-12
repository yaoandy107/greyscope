"""Text preprocessing, ported from pangramlabs/EditLens scripts/preprocess.py.

The EditLens pipeline (emoji / think-tag / header / lowercase / whitespace) is kept
intact so EN numbers stay comparable to the OpenPangram baselines. This repo prepends
Unicode hardening (`normalize_unicode`: homoglyph fold + invisible strip + NFKC) as the
default — a near-no-op on clean text, an attack defense on adversarial input. Pass
`clean_text(text, normalize=False)` to reproduce the exact EditLens preprocessing
(used for the EditLens-baseline comparison arm).

Source: https://github.com/pangramlabs/EditLens/blob/main/scripts/preprocess.py
"""

from __future__ import annotations

import re
import unicodedata

import emoji


BOILERPLATE_STARTS = [
    "Sure",
    "Here",
    "Abstract",
    "Title",
    "I'm happy to help",
    "Certainly",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_emoji(text: str) -> str:
    return emoji.demojize(text)


def remove_think_tag(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    return text


def remove_ai_header(text: str) -> str:
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) == 0:
        return text
    first_paragraph = paragraphs[0]
    first_paragraph = re.sub(r"^[^a-zA-Z0-9]*", "", first_paragraph)
    first_paragraph = emoji.replace_emoji(first_paragraph, "")
    if any(first_paragraph.startswith(phrase) for phrase in BOILERPLATE_STARTS):
        if len(paragraphs) > 1:
            text = "\n".join(paragraphs[1:])
    return text


# Invisible characters attacks insert to break tokenization without changing the
# rendered text. Stripped outright (mapped to None in the translation table).
_INVISIBLE = [
    "­",  # soft hyphen
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "‎",  # left-to-right mark
    "‏",  # right-to-left mark
    "⁠",  # word joiner
    "᠎",  # mongolian vowel separator
    "﻿",  # zero-width no-break space / BOM
]

# Cross-script homoglyphs (Cyrillic / Greek lookalikes of ASCII Latin) that NFKC does
# NOT fold — the basis of RAID's homoglyph attack. Mapped back to ASCII so the model
# sees the word it was trained on. Only these specific codepoints map, so CJK (Han /
# kana / hangul) is untouched; NFKC's own CJK effect (half/full-width kana + digits →
# canonical) is the intended ja/zh-tw normalization.
_CONFUSABLES = {
    # Cyrillic → Latin (uppercase)
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
    # Cyrillic → Latin (lowercase)
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j",
    # Greek → Latin (uppercase)
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Greek → Latin (lowercase, near-identical glyphs only)
    "ο": "o", "ρ": "p",
}

_TRANSLATE = {ord(k): v for k, v in _CONFUSABLES.items()}
_TRANSLATE.update({ord(c): None for c in _INVISIBLE})


def normalize_unicode(text: str) -> str:
    """Unicode-harden against tokenization attacks: fold cross-script homoglyphs to
    ASCII, strip invisible characters, then apply NFKC.

    NFKC folds full-width / ligatures / compatibility forms but NOT Cyrillic/Greek
    lookalikes, so the confusables map runs first. Near-idempotent on clean text.
    """
    return unicodedata.normalize("NFKC", text.translate(_TRANSLATE))


def clean_text(text: str, *, normalize: bool = True) -> str:
    if normalize:
        text = normalize_unicode(text)
    text = normalize_emoji(text)
    text = remove_think_tag(text)
    text = remove_ai_header(text)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def score_to_bucket(score: float, n_buckets: int, lo_threshold: float, hi_threshold: float) -> int:
    """Map a continuous `cosine_score` in [0, 1] to a bucket index.

    Bucket 0:       score <= lo_threshold   (human)
    Bucket n-1:     score >= hi_threshold   (heavily AI-edited / AI-generated)
    Buckets 1..n-2: evenly spaced between the thresholds

    Matches EditLens's bucketization: lo=0.03, hi=0.15, n_buckets=4.
    """
    if score <= lo_threshold:
        return 0
    if score >= hi_threshold:
        return n_buckets - 1
    normalized = (score - lo_threshold) / (hi_threshold - lo_threshold)
    return 1 + int(normalized * (n_buckets - 2))
