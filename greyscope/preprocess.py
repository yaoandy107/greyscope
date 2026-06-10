"""Text preprocessing, ported from pangramlabs/EditLens scripts/preprocess.py.

`clean_text` is kept functionally identical so our numbers stay comparable to the
OpenPangram baselines, which were trained on `clean_text`-normalized inputs.

Source: https://github.com/pangramlabs/EditLens/blob/main/scripts/preprocess.py
"""

from __future__ import annotations

import re

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


def clean_text(text: str) -> str:
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
