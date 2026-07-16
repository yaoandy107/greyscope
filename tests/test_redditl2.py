"""Tests for the reddit-l2 (non-native-English) scraper (greyscope/pipeline/redditl2.py).

Pure functions only — no network calls. Guards the flair→country mapping (incl. the
English-native exclusion list), selftext cleaning (URL/markdown stripping), the length +
deleted/removed + English-heuristic filters, and submission parsing end-to-end.
"""

from greyscope.pipeline import redditl2 as rl2


def test_flair_maps_known_non_native_countries():
    assert rl2.country_from_flair("Germany") == "Germany"
    assert rl2.country_from_flair(":flag-de: Germany") == "Germany"
    assert rl2.country_from_flair(":flag-ru: Russia") == "Russia"
    assert rl2.country_from_flair("Czech Republic") == "Czechia"
    assert rl2.country_from_flair("Czechia") == "Czechia"


def test_flair_excludes_anglophone_countries():
    for flair in (":flag-us: United States of America", "United States", "United Kingdom",
                 "Ireland", "Canada", "Australia", "New Zealand", "Malta", "England"):
        assert rl2.country_from_flair(flair) is None


def test_flair_rejects_ambiguous_or_absent():
    assert rl2.country_from_flair(None) is None
    assert rl2.country_from_flair("") is None
    assert rl2.country_from_flair("   ") is None
    assert rl2.country_from_flair("Undecided") is None
    assert rl2.country_from_flair(":flag-eu: European Union") is None


def test_clean_selftext_strips_urls_and_markdown():
    raw = ("Check this out: https://example.com/path?q=1 it's **great**!\n\n"
          "# A header\n> a quoted line\n- bullet one\n- bullet two\n`inline code`")
    cleaned = rl2.clean_selftext(raw)
    assert "http" not in cleaned
    assert "**" not in cleaned and "#" not in cleaned and "`" not in cleaned
    assert "great" in cleaned and "bullet one" in cleaned


def test_clean_selftext_converts_markdown_links_to_text():
    cleaned = rl2.clean_selftext("See [this article](https://example.com/x) for details.")
    assert "this article" in cleaned and "https://example.com" not in cleaned


def test_is_deleted_or_removed():
    assert rl2.is_deleted_or_removed("[deleted]")
    assert rl2.is_deleted_or_removed("[removed]")
    assert rl2.is_deleted_or_removed("")
    assert rl2.is_deleted_or_removed("   ")
    assert not rl2.is_deleted_or_removed("A real post body.")


def test_looks_english_accepts_ascii_rejects_other_scripts():
    assert rl2.looks_english("This is a perfectly normal English sentence about Europe.")
    assert not rl2.looks_english("Это совершенно нормальное предложение на русском языке.")
    assert not rl2.looks_english("これは日本語の文章です。これはテストです。")
    assert not rl2.looks_english("")


def test_looks_english_tolerates_occasional_accented_names():
    # a stray non-ASCII proper noun (e.g. a place name) shouldn't tank an otherwise-English post
    text = "I visited München last year and it was a wonderful city with great food and history."
    assert rl2.looks_english(text)


def test_passes_filters_length_floor():
    short = "Too short."
    long_enough = "word " * 100  # 500 chars, clears the 400-char floor
    assert not rl2.passes_filters(rl2.clean_selftext(short))
    assert rl2.passes_filters(rl2.clean_selftext(long_enough))


def _raw_submission(**overrides):
    base = {
        "id": "abc123",
        "subreddit": "AskEurope",
        "author_flair_text": "Germany",
        "selftext": "word " * 100,
        "title": "A question about Europe",
        "created_utc": 1600000000,
    }
    return {**base, **overrides}


def test_parse_submission_accepts_valid_l2_post():
    parsed = rl2.parse_submission(_raw_submission())
    assert parsed is not None
    assert parsed["country"] == "Germany"
    assert parsed["subreddit"] == "AskEurope"
    assert parsed["id"] == "abc123"
    assert len(parsed["body"]) >= rl2.MIN_CHARS


def test_parse_submission_rejects_deleted_removed_short_and_excluded_flair():
    assert rl2.parse_submission(_raw_submission(selftext="[deleted]")) is None
    assert rl2.parse_submission(_raw_submission(selftext="[removed]")) is None
    assert rl2.parse_submission(_raw_submission(selftext="too short")) is None
    assert rl2.parse_submission(_raw_submission(author_flair_text="United Kingdom")) is None
    assert rl2.parse_submission(_raw_submission(author_flair_text=None)) is None


def test_parse_submission_rejects_non_english_body():
    russian = "это " * 150  # long enough, but not English
    assert rl2.parse_submission(_raw_submission(selftext=russian)) is None
