"""
Legal heading detector — identifies structured headings in legal documents
using language-aware regex patterns.

Supported languages:
  - ``en``  — English (Article, Section, Chapter, Part, Clause, Schedule, …)
  - ``id``  — Bahasa Indonesia (BAB, Bagian, Paragraf, Pasal, Ayat, …)

The detector is intentionally conservative: it only promotes items whose
``text_level`` is ``-1`` (body text) and whose *entire trimmed text* matches
a known legal heading pattern.  This avoids false-positives on sentences
that merely *mention* a legal keyword.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

try:
    from langdetect import detect as _langdetect_detect
    from langdetect import DetectorFactory

    # Make langdetect deterministic
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:  # pragma: no cover
    _HAS_LANGDETECT = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------
# Each entry is ``(compiled_regex, assigned_text_level)``.
# Lower ``text_level`` values indicate higher hierarchy levels so that the
# outline extractor can nest them correctly.
#
# ``text_level`` assignments (per language):
#   0 = top-level division (BAB / Chapter / Title / Part)
#   1 = section / Bagian
#   2 = sub-section / Paragraf
#   3 = article / Pasal / Section / Clause
#   4 = clause / verse / Ayat / sub-clause
# ---------------------------------------------------------------------------

_NUM = r"(?:[0-9]+(?:\.[0-9]+)*|[IVXLCDM]+|[ivxlcdm]+)"
_ALPHA = r"[A-Za-z]"

# ── English patterns ──────────────────────────────────────────────────────

_EN_PATTERNS: List[Tuple[re.Pattern, int]] = [
    # Level 0 – top-level
    (re.compile(rf"^(?:TITLE|Title)\s+{_NUM}\.?(?:\s.*)?$"), 0),
    (re.compile(rf"^(?:PART|Part)\s+{_NUM}\.?(?:\s.*)?$"), 0),
    (re.compile(rf"^(?:CHAPTER|Chapter)\s+{_NUM}\.?(?:\s.*)?$"), 0),
    # Level 1 – section
    (re.compile(rf"^(?:DIVISION|Division)\s+{_NUM}\.?(?:\s.*)?$"), 1),
    (re.compile(rf"^(?:ARTICLE|Article)\s+{_NUM}\.?(?:\s.*)?$"), 1),
    (re.compile(rf"^(?:SCHEDULE|Schedule)\s+{_NUM}\.?(?:\s.*)?$"), 1),
    # Level 2 – sub-section
    (re.compile(rf"^(?:SECTION|Section)\s+{_NUM}\.?(?:\s.*)?$"), 2),
    (re.compile(rf"^(?:ANNEX|Annex)\s+{_ALPHA}\.?(?:\s.*)?$"), 2),
    # Level 3 – clause
    (re.compile(rf"^(?:CLAUSE|Clause)\s+{_NUM}\.?(?:\s.*)?$"), 3),
    (re.compile(rf"^§\s*{_NUM}\.?(?:\s.*)?$"), 3),
    # Level 4 – sub-clause
    (re.compile(rf"^(?:SUB-?CLAUSE|Sub-?clause)\s+{_NUM}\.?(?:\s.*)?$"), 4),
]

# ── Bahasa Indonesia patterns ────────────────────────────────────────────

_ID_PATTERNS: List[Tuple[re.Pattern, int]] = [
    # Level 0 – BAB (Chapter)
    (re.compile(rf"^BAB\s+{_NUM}\.?(?:\s.*)?$", re.IGNORECASE), 0),
    # Level 1 – Bagian (Part/Section)
    (re.compile(rf"^Bagian\s+(?:Kesatu|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|Kedelapan|Kesembilan|Kesepuluh|{_NUM})\.?(?:\s.*)?$", re.IGNORECASE), 1),
    # Level 2 – Paragraf (Paragraph/Sub-section)
    (re.compile(rf"^Paragraf\s+{_NUM}\.?(?:\s.*)?$", re.IGNORECASE), 2),
    # Level 3 – Pasal (Article)
    (re.compile(rf"^Pasal\s+{_NUM}\.?(?:\s.*)?$", re.IGNORECASE), 3),
    # Level 4 – Ayat (Verse/Clause) — usually inline, rarely standalone
    (re.compile(rf"^Ayat\s+\({_NUM}\)\.?(?:\s.*)?$", re.IGNORECASE), 4),
]

# ── Language → patterns map ──────────────────────────────────────────────

_LANG_PATTERNS: Dict[str, List[Tuple[re.Pattern, int]]] = {
    "en": _EN_PATTERNS,
    "id": _ID_PATTERNS,
}


def _match_heading(
    text: str, patterns: List[Tuple[re.Pattern, int]]
) -> Optional[int]:
    """Return the ``text_level`` if *text* matches any pattern, else ``None``."""
    stripped = text.strip()
    if not stripped:
        return None
    for pat, level in patterns:
        if pat.match(stripped):
            return level
    return None


def detect_legal_headings(
    pdf_list: List[Optional[Dict]],
    lang: str = "en",
) -> List[Optional[Dict]]:
    """Scan *pdf_list* and promote body-text items that match legal heading
    patterns to headings by setting their ``text_level``.

    Parameters
    ----------
    pdf_list:
        The pipeline's intermediate list of content dicts.
    lang:
        ISO 639-1 language code.  Falls back to English patterns if the
        requested language is not registered.

    Returns
    -------
    The same *pdf_list* (mutated in-place) with matched items promoted.
    """
    patterns = _LANG_PATTERNS.get(lang, _LANG_PATTERNS.get("en", []))
    if not patterns:
        log.warning("No legal heading patterns for lang='%s'; skipping.", lang)
        return pdf_list

    promoted = 0
    for content in pdf_list:
        if content is None:
            continue
        # Only consider body-text items (text_level == -1 or absent)
        if content.get("type") != "text":
            continue
        current_level = content.get("text_level", -1)
        if current_level >= 0:
            continue  # already a heading — don't override parser

        text = content.get("text", "")
        level = _match_heading(text, patterns)
        if level is not None:
            content["text_level"] = level
            promoted += 1
            log.debug("Promoted to heading (level %d): %s", level, text[:80])

    log.info(
        "Legal heading detection (lang=%s): promoted %d items to headings.",
        lang,
        promoted,
    )
    return pdf_list



# ---------------------------------------------------------------------------
# Automatic language detection from extracted text
# ---------------------------------------------------------------------------
_SUPPORTED_LANGS = {"en", "id", "de", "fr", "es", "pt", "it", "nl", "th", "zh", "ja", "ko", "ar"}


def detect_document_language(
    pdf_list: List[Optional[Dict]],
    fallback: str = "en",
    sample_chars: int = 2000,
) -> str:
    """Detect the dominant language of the document from its extracted text.

    Collects the first *sample_chars* characters of body text from *pdf_list*
    and runs ``langdetect`` on the sample.

    Parameters
    ----------
    pdf_list:
        The pipeline's intermediate list of content dicts.
    fallback:
        Language code to return when detection fails or ``langdetect`` is not
        installed.
    sample_chars:
        Maximum number of characters to sample for detection.

    Returns
    -------
    An ISO 639-1 language code (e.g. ``"en"``, ``"id"``).
    """
    if not _HAS_LANGDETECT:
        log.warning("langdetect is not installed; falling back to '%s'.", fallback)
        return fallback

    # Collect body text (text_level == -1 or absent)
    sample_parts: list[str] = []
    collected = 0
    for content in pdf_list:
        if content is None:
            continue
        if content.get("type") != "text":
            continue
        if content.get("text_level", -1) >= 0:
            continue  # skip headings — they may be too short / formulaic
        text = content.get("text", "").strip()
        if not text:
            continue
        sample_parts.append(text)
        collected += len(text)
        if collected >= sample_chars:
            break

    sample = " ".join(sample_parts)
    if len(sample) < 20:
        log.info("Not enough text for language detection; falling back to '%s'.", fallback)
        return fallback

    try:
        detected = _langdetect_detect(sample)
        # langdetect may return sub-tags like "zh-cn"; normalise
        lang = detected.split("-")[0].lower()
        if lang not in _SUPPORTED_LANGS:
            log.info(
                "Detected language '%s' is not supported; falling back to '%s'.",
                lang, fallback,
            )
            return fallback
        log.info("Auto-detected document language: '%s'.", lang)
        return lang
    except Exception as exc:
        log.warning("Language detection failed (%s); falling back to '%s'.", exc, fallback)
        return fallback