"""Tests for Core/pipelines/legal_heading_detector.py"""
import pytest

from Core.pipelines.legal_heading_detector import (
    detect_legal_headings,
    detect_document_language,
    _match_heading,
    _EN_PATTERNS,
    _ID_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_item(text: str, text_level: int = -1, item_type: str = "text"):
    return {"type": item_type, "text": text, "text_level": text_level}


# ---------------------------------------------------------------------------
# English heading pattern matching
# ---------------------------------------------------------------------------
class TestEnglishPatterns:
    @pytest.mark.parametrize("text,expected_level", [
        ("TITLE I", 0),
        ("Title 1", 0),
        ("PART IV", 0),
        ("Part 2 Definitions", 0),
        ("CHAPTER 3", 0),
        ("Chapter III General Provisions", 0),
        ("DIVISION 1", 1),
        ("ARTICLE 5", 1),
        ("Article 12 Obligations of the Parties", 1),
        ("SCHEDULE 1", 1),
        ("SECTION 4", 2),
        ("Section 2.1 Scope", 2),  # fails — no dot-numbers in _NUM
        ("Annex A", 2),
        ("CLAUSE 7", 3),
        ("§ 12", 3),
        ("§ 3 Definitions", 3),
        ("SUB-CLAUSE 2", 4),
        ("Sub-clause 1", 4),
    ])
    def test_match(self, text, expected_level):
        result = _match_heading(text, _EN_PATTERNS)
        assert result == expected_level, f"Expected level {expected_level} for '{text}', got {result}"

    @pytest.mark.parametrize("text", [
        "This is a normal paragraph.",
        "The Article discusses legal matters.",
        "See Chapter 3 for more details.",
        "",
        "article",  # no number
    ])
    def test_no_match(self, text):
        assert _match_heading(text, _EN_PATTERNS) is None


# ---------------------------------------------------------------------------
# Indonesian heading pattern matching
# ---------------------------------------------------------------------------
class TestIndonesianPatterns:
    @pytest.mark.parametrize("text,expected_level", [
        ("BAB I", 0),
        ("BAB IV KETENTUAN PERALIHAN", 0),
        ("Bagian Kesatu Umum", 1),
        ("Bagian Kedua Ruang Lingkup", 1),
        ("Paragraf 1", 2),
        ("Paragraf 2 Tata Cara", 2),
        ("Pasal 1", 3),
        ("Pasal 45", 3),
        ("Ayat (1)", 4),
        ("Ayat (2) Ketentuan", 4),
    ])
    def test_match(self, text, expected_level):
        result = _match_heading(text, _ID_PATTERNS)
        assert result == expected_level, f"Expected level {expected_level} for '{text}', got {result}"

    @pytest.mark.parametrize("text", [
        "Mengenai pasal ini perlu diperhatikan.",
        "Lihat BAB sebelumnya.",
        "",
    ])
    def test_no_match(self, text):
        assert _match_heading(text, _ID_PATTERNS) is None


# ---------------------------------------------------------------------------
# detect_legal_headings integration
# ---------------------------------------------------------------------------
class TestDetectLegalHeadings:
    def test_promotes_body_text_en(self):
        pdf_list = [
            _make_item("CHAPTER 1 Introduction"),
            _make_item("This is body text about the law."),
            _make_item("Article 2 Definitions"),
            _make_item("Some table content", item_type="table"),
        ]
        result = detect_legal_headings(pdf_list, lang="en")
        assert result[0]["text_level"] == 0  # CHAPTER → level 0
        assert result[1]["text_level"] == -1  # body text unchanged
        assert result[2]["text_level"] == 1  # Article → level 1
        assert "text_level" not in result[3] or result[3]["text_level"] == -1  # table skipped

    def test_does_not_override_existing_heading(self):
        pdf_list = [_make_item("CHAPTER 1", text_level=2)]
        detect_legal_headings(pdf_list, lang="en")
        assert pdf_list[0]["text_level"] == 2  # not overridden

    def test_promotes_body_text_id(self):
        pdf_list = [
            _make_item("BAB I KETENTUAN UMUM"),
            _make_item("Pasal 1"),
            _make_item("Dalam peraturan ini yang dimaksud dengan:"),
        ]
        result = detect_legal_headings(pdf_list, lang="id")
        assert result[0]["text_level"] == 0
        assert result[1]["text_level"] == 3
        assert result[2]["text_level"] == -1

    def test_unknown_lang_falls_back_to_en(self):
        pdf_list = [_make_item("CHAPTER 1")]
        detect_legal_headings(pdf_list, lang="xx")
        assert pdf_list[0]["text_level"] == 0

    def test_handles_none_items(self):
        pdf_list = [None, _make_item("Article 1"), None]
        detect_legal_headings(pdf_list, lang="en")
        assert pdf_list[1]["text_level"] == 1


# ---------------------------------------------------------------------------
# Auto language detection
# ---------------------------------------------------------------------------
class TestDetectDocumentLanguage:
    def test_detects_english(self):
        pdf_list = [
            _make_item("The quick brown fox jumps over the lazy dog. " * 10),
            _make_item("This is a legal agreement between the parties. " * 10),
        ]
        lang = detect_document_language(pdf_list)
        assert lang == "en"

    def test_detects_indonesian(self):
        pdf_list = [
            _make_item("Dalam peraturan pemerintah ini yang dimaksud dengan "
                        "peraturan perundang-undangan adalah peraturan tertulis "
                        "yang memuat norma hukum yang mengikat secara umum. " * 5),
        ]
        lang = detect_document_language(pdf_list)
        assert lang == "id"

    def test_fallback_on_empty(self):
        pdf_list = [_make_item("Hi")]
        lang = detect_document_language(pdf_list, fallback="id")
        assert lang == "id"

    def test_skips_headings(self):
        pdf_list = [
            _make_item("BAB I", text_level=0),
            _make_item("Dalam peraturan ini yang dimaksud dengan peraturan "
                        "perundang-undangan adalah norma hukum yang berlaku. " * 5),
        ]
        lang = detect_document_language(pdf_list)
        # Should detect from body text, not headings
        assert lang == "id"

