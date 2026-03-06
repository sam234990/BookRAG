"""Tests for language-aware is_likely_incomplete_paragraph in pdf_refiner.py"""
import pytest

from Core.pipelines.pdf_refiner import is_likely_incomplete_paragraph


class TestEnglishIncomplete:
    """Existing English behaviour should be preserved."""

    def test_complete_sentence(self):
        assert is_likely_incomplete_paragraph(
            'He said, "This method is the best."', lang="en"
        ) is False

    def test_incomplete_ending_and(self):
        assert is_likely_incomplete_paragraph(
            "The quick brown fox jumps over the lazy dog and", lang="en"
        ) is True

    def test_incomplete_hyphen(self):
        assert is_likely_incomplete_paragraph(
            "The results demonstrate a signifi-", lang="en"
        ) is True

    def test_incomplete_comma(self):
        assert is_likely_incomplete_paragraph(
            "In the following sections, we discuss the approach,", lang="en"
        ) is True

    def test_complete_exclamation(self):
        assert is_likely_incomplete_paragraph(
            "This is absolutely correct for all cases!", lang="en"
        ) is False

    def test_short_text_not_incomplete(self):
        assert is_likely_incomplete_paragraph("Hello", lang="en") is False

    def test_empty_text(self):
        assert is_likely_incomplete_paragraph("", lang="en") is False

    def test_connector_word_the(self):
        assert is_likely_incomplete_paragraph(
            "This regulation applies to all persons under the", lang="en"
        ) is True


class TestIndonesianIncomplete:
    """Indonesian-specific terminal punctuation and connector words."""

    def test_complete_sentence(self):
        assert is_likely_incomplete_paragraph(
            "Peraturan ini berlaku sejak tanggal diundangkan.", lang="id"
        ) is False

    def test_incomplete_no_period(self):
        assert is_likely_incomplete_paragraph(
            "Dalam peraturan pemerintah ini yang dimaksud dengan peraturan", lang="id"
        ) is True

    def test_incomplete_connector_dan(self):
        assert is_likely_incomplete_paragraph(
            "Pasal ini mengatur tentang hak dan.", lang="id"
        ) is True

    def test_incomplete_connector_yang(self):
        assert is_likely_incomplete_paragraph(
            "Setiap orang berhak atas perlindungan hukum yang.", lang="id"
        ) is True

    def test_incomplete_connector_dengan(self):
        assert is_likely_incomplete_paragraph(
            "Peraturan ini disusun dengan memperhatikan ketentuan dengan.", lang="id"
        ) is True

    def test_complete_question(self):
        assert is_likely_incomplete_paragraph(
            "Apakah peraturan ini sudah sesuai dengan undang-undang?", lang="id"
        ) is False

    def test_incomplete_comma_id(self):
        assert is_likely_incomplete_paragraph(
            "Sebagaimana dimaksud dalam Pasal 1 ayat satu,", lang="id"
        ) is True


class TestDefaultLang:
    """When lang is omitted, should behave as English."""

    def test_defaults_to_english(self):
        assert is_likely_incomplete_paragraph(
            "The quick brown fox jumps over the lazy dog and"
        ) is True

    def test_defaults_complete(self):
        assert is_likely_incomplete_paragraph(
            "This sentence is complete and well-formed."
        ) is False


class TestUnsupportedLang:
    """Unsupported language should fall back to English rules."""

    def test_fallback_terminal_punctuation(self):
        # No terminal punctuation → incomplete even for unknown lang
        assert is_likely_incomplete_paragraph(
            "This sentence has no ending punctuation mark", lang="xx"
        ) is True

    def test_fallback_complete(self):
        assert is_likely_incomplete_paragraph(
            "This sentence ends properly with a period.", lang="xx"
        ) is False

