"""
Tests for ResponseFormatter.

These are pure unit tests — no models, no I/O.
They must pass before AND after the restructure (only import paths change).

REFACTORING NOTE: update import line below after restructure:
  src.evaluation.text_complexity  ->  src.text_complexity
"""

import pytest
from src.text_complexity.response_formatter import (
    ResponseFormatter,
    clean_for_evaluation,
    clean_preserving_structure,
)


class TestResponseFormatterInstantiation:

    def test_instantiation(self):
        formatter = ResponseFormatter()
        assert formatter is not None

    def test_has_all_patterns(self):
        formatter = ResponseFormatter()
        assert formatter.emoji_pattern is not None
        assert formatter.arrow_pattern is not None
        assert formatter.bullet_pattern is not None
        assert formatter.whitespace_pattern is not None
        assert formatter.markdown_pattern is not None


class TestRemoveEmojis:

    def setup_method(self):
        self.formatter = ResponseFormatter()

    def test_removes_smiley(self):
        result = self.formatter.remove_emojis("Hello! 😊 How are you?")
        assert "😊" not in result
        assert "Hello!" in result

    def test_removes_multiple_emojis(self):
        result = self.formatter.remove_emojis("🎉 Great work! 🔥 Amazing! 🌟")
        assert "🎉" not in result
        assert "🔥" not in result
        assert "🌟" not in result

    def test_preserves_plain_text(self):
        text = "This is a plain sentence without emojis."
        result = self.formatter.remove_emojis(text)
        assert result == text

    def test_empty_string(self):
        result = self.formatter.remove_emojis("")
        assert result == ""


class TestRemoveArrowsAndBullets:

    def setup_method(self):
        self.formatter = ResponseFormatter()

    def test_removes_arrow(self):
        result = self.formatter.remove_arrows_and_bullets("Step one → Step two")
        assert "→" not in result

    def test_removes_bullet(self):
        result = self.formatter.remove_arrows_and_bullets("• First item\n• Second item")
        assert "•" not in result

    def test_preserves_words(self):
        result = self.formatter.remove_arrows_and_bullets("• First item → Second item")
        assert "First" in result
        assert "Second" in result

    def test_empty_string(self):
        result = self.formatter.remove_arrows_and_bullets("")
        assert result == ""


class TestNormalizeWhitespace:

    def setup_method(self):
        self.formatter = ResponseFormatter()

    def test_collapses_multiple_spaces(self):
        result = self.formatter.normalize_whitespace("Hello   world")
        assert result == "Hello world"

    def test_collapses_tabs(self):
        result = self.formatter.normalize_whitespace("Hello\t\tworld")
        assert result == "Hello world"

    def test_collapses_newlines(self):
        result = self.formatter.normalize_whitespace("Hello\n\n\nworld")
        assert result == "Hello world"

    def test_strips_leading_and_trailing(self):
        result = self.formatter.normalize_whitespace("   hello   ")
        assert result == "hello"

    def test_empty_string(self):
        result = self.formatter.normalize_whitespace("")
        assert result == ""


class TestRemoveMarkdownFormatting:

    def setup_method(self):
        self.formatter = ResponseFormatter()

    def test_removes_bold_markers(self):
        result = self.formatter.remove_markdown_formatting("**Bold text** here")
        assert "**" not in result
        assert "Bold" in result

    def test_removes_italic_markers(self):
        result = self.formatter.remove_markdown_formatting("*italic* word")
        assert "*" not in result
        assert "italic" in result

    def test_removes_backticks(self):
        result = self.formatter.remove_markdown_formatting("`code snippet`")
        assert "`" not in result
        assert "code" in result

    def test_removes_hash_signs(self):
        result = self.formatter.remove_markdown_formatting("## Heading")
        assert "#" not in result
        assert "Heading" in result


class TestCleanResponseForEvaluation:

    def setup_method(self):
        self.formatter = ResponseFormatter()

    def test_removes_emojis(self):
        result = self.formatter.clean_response_for_evaluation("Hello 😊 world!")
        assert "😊" not in result

    def test_removes_arrows(self):
        result = self.formatter.clean_response_for_evaluation("Step one → Step two")
        assert "→" not in result

    def test_removes_markdown(self):
        result = self.formatter.clean_response_for_evaluation("**Bold** and *italic*")
        assert "**" not in result
        assert "*" not in result

    def test_normalizes_whitespace(self):
        result = self.formatter.clean_response_for_evaluation("Hello   world")
        assert "  " not in result  # No double spaces

    def test_empty_string_returns_empty(self):
        result = self.formatter.clean_response_for_evaluation("")
        assert result == ""

    def test_whitespace_only_returns_empty(self):
        result = self.formatter.clean_response_for_evaluation("   \n\t  ")
        assert result == ""

    def test_plain_text_is_preserved(self):
        text = "A dog is a friendly animal."
        result = self.formatter.clean_response_for_evaluation(text)
        # Core words should still be present
        assert "dog" in result
        assert "friendly" in result
        assert "animal" in result

    def test_complex_messy_response(self):
        messy = "🔥 **Great** answer! → Here is the explanation:\n\n• Dogs are friendly\n• Cats are calm"
        result = self.formatter.clean_response_for_evaluation(messy)
        assert "🔥" not in result
        assert "**" not in result
        assert "→" not in result
        assert "•" not in result
        assert "Dogs" in result
        assert "Cats" in result


class TestCleanPreservingStructure:

    def setup_method(self):
        self.formatter = ResponseFormatter()

    def test_removes_emojis(self):
        result = self.formatter.clean_preserving_structure("Hello 😊 world!")
        assert "😊" not in result

    def test_preserves_newlines(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = self.formatter.clean_preserving_structure(text)
        # Should preserve double newlines (paragraph breaks)
        assert "\n" in result

    def test_empty_string_returns_empty(self):
        result = self.formatter.clean_preserving_structure("")
        assert result == ""

    def test_plain_text_preserved(self):
        text = "A dog is a friendly animal."
        result = self.formatter.clean_preserving_structure(text)
        assert "dog" in result


class TestConvenienceFunctions:

    def test_clean_for_evaluation(self):
        result = clean_for_evaluation("Hello 😊 world")
        assert "😊" not in result
        assert "Hello" in result

    def test_clean_preserving_structure(self):
        result = clean_preserving_structure("Hello 😊 world")
        assert "😊" not in result
        assert "Hello" in result
