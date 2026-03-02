"""
Tests for TextEvaluator.

These tests rely only on textstat (and optionally nltk), no LLM models needed.
They must pass before AND after the restructure (only import paths change).

REFACTORING NOTE: update import line below after restructure:
  src.evaluation.text_complexity  ->  src.text_complexity
"""

import pytest
from src.text_complexity.text_evaluator import TextEvaluator


# Keep texts SHORT — textstat is fast but we want millisecond-level tests
SIMPLE_TEXT = "A dog is a friendly animal. Dogs like to play."
COMPLEX_TEXT = "The unprecedented proliferation of computational methodologies reconstituted epistemological frameworks."

A1_VOCAB = {"dog", "cat", "run", "play", "happy", "big", "small", "fun", "animal", "pet"}


# ══════════════════════════════════════════════════════════════════════════════
# Instantiation
# ══════════════════════════════════════════════════════════════════════════════

class TestTextEvaluatorInstantiation:

    def test_default_instantiation(self):
        evaluator = TextEvaluator()
        assert evaluator is not None
        assert evaluator.tokenizer is None

    def test_instantiation_with_tokenizer(self):
        tokenizer = lambda text: text.split()
        evaluator = TextEvaluator(tokenizer=tokenizer)
        assert evaluator.tokenizer is not None


# ══════════════════════════════════════════════════════════════════════════════
# get_grade_level_indices
# ══════════════════════════════════════════════════════════════════════════════

class TestGetGradeLevelIndices:

    def setup_method(self):
        self.evaluator = TextEvaluator()

    def test_returns_dict(self):
        result = self.evaluator.get_grade_level_indices(SIMPLE_TEXT)
        assert isinstance(result, dict)

    def test_returns_all_three_keys(self):
        result = self.evaluator.get_grade_level_indices(SIMPLE_TEXT)
        assert "flesch_kincaid_grade" in result
        assert "gunning_fog" in result
        assert "smog_index" in result

    def test_values_are_floats(self):
        result = self.evaluator.get_grade_level_indices(SIMPLE_TEXT)
        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} is not a float"

    def test_empty_text_returns_zeros(self):
        result = self.evaluator.get_grade_level_indices("")
        assert result["flesch_kincaid_grade"] == 0.0
        assert result["gunning_fog"] == 0.0
        assert result["smog_index"] == 0.0

    def test_whitespace_only_returns_zeros(self):
        result = self.evaluator.get_grade_level_indices("   \n\t  ")
        assert result["flesch_kincaid_grade"] == 0.0

    def test_simple_text_lower_grade_than_complex(self):
        simple = self.evaluator.get_grade_level_indices(SIMPLE_TEXT)
        complex_ = self.evaluator.get_grade_level_indices(COMPLEX_TEXT)
        # Complex academic text should have higher Flesch-Kincaid grade
        assert simple["flesch_kincaid_grade"] < complex_["flesch_kincaid_grade"]

    def test_simple_text_lower_fog_than_complex(self):
        simple = self.evaluator.get_grade_level_indices(SIMPLE_TEXT)
        complex_ = self.evaluator.get_grade_level_indices(COMPLEX_TEXT)
        assert simple["gunning_fog"] < complex_["gunning_fog"]

    def test_values_are_rounded_to_two_decimals(self):
        result = self.evaluator.get_grade_level_indices(SIMPLE_TEXT)
        for key, value in result.items():
            if value != 0.0:
                # Check that value has at most 2 decimal places
                assert round(value, 2) == value, f"{key}={value} has more than 2 decimal places"


# ══════════════════════════════════════════════════════════════════════════════
# get_readability_scores
# ══════════════════════════════════════════════════════════════════════════════

class TestGetReadabilityScores:

    def setup_method(self):
        self.evaluator = TextEvaluator()

    def test_returns_dict(self):
        result = self.evaluator.get_readability_scores(SIMPLE_TEXT)
        assert isinstance(result, dict)

    def test_returns_spache_key(self):
        result = self.evaluator.get_readability_scores(SIMPLE_TEXT)
        assert "spache_readability" in result

    def test_spache_is_float(self):
        result = self.evaluator.get_readability_scores(SIMPLE_TEXT)
        assert isinstance(result["spache_readability"], (int, float))

    def test_empty_text_returns_zero(self):
        result = self.evaluator.get_readability_scores("")
        assert result["spache_readability"] == 0.0

    def test_simple_text_lower_spache_than_complex(self):
        simple = self.evaluator.get_readability_scores(SIMPLE_TEXT)
        complex_ = self.evaluator.get_readability_scores(COMPLEX_TEXT)
        assert simple["spache_readability"] < complex_["spache_readability"]


# ══════════════════════════════════════════════════════════════════════════════
# get_text_statistics
# ══════════════════════════════════════════════════════════════════════════════

class TestGetTextStatistics:

    def setup_method(self):
        self.evaluator = TextEvaluator()

    def test_returns_dict(self):
        result = self.evaluator.get_text_statistics(SIMPLE_TEXT)
        assert isinstance(result, dict)

    def test_returns_word_count_and_difficult_words(self):
        result = self.evaluator.get_text_statistics(SIMPLE_TEXT)
        assert "word_count" in result
        assert "difficult_words" in result

    def test_word_count_is_int(self):
        result = self.evaluator.get_text_statistics(SIMPLE_TEXT)
        assert isinstance(result["word_count"], int)

    def test_difficult_words_is_int(self):
        result = self.evaluator.get_text_statistics(SIMPLE_TEXT)
        assert isinstance(result["difficult_words"], int)

    def test_empty_text_returns_zeros(self):
        result = self.evaluator.get_text_statistics("")
        assert result["word_count"] == 0
        assert result["difficult_words"] == 0

    def test_word_count_positive_for_real_text(self):
        result = self.evaluator.get_text_statistics(SIMPLE_TEXT)
        assert result["word_count"] > 0

    def test_complex_text_has_more_difficult_words_than_simple(self):
        simple = self.evaluator.get_text_statistics(SIMPLE_TEXT)
        complex_ = self.evaluator.get_text_statistics(COMPLEX_TEXT)
        assert complex_["difficult_words"] > simple["difficult_words"]

    def test_token_count_present_when_tokenizer_provided(self):
        tokenizer = lambda text: text.split()
        evaluator = TextEvaluator(tokenizer=tokenizer)
        result = evaluator.get_text_statistics(SIMPLE_TEXT)
        assert "token_count" in result
        assert result["token_count"] > 0

    def test_token_count_absent_without_tokenizer(self):
        result = self.evaluator.get_text_statistics(SIMPLE_TEXT)
        assert "token_count" not in result


# ══════════════════════════════════════════════════════════════════════════════
# evaluate_text_comprehensive
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluateTextComprehensive:

    def setup_method(self):
        self.evaluator = TextEvaluator()

    def test_returns_dict_with_three_sections(self):
        result = self.evaluator.evaluate_text_comprehensive(SIMPLE_TEXT)
        assert "grade_level_indices" in result
        assert "readability_scores" in result
        assert "text_statistics" in result

    def test_empty_text_returns_zero_valued_sections(self):
        result = self.evaluator.evaluate_text_comprehensive("")
        assert result["grade_level_indices"]["flesch_kincaid_grade"] == 0.0
        assert result["readability_scores"]["spache_readability"] == 0.0
        assert result["text_statistics"]["word_count"] == 0

    def test_full_result_matches_individual_methods(self):
        full = self.evaluator.evaluate_text_comprehensive(SIMPLE_TEXT)
        grades = self.evaluator.get_grade_level_indices(SIMPLE_TEXT)
        read = self.evaluator.get_readability_scores(SIMPLE_TEXT)
        stats = self.evaluator.get_text_statistics(SIMPLE_TEXT)

        assert full["grade_level_indices"] == grades
        assert full["readability_scores"] == read
        assert full["text_statistics"] == stats

    def test_is_deterministic(self):
        r1 = self.evaluator.evaluate_text_comprehensive(SIMPLE_TEXT)
        r2 = self.evaluator.evaluate_text_comprehensive(SIMPLE_TEXT)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════════════════
# extract_content_words
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractContentWords:

    def setup_method(self):
        self.evaluator = TextEvaluator()

    def test_returns_set(self):
        result = self.evaluator.extract_content_words(SIMPLE_TEXT)
        assert isinstance(result, set)

    def test_non_empty_for_real_text(self):
        result = self.evaluator.extract_content_words(SIMPLE_TEXT)
        assert len(result) > 0

    def test_common_content_words_included(self):
        # "dog" and "friendly" and "animal" are content words
        result = self.evaluator.extract_content_words("A dog is a friendly animal.")
        # At least one of these should be found
        assert any(w in result for w in {"dog", "friendly", "animal"})

    def test_function_words_excluded(self):
        result = self.evaluator.extract_content_words("The cat is in the box.")
        # Common function words should not appear
        assert "the" not in result
        assert "is" not in result
        assert "in" not in result

    def test_empty_text_returns_empty_set(self):
        result = self.evaluator.extract_content_words("")
        assert result == set() or len(result) == 0

    def test_caching_consistent(self):
        r1 = self.evaluator.extract_content_words(SIMPLE_TEXT)
        r2 = self.evaluator.extract_content_words(SIMPLE_TEXT)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════════════════
# calculate_a1_word_ratio
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateA1WordRatio:

    def setup_method(self):
        self.evaluator = TextEvaluator()

    def test_returns_tuple_of_three(self):
        result = self.evaluator.calculate_a1_word_ratio(SIMPLE_TEXT, A1_VOCAB)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ratio_between_zero_and_one(self):
        ratio, a1_count, content_count = self.evaluator.calculate_a1_word_ratio(
            SIMPLE_TEXT, A1_VOCAB
        )
        assert 0.0 <= ratio <= 1.0

    def test_counts_are_non_negative_ints(self):
        ratio, a1_count, content_count = self.evaluator.calculate_a1_word_ratio(
            SIMPLE_TEXT, A1_VOCAB
        )
        assert isinstance(a1_count, int) and a1_count >= 0
        assert isinstance(content_count, int) and content_count >= 0

    def test_empty_vocab_gives_zero_ratio(self):
        ratio, a1_count, content_count = self.evaluator.calculate_a1_word_ratio(
            SIMPLE_TEXT, set()
        )
        assert ratio == 0.0
        assert a1_count == 0

    def test_empty_text_returns_zero_tuple(self):
        ratio, a1_count, content_count = self.evaluator.calculate_a1_word_ratio(
            "", A1_VOCAB
        )
        assert ratio == 0.0
        assert a1_count == 0
        assert content_count == 0

    def test_full_vocab_match_gives_high_ratio(self):
        # Text whose content words are all in A1 vocab
        text = "The dog and cat run and play."
        # "dog", "cat", "run", "play" are in A1_VOCAB
        ratio, a1_count, content_count = self.evaluator.calculate_a1_word_ratio(
            text, A1_VOCAB
        )
        assert ratio > 0.0

    def test_a1_count_leq_content_count(self):
        ratio, a1_count, content_count = self.evaluator.calculate_a1_word_ratio(
            SIMPLE_TEXT, A1_VOCAB
        )
        assert a1_count <= content_count

    def test_ratio_equals_a1_count_over_content_count(self):
        ratio, a1_count, content_count = self.evaluator.calculate_a1_word_ratio(
            SIMPLE_TEXT, A1_VOCAB
        )
        if content_count > 0:
            assert ratio == pytest.approx(a1_count / content_count)
