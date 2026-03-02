"""
Tests for core data models: ExperimentConfig, ExperimentResult, ExperimentDataManager.

These tests are pure unit tests — no model loading, no file I/O beyond tmp_path.
They must pass before AND after the restructure (only import paths change).

REFACTORING NOTE: update import line below after restructure:
  src.evaluation.experiment_framework  ->  src.framework
"""

import os
import pytest
from datetime import datetime

from src.framework.core.data_models import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentDataManager,
)


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentConfig:

    def test_default_instantiation(self):
        config = ExperimentConfig()
        assert config.model_name == "Qwen3"
        assert config.config_weighting is False
        assert config.config_prompting is False
        assert config.weight_factor == 1.0
        assert config.temperature == 0.7
        assert config.max_new_tokens == 200

    def test_custom_instantiation(self, sample_config):
        assert sample_config.model_name == "Qwen3"
        assert sample_config.experiment_name == "Qwen3_control"
        assert sample_config.prompt_id == "P1"

    def test_to_dict_returns_dict(self, sample_config):
        d = sample_config.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_fields(self, sample_config):
        d = sample_config.to_dict()
        expected_keys = [
            "model_name", "model_id", "system_prompt",
            "config_weighting", "config_prompting",
            "weighted_words_enabled", "weight_factor",
            "temperature", "top_k", "top_p", "max_new_tokens",
            "experiment_name", "description", "prompt_id",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_weighting_intervention_flags(self, weighted_config):
        assert weighted_config.config_weighting is True
        assert weighted_config.weighted_words_enabled is True

    def test_prompting_intervention_flags(self, prompted_config):
        assert prompted_config.config_prompting is True

    def test_both_interventions(self, both_config):
        assert both_config.config_weighting is True
        assert both_config.config_prompting is True

    def test_prompt_id_mutable(self, sample_config):
        sample_config.prompt_id = "P5"
        assert sample_config.prompt_id == "P5"


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentResult
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentResult:

    def test_create_from_response_returns_instance(self, sample_config, canned_text_metrics):
        result = ExperimentResult.create_from_response(
            prompt="What is a dog?",
            response="A dog is a pet.",
            config=sample_config,
            response_time=0.5,
            text_metrics=canned_text_metrics,
            experiment_name="Qwen3_control",
            cleaned_response="A dog is a pet.",
        )
        assert isinstance(result, ExperimentResult)

    def test_create_from_response_copies_config_fields(self, sample_config, canned_text_metrics):
        result = ExperimentResult.create_from_response(
            prompt="What is a dog?",
            response="A dog is a pet.",
            config=sample_config,
            response_time=0.5,
            text_metrics=canned_text_metrics,
            experiment_name="Qwen3_control",
            cleaned_response="A dog is a pet.",
        )
        assert result.model == "Qwen3"
        assert result.model_id == sample_config.model_id
        assert result.config_weighting is False
        assert result.config_prompting is False
        assert result.weight_factor == 1.5
        assert result.temperature == 0.7
        assert result.prompt_id == "P1"

    def test_create_from_response_stores_prompt_and_response(self, sample_config, canned_text_metrics):
        result = ExperimentResult.create_from_response(
            prompt="What is a dog?",
            response="A dog is a pet.",
            config=sample_config,
            response_time=0.5,
            text_metrics=canned_text_metrics,
            experiment_name="Qwen3_control",
            cleaned_response="A dog is a pet.",
        )
        assert result.prompt == "What is a dog?"
        assert result.response == "A dog is a pet."
        assert result.cleaned_response == "A dog is a pet."

    def test_create_from_response_stores_metrics(self, sample_config, canned_text_metrics):
        result = ExperimentResult.create_from_response(
            prompt="What is a dog?",
            response="A dog is a pet.",
            config=sample_config,
            response_time=0.5,
            text_metrics=canned_text_metrics,
            experiment_name="Qwen3_control",
            cleaned_response="A dog is a pet.",
        )
        assert result.flesch_kincaid_grade == 2.5
        assert result.gunning_fog == 3.1
        assert result.smog_index == 1.0
        assert result.spache_readability == 1.8
        assert result.word_count == 22
        assert result.difficult_words == 1

    def test_create_from_response_stores_timing(self, sample_config, canned_text_metrics):
        result = ExperimentResult.create_from_response(
            prompt="Q",
            response="A",
            config=sample_config,
            response_time=2.71,
            text_metrics=canned_text_metrics,
            experiment_name="x",
            cleaned_response="A",
        )
        assert result.response_time_seconds == pytest.approx(2.71)

    def test_create_from_response_has_uuid(self, sample_config, canned_text_metrics):
        r1 = ExperimentResult.create_from_response(
            prompt="Q", response="A", config=sample_config,
            response_time=0.1, text_metrics=canned_text_metrics,
            experiment_name="x", cleaned_response="A",
        )
        r2 = ExperimentResult.create_from_response(
            prompt="Q", response="A", config=sample_config,
            response_time=0.1, text_metrics=canned_text_metrics,
            experiment_name="x", cleaned_response="A",
        )
        assert r1.experiment_id != r2.experiment_id

    def test_create_from_response_has_timestamp(self, sample_config, canned_text_metrics):
        result = ExperimentResult.create_from_response(
            prompt="Q", response="A", config=sample_config,
            response_time=0.1, text_metrics=canned_text_metrics,
            experiment_name="x", cleaned_response="A",
        )
        assert isinstance(result.timestamp, datetime)

    def test_create_from_response_empty_metrics(self, sample_config):
        empty_metrics = {
            "grade_level_indices": {},
            "readability_scores": {},
            "text_statistics": {},
        }
        result = ExperimentResult.create_from_response(
            prompt="Q", response="", config=sample_config,
            response_time=0.0, text_metrics=empty_metrics,
            experiment_name="x", cleaned_response="",
        )
        assert result.flesch_kincaid_grade == 0.0
        assert result.word_count == 0

    def test_to_dict_serializes_timestamp_as_string(self, sample_result):
        d = sample_result.to_dict()
        assert isinstance(d["timestamp"], str)
        # Should be ISO format parseable
        datetime.fromisoformat(d["timestamp"])

    def test_to_dict_contains_all_primary_metric_keys(self, sample_result):
        d = sample_result.to_dict()
        primary_keys = [
            "flesch_kincaid_grade", "gunning_fog", "smog_index",
            "spache_readability", "word_count", "difficult_words",
        ]
        for key in primary_keys:
            assert key in d, f"Missing key: {key}"

    def test_create_from_beam_response_sets_beam_fields(self, sample_config, canned_text_metrics):
        result = ExperimentResult.create_from_beam_response(
            prompt="Q",
            response="A simple answer.",
            config=sample_config,
            response_time=0.9,
            text_metrics=canned_text_metrics,
            experiment_name="beam_test",
            cleaned_response="A simple answer.",
            beam_selection_method="a1_ratio",
            beam_a1_ratio=0.75,
            beam_a1_count=6,
            beam_content_word_count=8,
            beam_cumulative_logprob=-12.3,
            beam_width=4,
        )
        assert result.beam_selection_method == "a1_ratio"
        assert result.beam_a1_ratio == pytest.approx(0.75)
        assert result.beam_a1_count == 6
        assert result.beam_content_word_count == 8
        assert result.beam_cumulative_logprob == pytest.approx(-12.3)
        assert result.beam_width == 4

    def test_regular_result_beam_fields_are_none(self, sample_result):
        assert sample_result.beam_selection_method is None
        assert sample_result.beam_a1_ratio is None
        assert sample_result.beam_width is None


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentDataManager
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentDataManager:

    def test_empty_manager_has_no_results(self):
        manager = ExperimentDataManager()
        assert len(manager.results) == 0

    def test_add_result(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        assert len(manager.results) == 1

    def test_add_results_bulk(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_results([sample_result, sample_result])
        assert len(manager.results) == 2

    def test_clear_empties_results(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        manager.clear()
        assert len(manager.results) == 0

    def test_to_dataframe_empty(self):
        manager = ExperimentDataManager()
        df = manager.to_dataframe()
        assert df.empty

    def test_to_dataframe_shape(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        manager.add_result(sample_result)
        df = manager.to_dataframe()
        assert len(df) == 2

    def test_to_dataframe_has_model_column(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        df = manager.to_dataframe()
        assert "model" in df.columns
        assert df["model"].iloc[0] == "Qwen3"

    def test_save_to_csv_creates_file(self, sample_result, tmp_path):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        csv_path = str(tmp_path / "test_output.csv")
        manager.save_to_csv(csv_path)
        assert os.path.exists(csv_path)
        assert os.path.getsize(csv_path) > 0

    def test_save_to_csv_empty_does_not_crash(self, tmp_path):
        manager = ExperimentDataManager()
        csv_path = str(tmp_path / "empty.csv")
        manager.save_to_csv(csv_path)  # Should not crash
        assert not os.path.exists(csv_path)  # Empty: nothing saved

    def test_export_to_csv_specification_format_creates_file(self, sample_result, tmp_path):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        csv_path = str(tmp_path / "spec_output.csv")
        manager.export_to_csv_specification_format(csv_path)
        assert os.path.exists(csv_path)

    def test_export_to_csv_specification_format_has_required_columns(self, sample_result, tmp_path):
        import pandas as pd
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        csv_path = str(tmp_path / "spec_output.csv")
        manager.export_to_csv_specification_format(csv_path)
        # Read back (spec format uses comma as decimal separator)
        df = pd.read_csv(csv_path)
        required_cols = [
            "model", "config_weighting", "config_prompting", "weight_factor", "prompt_id",
            "answer", "time_spent",
            "flesch_kincaid_grade", "gunning_fog", "smog_index",
            "spache_readability",
            "word_count", "difficult_words",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing specification column: {col}"

    def test_get_summary_stats_empty(self):
        manager = ExperimentDataManager()
        stats = manager.get_summary_stats()
        assert stats == {}

    def test_get_summary_stats_structure(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        stats = manager.get_summary_stats()
        assert "overall" in stats
        assert "by_config" in stats
        assert "metadata" in stats

    def test_get_summary_stats_total_count(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        manager.add_result(sample_result)
        stats = manager.get_summary_stats()
        assert stats["metadata"]["total_experiments"] == 2

    def test_get_summary_stats_overall_contains_metric_keys(self, sample_result):
        manager = ExperimentDataManager()
        manager.add_result(sample_result)
        stats = manager.get_summary_stats()
        expected_metrics = [
            "flesch_kincaid_grade", "gunning_fog", "smog_index",
            "spache_readability", "word_count",
        ]
        for metric in expected_metrics:
            assert metric in stats["overall"], f"Missing metric in overall stats: {metric}"
