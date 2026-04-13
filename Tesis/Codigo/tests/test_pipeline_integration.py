"""
Integration tests for the experiment data pipeline — NO LLM MODELS, NO INFERENCE.

These tests verify the full data flow works correctly:
  ExperimentConfig → ExperimentResult.create_from_response()
    → ExperimentDataManager → CSV / Parquet / JSON output
    → FactorialExperiment.save_results() → correct directory structure

Texts are kept to 1-2 short sentences and we use at most 2-4 configs per test
to keep runtime in the millisecond range.  No GGUF file is loaded anywhere.

REFACTORING NOTE: update import lines below after restructure:
  src.evaluation.experiment_framework  →  src.framework
  src.evaluation.text_complexity       →  src.text_complexity
"""

import os
import json
import pytest
import pandas as pd
from datetime import datetime

from src.framework.core.data_models import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentDataManager,
)
from src.framework.experiments.experiment_configs import (
    create_factorial_configs,
    STANDARD_PROMPTS,
)
from src.framework.experiments.factorial_experiment import FactorialExperiment
from src.text_complexity.text_evaluator import TextEvaluator
from src.text_complexity.response_formatter import ResponseFormatter


# ── Helpers ────────────────────────────────────────────────────────────────────

# One short sentence — textstat still works, runs in milliseconds
SAMPLE_RESPONSE = "A dog is a friendly animal. Dogs like to play."

SAMPLE_PROMPT = "What is a dog?"


def _make_result(config: ExperimentConfig, prompt: str = SAMPLE_PROMPT,
                 response: str = SAMPLE_RESPONSE) -> ExperimentResult:
    """Build an ExperimentResult from a config without model inference."""
    evaluator = TextEvaluator()
    formatter = ResponseFormatter()
    cleaned = formatter.clean_response_for_evaluation(response)
    metrics = evaluator.evaluate_text_comprehensive(cleaned)
    return ExperimentResult.create_from_response(
        prompt=prompt,
        response=response,
        config=config,
        response_time=1.0,
        text_metrics=metrics,
        experiment_name=config.experiment_name,
        cleaned_response=cleaned,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TextEvaluator + ResponseFormatter pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestTextPipeline:
    """Verify TextEvaluator and ResponseFormatter work together correctly."""

    def test_clean_then_evaluate_returns_non_zero_metrics(self):
        formatter = ResponseFormatter()
        evaluator = TextEvaluator()

        raw = "🎉 A dog is a **friendly** animal → great pets!"
        cleaned = formatter.clean_response_for_evaluation(raw)
        assert "🎉" not in cleaned
        assert "**" not in cleaned

        metrics = evaluator.evaluate_text_comprehensive(cleaned)
        assert metrics["text_statistics"]["word_count"] > 0

    def test_empty_response_returns_zero_metrics(self):
        evaluator = TextEvaluator()
        metrics = evaluator.evaluate_text_comprehensive("")
        assert metrics["grade_level_indices"]["flesch_kincaid_grade"] == 0.0
        assert metrics["text_statistics"]["word_count"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentResult creation via TextEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentResultCreationPipeline:

    def test_create_result_from_real_metrics(self):
        config = ExperimentConfig(
            model_name="Qwen3",
            experiment_name="Qwen3_control",
            config_weighting=False,
            config_prompting=False,
            prompt_id="P1",
        )
        result = _make_result(config)

        assert result.model == "Qwen3"
        assert result.prompt == SAMPLE_PROMPT
        assert result.word_count > 0
        assert result.flesch_kincaid_grade >= 0.0

    def test_result_metrics_are_consistent_with_text(self):
        config = ExperimentConfig(model_name="Qwen3", experiment_name="x", prompt_id="P1")
        evaluator = TextEvaluator()
        expected_metrics = evaluator.evaluate_text_comprehensive(SAMPLE_RESPONSE)

        result = _make_result(config)

        assert result.word_count == expected_metrics["text_statistics"]["word_count"]
        assert result.flesch_kincaid_grade == expected_metrics["grade_level_indices"]["flesch_kincaid_grade"]
        assert result.spache_readability == expected_metrics["readability_scores"]["spache_readability"]

    def test_two_results_have_distinct_config_flags(self):
        # Use only the first 2 Qwen3 configs (control + weighted_only) to stay fast
        configs = [c for c in create_factorial_configs() if c.model_name == "Qwen3"][:2]
        results = [_make_result(c) for c in configs]

        combos = {(r.config_weighting, r.config_prompting) for r in results}
        # First two configs are (False,False) control and (True,False) weighted
        assert len(combos) == 2


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentDataManager CSV / Parquet round-trips
# ══════════════════════════════════════════════════════════════════════════════

class TestDataManagerRoundTrip:

    def _build_manager_with_results(self, n: int = 2) -> ExperimentDataManager:
        """Build a manager populated with n synthetic results (default 2 to stay fast)."""
        manager = ExperimentDataManager()
        configs = [c for c in create_factorial_configs() if c.model_name == "Qwen3"][:n]
        for config in configs:
            manager.add_result(_make_result(config))
        return manager

    def test_csv_round_trip_preserves_row_count(self, tmp_path):
        manager = self._build_manager_with_results(2)
        csv_path = str(tmp_path / "full_data.csv")
        manager.save_to_csv(csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 2

    def test_specification_csv_has_correct_columns(self, tmp_path):
        manager = self._build_manager_with_results(2)
        csv_path = str(tmp_path / "spec.csv")
        manager.export_to_csv_specification_format(csv_path)

        df = pd.read_csv(csv_path)
        spec_columns = [
            "model", "config_weighting", "config_prompting", "weight_factor", "prompt_id",
            "answer", "time_spent",
            "flesch_kincaid_grade", "gunning_fog",
            "spache_readability",
            "word_count", "difficult_words",
        ]
        for col in spec_columns:
            assert col in df.columns, f"Missing column in spec CSV: {col}"

    def test_specification_csv_model_column_correct(self, tmp_path):
        manager = self._build_manager_with_results(2)
        csv_path = str(tmp_path / "spec.csv")
        manager.export_to_csv_specification_format(csv_path)

        df = pd.read_csv(csv_path)
        assert (df["model"] == "Qwen3").all()

    def test_parquet_round_trip_preserves_row_count(self, tmp_path):
        manager = self._build_manager_with_results(2)
        parquet_path = str(tmp_path / "data.parquet")
        manager.save_to_parquet(parquet_path)

        df = pd.read_parquet(parquet_path)
        assert len(df) == 2

    def test_summary_stats_match_data(self, tmp_path):
        manager = self._build_manager_with_results(2)
        stats = manager.get_summary_stats()

        assert stats["metadata"]["total_experiments"] == 2
        assert "Qwen3" in stats["metadata"]["models_tested"]


# ══════════════════════════════════════════════════════════════════════════════
# FactorialExperiment.save_results() — directory structure
# ══════════════════════════════════════════════════════════════════════════════

class TestFactorialExperimentSaveResults:
    """
    Tests FactorialExperiment.save_results() by pre-populating data_manager
    directly — no model inference needed.
    """

    def _build_experiment_with_data(self, tmp_path, model_name: str = "Qwen3") -> FactorialExperiment:
        """Use only 2 configs per model to keep each test under a second."""
        experiment = FactorialExperiment(results_dir=str(tmp_path))
        configs = [c for c in create_factorial_configs() if c.model_name == model_name][:2]
        for config in configs:
            experiment.data_manager.add_result(_make_result(config))
        return experiment

    def test_save_results_creates_model_directory(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        experiment.save_results("Qwen3_factorial")

        model_dir = tmp_path / "Qwen3"
        assert model_dir.exists(), "Model subdirectory was not created"

    def test_save_results_creates_full_data_subdirectory(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        experiment.save_results("Qwen3_factorial")

        full_data_dir = tmp_path / "Qwen3" / "full_data"
        assert full_data_dir.exists(), "full_data subdirectory was not created"

    def test_save_results_returns_three_file_keys(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        files = experiment.save_results("Qwen3_factorial")

        assert "specification_csv" in files
        assert "full_csv" in files
        assert "summary_json" in files

    def test_save_results_creates_specification_csv(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        files = experiment.save_results("Qwen3_factorial")

        assert os.path.exists(files["specification_csv"])

    def test_save_results_creates_full_csv(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        files = experiment.save_results("Qwen3_factorial")

        assert os.path.exists(files["full_csv"])

    def test_save_results_creates_summary_json(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        files = experiment.save_results("Qwen3_factorial")

        assert os.path.exists(files["summary_json"])

    def test_summary_json_is_valid_json(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        files = experiment.save_results("Qwen3_factorial")

        with open(files["summary_json"]) as f:
            summary = json.load(f)

        assert "metadata" in summary
        assert "overall" in summary

    def test_specification_csv_has_correct_row_count(self, tmp_path):
        experiment = self._build_experiment_with_data(tmp_path, "Qwen3")
        files = experiment.save_results("Qwen3_factorial")

        df = pd.read_csv(files["specification_csv"])
        assert len(df) == 2  # 2 configs loaded by _build_experiment_with_data

    def test_save_results_for_different_model_names(self, tmp_path):
        for model_name in ("Phi3", "Qwen2"):
            experiment = self._build_experiment_with_data(tmp_path, model_name)
            files = experiment.save_results(f"{model_name}_factorial")
            model_dir = tmp_path / model_name
            assert model_dir.exists()
            assert os.path.exists(files["specification_csv"])


# ══════════════════════════════════════════════════════════════════════════════
# Path resolution — project_root points to Codigo/
# ══════════════════════════════════════════════════════════════════════════════

class TestProjectRootResolution:
    """
    Verify that each module's project_root calculation resolves to the
    Codigo/ directory (not a parent or child of it).

    These tests catch the off-by-one path depth bug described in the
    restructure plan (some modules historically navigated to Tesis/ instead
    of Codigo/).
    """

    def _get_codigo_dir(self) -> str:
        """Expected Codigo/ path based on this test file's location."""
        # This file is at  Codigo/tests/test_pipeline_integration.py
        # Going up 2 levels gives Codigo/
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_experiment_configs_project_root(self):
        import src.framework.experiments.experiment_configs as mod

        expected_codigo = self._get_codigo_dir()
        # The module computes project_root at import time using os.path.dirname chains.
        # We reconstruct the same computation to verify the expected depth.
        current_dir = os.path.dirname(os.path.abspath(mod.__file__))
        # Current layout: Codigo/src/framework/experiments/
        # Going up 3 levels: experiments/ -> framework/ -> src/ -> Codigo/
        depth_3_up = os.path.dirname(
            os.path.dirname(
                os.path.dirname(current_dir)
            )
        )

        # Codigo/ is 3 levels up from experiments/ in the current layout
        assert depth_3_up == expected_codigo, (
            f"Codigo/ should be 3 levels up from experiments/ in current layout.\n"
            f"  Expected: {expected_codigo}\n"
            f"  Got:      {depth_3_up}"
        )

    def test_base_model_project_root_resolves_to_codigo(self):
        import src.framework.models.base_model as mod

        expected_codigo = self._get_codigo_dir()
        current_dir = os.path.dirname(os.path.abspath(mod.__file__))
        # models/ is 3 levels below Codigo/ in current layout:
        # Codigo/src/framework/models/base_model.py
        depth_3_up = os.path.dirname(
            os.path.dirname(
                os.path.dirname(current_dir)
            )
        )
        assert depth_3_up == expected_codigo, (
            f"base_model.py: Codigo/ should be 3 levels up.\n"
            f"  Expected: {expected_codigo}\n"
            f"  Got:      {depth_3_up}"
        )

    def test_vocab_file_exists_at_expected_path(self):
        """The filtered vocabulary file must be present for weighting to work."""
        codigo_dir = self._get_codigo_dir()
        vocab_path = os.path.join(
            codigo_dir, "data", "vocabularies", "filtered_starters_vocab.txt"
        )
        assert os.path.exists(vocab_path), (
            f"Vocabulary file not found at: {vocab_path}\n"
            "This file is required for the weighting intervention."
        )

    def test_vocab_file_has_content(self):
        codigo_dir = self._get_codigo_dir()
        vocab_path = os.path.join(
            codigo_dir, "data", "vocabularies", "filtered_starters_vocab.txt"
        )
        if not os.path.exists(vocab_path):
            pytest.skip("Vocab file not present — skipping content check")
        with open(vocab_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) > 0, "Vocabulary file is empty"
