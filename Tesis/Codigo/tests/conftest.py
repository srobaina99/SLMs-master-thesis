"""
Shared pytest configuration and fixtures.

This file is auto-loaded by pytest before any test file.
It adds Codigo/ to sys.path so that `import src.X` always works
regardless of where pytest is invoked from.

REFACTORING NOTE:
  After the restructure described in thoughts/shared/plans/2026-02-20-codigo-restructure.md,
  update the import strings below and in every test file:
    src.evaluation.experiment_framework.X  ->  src.framework.X
    src.evaluation.text_complexity.X       ->  src.text_complexity.X
"""

import sys
import os
import pytest
from datetime import datetime

# ── Path setup ─────────────────────────────────────────────────────────────────
# conftest.py is at  Codigo/tests/conftest.py
# Going up two levels gives Codigo/
CODIGO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODIGO_DIR not in sys.path:
    sys.path.insert(0, CODIGO_DIR)

# ── Lazy imports (after path is set) ───────────────────────────────────────────
from src.framework.core.data_models import (
    ExperimentConfig,
    ExperimentResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

SIMPLE_A1_TEXT = (
    "A dog is a friendly animal. Dogs like to play and run outside. "
    "They are fun pets for children."
)

COMPLEX_TEXT = (
    "The unprecedented proliferation of computational methodologies has fundamentally "
    "reconstituted the epistemological frameworks underlying contemporary pedagogical "
    "discourse in second-language acquisition research."
)


@pytest.fixture
def sample_config():
    """A minimal ExperimentConfig for the Qwen3 control condition."""
    return ExperimentConfig(
        model_name="Qwen3",
        model_id="ggml-org/Qwen3-0.6B-GGUF",
        system_prompt="You are a helpful English teacher for beginner students.",
        config_weighting=False,
        config_prompting=False,
        weighted_words_enabled=False,
        weight_factor=1.5,
        enable_thinking=False,
        verbose=False,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=200,
        experiment_name="Qwen3_control",
        description="Control condition",
        prompt_id="P1",
    )


@pytest.fixture
def weighted_config(sample_config):
    """ExperimentConfig with weighting intervention active."""
    sample_config.config_weighting = True
    sample_config.weighted_words_enabled = True
    sample_config.weight_factor = 1.5
    sample_config.experiment_name = "Qwen3_weighted"
    return sample_config


@pytest.fixture
def prompted_config(sample_config):
    """ExperimentConfig with prompting intervention active."""
    sample_config.config_prompting = True
    sample_config.experiment_name = "Qwen3_prompted"
    return sample_config


@pytest.fixture
def both_config(sample_config):
    """ExperimentConfig with both interventions active."""
    sample_config.config_weighting = True
    sample_config.weighted_words_enabled = True
    sample_config.config_prompting = True
    sample_config.experiment_name = "Qwen3_weighted_prompted"
    return sample_config


@pytest.fixture
def simple_text():
    """Simple A1-level English text."""
    return SIMPLE_A1_TEXT


@pytest.fixture
def complex_text():
    """Complex academic text (high grade level)."""
    return COMPLEX_TEXT


@pytest.fixture
def canned_text_metrics():
    """
    Pre-computed metrics dict matching the structure returned by
    TextEvaluator.evaluate_text_comprehensive().
    """
    return {
        "grade_level_indices": {
            "flesch_kincaid_grade": 2.5,
            "gunning_fog": 3.1,
        },
        "readability_scores": {
            "spache_readability": 1.8,
        },
        "text_statistics": {
            "word_count": 22,
            "difficult_words": 1,
        },
    }


@pytest.fixture
def sample_result(sample_config, canned_text_metrics):
    """A fully-populated ExperimentResult for use in pipeline tests."""
    return ExperimentResult.create_from_response(
        prompt="What is a dog?",
        response=SIMPLE_A1_TEXT,
        config=sample_config,
        response_time=1.23,
        text_metrics=canned_text_metrics,
        experiment_name="Qwen3_control",
        cleaned_response=SIMPLE_A1_TEXT,
    )
