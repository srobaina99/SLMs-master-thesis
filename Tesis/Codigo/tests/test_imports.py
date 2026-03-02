"""
Import smoke tests — verifies that all public modules are importable.

These tests WILL FAIL during the refactoring (when modules are being moved)
and MUST BE UPDATED after the restructure is complete.

Running this test after the restructure (with updated paths) confirms nothing was lost.
"""

import importlib
import pytest


# ── Helper ─────────────────────────────────────────────────────────────────────

def assert_importable(module_path: str):
    """Assert that a module can be imported without error."""
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        pytest.fail(f"Cannot import '{module_path}': {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Core data models
# ══════════════════════════════════════════════════════════════════════════════

def test_import_data_models():
    assert_importable("src.framework.core.data_models")


def test_import_experiment_runner():
    assert_importable("src.framework.core.experiment_runner")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment definitions
# ══════════════════════════════════════════════════════════════════════════════

def test_import_experiment_configs():
    assert_importable("src.framework.experiments.experiment_configs")


def test_import_factorial_experiment():
    assert_importable("src.framework.experiments.factorial_experiment")


# ══════════════════════════════════════════════════════════════════════════════
# Model wrappers
# ══════════════════════════════════════════════════════════════════════════════

def test_import_base_model():
    assert_importable("src.framework.models.base_model")


def test_import_llamacpp_base():
    assert_importable("src.framework.models.llamacpp_base")


def test_import_qwen3_wrapper():
    assert_importable("src.framework.models.qwen3_llamacpp_wrapper")


def test_import_qwen2_wrapper():
    assert_importable("src.framework.models.qwen2_llamacpp_wrapper")


def test_import_phi3_wrapper():
    assert_importable("src.framework.models.phi3_llamacpp_wrapper")


def test_import_tinyllama_wrapper():
    assert_importable("src.framework.models.tinyllama_llamacpp_wrapper")


# ══════════════════════════════════════════════════════════════════════════════
# Text complexity
# ══════════════════════════════════════════════════════════════════════════════

def test_import_text_evaluator():
    assert_importable("src.text_complexity.text_evaluator")


def test_import_response_formatter():
    assert_importable("src.text_complexity.response_formatter")


# ══════════════════════════════════════════════════════════════════════════════
# Public names exported by each module
# ══════════════════════════════════════════════════════════════════════════════

def test_data_models_exports_experiment_config():
    from src.framework.core.data_models import ExperimentConfig
    assert ExperimentConfig is not None


def test_data_models_exports_experiment_result():
    from src.framework.core.data_models import ExperimentResult
    assert ExperimentResult is not None


def test_data_models_exports_experiment_data_manager():
    from src.framework.core.data_models import ExperimentDataManager
    assert ExperimentDataManager is not None


def test_experiment_configs_exports_standard_prompts():
    from src.framework.experiments.experiment_configs import STANDARD_PROMPTS
    assert isinstance(STANDARD_PROMPTS, list)


def test_experiment_configs_exports_create_factorial_configs():
    from src.framework.experiments.experiment_configs import create_factorial_configs
    assert callable(create_factorial_configs)


def test_text_evaluator_exports_class():
    from src.text_complexity.text_evaluator import TextEvaluator
    assert TextEvaluator is not None


def test_response_formatter_exports_class():
    from src.text_complexity.response_formatter import ResponseFormatter
    assert ResponseFormatter is not None


def test_base_model_exports_base_model_wrapper():
    from src.framework.models.base_model import BaseModelWrapper
    assert BaseModelWrapper is not None
