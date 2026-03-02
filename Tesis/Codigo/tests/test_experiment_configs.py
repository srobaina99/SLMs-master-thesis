"""
Tests for experiment configuration generation functions.

These verify the shape and correctness of all config factories.
No model loading is needed.

REFACTORING NOTE: update import line below after restructure:
  src.evaluation.experiment_framework  ->  src.framework
"""

import pytest

from src.framework.experiments.experiment_configs import (
    STANDARD_PROMPTS,
    MODEL_CONFIGS,
    create_factorial_configs,
    create_multi_weight_configs,
    create_beam_search_configs,
    get_config_by_name,
    get_configs_for_model,
)
from src.framework.core.data_models import ExperimentConfig


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD_PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStandardPrompts:

    def test_is_a_list(self):
        assert isinstance(STANDARD_PROMPTS, list)

    def test_all_items_are_strings(self):
        for p in STANDARD_PROMPTS:
            assert isinstance(p, str), f"Expected str, got {type(p)}: {p!r}"

    def test_no_empty_prompts(self):
        for p in STANDARD_PROMPTS:
            assert p.strip() != "", "Found empty prompt in STANDARD_PROMPTS"

    def test_has_at_least_one_prompt(self):
        assert len(STANDARD_PROMPTS) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# MODEL_CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

class TestModelConfigs:

    def test_contains_four_models(self):
        assert len(MODEL_CONFIGS) == 4

    def test_expected_model_names(self):
        expected = {"Phi3", "Qwen2", "Qwen3", "TinyLlama"}
        assert set(MODEL_CONFIGS.keys()) == expected

    def test_each_entry_has_model_name_and_model_id(self):
        for model_key, cfg in MODEL_CONFIGS.items():
            assert "model_name" in cfg, f"{model_key} missing model_name"
            assert "model_id" in cfg, f"{model_key} missing model_id"


# ══════════════════════════════════════════════════════════════════════════════
# create_factorial_configs
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateFactorialConfigs:

    def setup_method(self):
        self.configs = create_factorial_configs()

    def test_returns_list(self):
        assert isinstance(self.configs, list)

    def test_returns_experiment_config_instances(self):
        for c in self.configs:
            assert isinstance(c, ExperimentConfig)

    def test_total_count_is_16(self):
        # 4 models × 4 intervention combinations
        assert len(self.configs) == 16

    def test_all_four_models_represented(self):
        models = {c.model_name for c in self.configs}
        assert models == {"Phi3", "Qwen2", "Qwen3", "TinyLlama"}

    def test_each_model_has_four_configs(self):
        for model_name in ("Phi3", "Qwen2", "Qwen3", "TinyLlama"):
            model_configs = [c for c in self.configs if c.model_name == model_name]
            assert len(model_configs) == 4, (
                f"{model_name} should have 4 configs, got {len(model_configs)}"
            )

    def test_all_four_intervention_combinations_present_per_model(self):
        for model_name in ("Phi3", "Qwen2", "Qwen3", "TinyLlama"):
            model_configs = [c for c in self.configs if c.model_name == model_name]
            combos = {(c.config_weighting, c.config_prompting) for c in model_configs}
            expected_combos = {
                (False, False),  # control
                (True,  False),  # weighting only
                (False, True),   # prompting only
                (True,  True),   # both
            }
            assert combos == expected_combos, (
                f"{model_name} missing intervention combinations: "
                f"{expected_combos - combos}"
            )

    def test_experiment_names_are_unique(self):
        names = [c.experiment_name for c in self.configs]
        assert len(names) == len(set(names)), "Duplicate experiment names found"

    def test_control_config_naming(self):
        controls = [c for c in self.configs
                    if not c.config_weighting and not c.config_prompting]
        for c in controls:
            assert c.experiment_name.endswith("_control"), (
                f"Control config has unexpected name: {c.experiment_name}"
            )

    def test_weighted_only_config_naming(self):
        weighted_only = [c for c in self.configs
                         if c.config_weighting and not c.config_prompting]
        for c in weighted_only:
            assert "_weighted" in c.experiment_name

    def test_prompted_only_config_naming(self):
        prompted_only = [c for c in self.configs
                         if not c.config_weighting and c.config_prompting]
        for c in prompted_only:
            assert "_prompted" in c.experiment_name

    def test_both_config_naming(self):
        both = [c for c in self.configs
                if c.config_weighting and c.config_prompting]
        for c in both:
            assert "weighted_prompted" in c.experiment_name

    def test_weight_factor_is_1_5(self):
        # The optimal weight factor from the multi-weight experiment
        for c in self.configs:
            assert c.weight_factor == 1.5

    def test_max_new_tokens_is_200(self):
        for c in self.configs:
            assert c.max_new_tokens == 200

    def test_enable_thinking_is_false(self):
        for c in self.configs:
            assert c.enable_thinking is False


# ══════════════════════════════════════════════════════════════════════════════
# create_multi_weight_configs
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateMultiWeightConfigs:

    def test_default_returns_12_configs(self):
        # 4 models × 3 default weight factors [1.5, 2.0, 4.0]
        configs = create_multi_weight_configs()
        assert len(configs) == 12

    def test_custom_weight_factors_count(self):
        configs = create_multi_weight_configs(weight_factors=[1.0, 2.0])
        assert len(configs) == 8  # 4 models × 2 factors

    def test_single_weight_factor(self):
        configs = create_multi_weight_configs(weight_factors=[3.0])
        assert len(configs) == 4  # 4 models × 1 factor

    def test_all_configs_have_weighting_enabled(self):
        configs = create_multi_weight_configs()
        for c in configs:
            assert c.config_weighting is True
            assert c.weighted_words_enabled is True

    def test_weight_factors_are_assigned_correctly(self):
        factors = [1.0, 2.5]
        configs = create_multi_weight_configs(weight_factors=factors)
        assigned_factors = sorted({c.weight_factor for c in configs})
        assert assigned_factors == sorted(factors)

    def test_all_four_models_represented(self):
        configs = create_multi_weight_configs()
        models = {c.model_name for c in configs}
        assert models == {"Phi3", "Qwen2", "Qwen3", "TinyLlama"}

    def test_returns_experiment_config_instances(self):
        configs = create_multi_weight_configs()
        for c in configs:
            assert isinstance(c, ExperimentConfig)

    def test_experiment_names_are_unique(self):
        configs = create_multi_weight_configs()
        names = [c.experiment_name for c in configs]
        assert len(names) == len(set(names))


# ══════════════════════════════════════════════════════════════════════════════
# create_beam_search_configs
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateBeamSearchConfigs:

    def test_returns_exactly_two_configs(self):
        configs = create_beam_search_configs()
        assert len(configs) == 2

    def test_configs_are_experiment_config_instances(self):
        configs = create_beam_search_configs()
        for c in configs:
            assert isinstance(c, ExperimentConfig)

    def test_both_selection_methods_present(self):
        configs = create_beam_search_configs()
        names = [c.experiment_name for c in configs]
        assert any("a1_ratio" in name for name in names)
        assert any("max_probability" in name for name in names)

    def test_all_configs_use_qwen3(self):
        configs = create_beam_search_configs()
        for c in configs:
            assert c.model_name == "Qwen3"

    def test_weighting_is_disabled(self):
        configs = create_beam_search_configs()
        for c in configs:
            assert c.config_weighting is False

    def test_prompting_enabled_by_default(self):
        configs = create_beam_search_configs(use_prompting=True)
        for c in configs:
            assert c.config_prompting is True

    def test_prompting_disabled_when_requested(self):
        configs = create_beam_search_configs(use_prompting=False)
        for c in configs:
            assert c.config_prompting is False

    def test_beam_width_in_description(self):
        configs = create_beam_search_configs(beam_width=8)
        for c in configs:
            assert "8" in c.description


# ══════════════════════════════════════════════════════════════════════════════
# get_config_by_name
# ══════════════════════════════════════════════════════════════════════════════

class TestGetConfigByName:

    def test_finds_known_config(self):
        config = get_config_by_name("Qwen3_control")
        assert config is not None
        assert config.experiment_name == "Qwen3_control"

    def test_finds_weighted_config(self):
        config = get_config_by_name("Qwen3_weighted")
        assert config.config_weighting is True
        assert config.config_prompting is False

    def test_finds_prompted_config(self):
        config = get_config_by_name("Qwen3_prompted")
        assert config.config_prompting is True
        assert config.config_weighting is False

    def test_finds_both_config(self):
        config = get_config_by_name("Qwen3_weighted_prompted")
        assert config.config_weighting is True
        assert config.config_prompting is True

    def test_raises_value_error_for_unknown_name(self):
        with pytest.raises(ValueError, match="not found"):
            get_config_by_name("NonExistent_config_xyz")

    def test_all_16_config_names_resolvable(self):
        all_configs = create_factorial_configs()
        for c in all_configs:
            found = get_config_by_name(c.experiment_name)
            assert found.experiment_name == c.experiment_name


# ══════════════════════════════════════════════════════════════════════════════
# get_configs_for_model
# ══════════════════════════════════════════════════════════════════════════════

class TestGetConfigsForModel:

    def test_returns_four_configs_for_each_model(self):
        for model_name in ("Phi3", "Qwen2", "Qwen3", "TinyLlama"):
            configs = get_configs_for_model(model_name)
            assert len(configs) == 4, (
                f"Expected 4 configs for {model_name}, got {len(configs)}"
            )

    def test_all_configs_belong_to_requested_model(self):
        configs = get_configs_for_model("Qwen3")
        for c in configs:
            assert c.model_name == "Qwen3"

    def test_returns_empty_for_unknown_model(self):
        configs = get_configs_for_model("NonExistentModel")
        assert configs == []
