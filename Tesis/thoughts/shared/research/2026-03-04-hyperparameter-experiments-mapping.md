---
date: 2026-03-04T12:00:00-03:00
researcher: claude
git_commit: 2fe39cb
branch: feature/refactor
repository: SLMs-master-thesis
topic: "Mapping HYPERPARAMETER_EXPERIMENTS.md to current codebase"
tags: [research, codebase, experiments, hyperparameters, beam-search, logit-bias, prompting]
status: complete
last_updated: 2026-03-04
last_updated_by: claude
---

# Research: How Hyperparameter Experiments Map to Current Codebase

**Date**: 2026-03-04
**Git Commit**: 2fe39cb
**Branch**: feature/refactor

## Research Question

How would the three hyperparameter experiments from `docs/HYPERPARAMETER_EXPERIMENTS.md` (prompting strategy, beam search width, logit bias weight) fit into the current refactored codebase?

## Summary

The current codebase **already supports two of the three experiments** (logit bias weight and beam search width) through existing infrastructure. The prompting strategy experiment (zero/one/few-shot) is the only one that requires new code, specifically modifications to how `_add_simplification_context()` works. Below is a detailed mapping from the old document's concepts to the current code.

---

## Detailed Findings

### 1. Prompting Strategy Experiment (Zero/One/Few-Shot)

**Status: Requires new code**

**Current state:**
- The prompting intervention is a boolean flag: `config.config_prompting` (True/False)
- When enabled, `BaseModelWrapper._add_simplification_context()` (`src/framework/models/base_model.py:117-133`) prepends a fixed zero-shot context block to the prompt
- There is **no mechanism** to vary the number of shots or change the prompt template per experiment

**What the HYPERPARAMETER_EXPERIMENTS.md describes:**
- Zero-shot: Context instruction only (this is what the current code does)
- One-shot: Context + 1 example Q&A
- Few-shot: Context + 3 example Q&As

**Mapping to current code:**
- `_add_simplification_context()` would need to accept a `shots` parameter (0, 1, 3) or be replaced with a configurable prompt builder
- `ExperimentConfig` (`src/framework/core/data_models.py:14-47`) would need a new field (e.g., `num_shots: int = 0` or `prompt_template: str = "zero_shot"`)
- A new config factory function (like `create_prompting_experiment_configs()`) would be needed in `experiment_configs.py`
- A new experiment runner method (like `run_prompting_experiment()`) in `FactorialExperiment` would iterate over shot counts

**Key files to modify:**
- `src/framework/models/base_model.py` — `_add_simplification_context()` method
- `src/framework/core/data_models.py` — `ExperimentConfig` dataclass
- `src/framework/experiments/experiment_configs.py` — new config factory
- `src/framework/experiments/factorial_experiment.py` — new runner method

---

### 2. Beam Search Width Experiment (n=4, 8, 10)

**Status: Already supported, just needs a parameterized script**

**Current state:**
- `BeamSearchGenerator` (`src/framework/models/beam_search_generator.py:24-213`) accepts `beam_width` as a constructor parameter
- `Qwen3LlamaCppWrapper.generate_with_beam_search()` (`src/framework/models/qwen3_llamacpp_wrapper.py:143-279`) takes `beam_width` as an argument
- `FactorialExperiment.run_beam_search_experiment()` (`src/framework/experiments/factorial_experiment.py:561-702`) takes `beam_width` parameter
- `create_beam_search_configs()` (`src/framework/experiments/experiment_configs.py:146-211`) takes `beam_width` parameter
- The existing script `scripts/run_beam_search_experiment.py` hardcodes `beam_width=8`
- `ExperimentResult` already has `beam_width`, `beam_a1_ratio`, `beam_a1_count`, etc. fields (`data_models.py:96-102`)
- Results already exist in `results/Qwen3/` from previous beam search runs

**What the HYPERPARAMETER_EXPERIMENTS.md describes:**
- Test beam widths: 4, 8, 10
- Use contextual prompting (zero-shot), disable logit bias
- Selection method: highest A1 ratio

**Mapping to current code:**
- The experiment infrastructure is **fully built**. The only thing needed is a script that loops over `beam_width=[4, 8, 10]` calling `experiment.run_beam_search_experiment(beam_width=N)` for each
- The current `run_beam_search_experiment()` already tests both selection methods (a1_ratio and max_probability) per beam width
- Beam search is only implemented on `Qwen3LlamaCppWrapper` — not available for other models

**Key observation:** The beam search generates `beam_width` independent samples (stochastic sampling per beam), not traditional deterministic beam search. Each "beam" is an independent generation with temperature/top_p/top_k sampling (`beam_search_generator.py:74-83`). The cumulative log prob is approximated, not exact (`beam_search_generator.py:93-98`).

---

### 3. Logit Bias Weight Experiment (1.0 to 4.0)

**Status: Already fully supported**

**Current state:**
- `ExperimentConfig.weight_factor` (`data_models.py:28`) stores the weight factor
- `LlamaCppBaseWrapper._create_logit_bias()` (`llamacpp_base.py:183-208`) creates the logit_bias dict from vocab + weight_factor
- `_generate_response_impl()` (`llamacpp_base.py:210-289`) applies logit_bias when `config.config_weighting` is True
- `create_multi_weight_configs()` (`experiment_configs.py:214-254`) creates configs for multiple weight factors across all models
- `FactorialExperiment.run_multi_weight_experiment()` (`factorial_experiment.py:421-559`) runs the full multi-weight sweep
- `ExperimentRunner.run_multi_weight_experiment()` (`experiment_runner.py:215-258`) provides the public API
- `scripts/run_experiment.py` supports `--experiment multi_weight --weights 1.0,1.3,1.5,2.0,2.5,3.0,4.0`

**What the HYPERPARAMETER_EXPERIMENTS.md describes:**
- Test weight factors: 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0
- Logit bias = log(weight_factor)
- Contextual prompting enabled, beam search disabled

**Mapping to current code:**
- This experiment can be run **today** with: `python scripts/run_experiment.py --experiment multi_weight --weights 1.0,1.3,1.5,2.0,2.5,3.0,4.0 --prompts 5`
- **Important discrepancy**: The doc says logit_bias = `log(weight_factor)`, but the current code passes `weight_factor` directly as the logit_bias value (`llamacpp_base.py:207`: `logit_bias[token_id] = weight_factor`). This means weight_factor=1.5 applies a logit bias of +1.5, not +0.41 as the doc implies
- **Another discrepancy**: The doc says prompting should be enabled, but `create_multi_weight_configs()` sets `config_prompting=False` (weighting only). To test "both" (weighting + prompting), the config factory would need modification

---

## Architecture Documentation

### Experiment Flow

```
run_experiment.py (CLI entry point)
  → ExperimentRunner (simplified public API)
    → FactorialExperiment (actual experiment loop)
      → _get_model() → lazy-loads model wrapper
      → model_wrapper.generate_response(prompt, config)
        → _add_simplification_context() if config_prompting
        → _format_prompt() (ChatML template)
        → _create_logit_bias() if config_weighting
        → llm() call via llama.cpp
      → TextEvaluator.evaluate_text_comprehensive()
      → ExperimentResult.create_from_response()
      → ExperimentDataManager.add_result()
    → save_results() → CSV, JSON files to results/{ModelName}/
```

### Config Structure

```
ExperimentConfig:
  model_name, model_id          # Which model
  config_weighting: bool        # Enable logit_bias
  config_prompting: bool        # Enable context prefix
  weight_factor: float          # Logit bias strength (default 1.5)
  temperature, top_k, top_p     # Generation params
  max_new_tokens: int           # Max output tokens
  prompt_id: str                # Prompt identifier (P1, P2, ...)
```

### Results Storage

Results go to `results/{ModelName}/`:
- `{prefix}_specification_{timestamp}.csv` — reduced columns, European decimals
- `{prefix}_summary_{timestamp}.json` — aggregated stats
- `full_data/{prefix}_full_{timestamp}.csv` — all columns

Beam search results also stored in `results/Qwen3/` with beam-specific columns.

---

## Code References

- `src/framework/models/base_model.py:117-133` — `_add_simplification_context()` (prompting intervention)
- `src/framework/models/llamacpp_base.py:183-208` — `_create_logit_bias()` (weighting intervention)
- `src/framework/models/llamacpp_base.py:210-289` — `_generate_response_impl()` (generation with interventions)
- `src/framework/models/beam_search_generator.py:24-213` — `BeamSearchGenerator` class
- `src/framework/models/qwen3_llamacpp_wrapper.py:143-279` — `generate_with_beam_search()` (Qwen3-specific)
- `src/framework/core/data_models.py:14-47` — `ExperimentConfig` dataclass
- `src/framework/core/data_models.py:96-102` — Beam search fields in `ExperimentResult`
- `src/framework/experiments/experiment_configs.py:146-211` — `create_beam_search_configs()`
- `src/framework/experiments/experiment_configs.py:214-254` — `create_multi_weight_configs()`
- `src/framework/experiments/factorial_experiment.py:421-559` — `run_multi_weight_experiment()`
- `src/framework/experiments/factorial_experiment.py:561-702` — `run_beam_search_experiment()`
- `scripts/run_experiment.py` — CLI entry point with `--experiment multi_weight --weights` support
- `scripts/run_beam_search_experiment.py` — Beam search script (hardcoded width=8)

## Open Questions

1. **Logit bias calculation**: The doc says `log(weight_factor)` but current code uses `weight_factor` directly. Which is intended?
2. **Prompting + weighting**: The doc's logit bias experiment uses "Both" (weighting + prompting), but `create_multi_weight_configs()` only enables weighting. Should it enable both?
3. **Beam search on other models**: Currently only `Qwen3LlamaCppWrapper` has `generate_with_beam_search()`. Should this be generalized to `LlamaCppBaseWrapper`?
4. **Beam search implementation**: The current implementation generates N independent samples, not traditional beam search. Is this intentional or should token-level beam search be implemented?
