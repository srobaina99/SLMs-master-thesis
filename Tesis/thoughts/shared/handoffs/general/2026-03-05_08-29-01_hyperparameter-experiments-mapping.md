---
date: 2026-03-05T08:29:01+0000
researcher: claude
git_commit: 2fe39cb
branch: feature/refactor
repository: SLMs-master-thesis
topic: "Hyperparameter Experiments Mapping to Current Codebase"
tags: [research, experiments, hyperparameters, beam-search, logit-bias, prompting]
status: complete
last_updated: 2026-03-05
last_updated_by: claude
type: implementation_strategy
---

# Handoff: Hyperparameter Experiments Codebase Mapping

## Task(s)

**Research (completed):** Read `Tesis/Codigo/docs/HYPERPARAMETER_EXPERIMENTS.md` and mapped its three proposed experiments to the current refactored codebase. The doc was written for an older iteration of the code, so the goal was to understand which experiments are already supported vs. what needs new code.

No code was written — this was a pure research session.

## Critical References

- `Tesis/Codigo/docs/HYPERPARAMETER_EXPERIMENTS.md` — The spec doc describing the three experiments (source of truth for what needs to be implemented)
- `Tesis/thoughts/shared/research/2026-03-04-hyperparameter-experiments-mapping.md` — Full research doc produced this session with all findings

## Recent Changes

No code changes made this session. One research document was produced:
- `Tesis/thoughts/shared/research/2026-03-04-hyperparameter-experiments-mapping.md` (new file)

## Learnings

### Experiment 1: Logit Bias Weight (1.0 to 4.0)
- **Status: Fully supported today.** Run with `python scripts/run_experiment.py --experiment multi_weight --weights 1.0,1.3,1.5,2.0,2.5,3.0,4.0 --prompts 5`
- **Discrepancy 1:** The doc says `logit_bias = log(weight_factor)`, but `llamacpp_base.py:207` applies `weight_factor` directly as the bias value (additive to raw logit pre-softmax). `weight_factor=1.5` → `+1.5` logit shift ≈ `e^1.5 ≈ 4.5x` probability multiplier.
- **Discrepancy 2:** `create_multi_weight_configs()` at `experiment_configs.py:214-254` sets `config_prompting=False`. The doc expects prompting to be enabled alongside weighting. The factorial "both" condition is separate from the multi-weight sweep.
- The bias is flat (same `+weight_factor` for all A1 vocab tokens, every generation step), built by `_create_logit_bias()` at `llamacpp_base.py:183-208`.

### Experiment 2: Beam Search Width (n=4, 8, 10)
- **Status: Infrastructure fully built, just needs a parameterized script.**
- All relevant classes accept `beam_width` as a parameter: `BeamSearchGenerator.__init__()`, `generate_with_beam_search()`, `run_beam_search_experiment()`, `create_beam_search_configs()`.
- The existing `scripts/run_beam_search_experiment.py` hardcodes `beam_width=8` — needs a loop.
- **Important:** The "beam search" is not token-level beam search. It runs N independent stochastic samples (`beam_search_generator.py:74-83`) and selects the best by A1 ratio or max cumulative log prob. The log prob is approximated, not exact (`beam_search_generator.py:93-98`).
- Only `Qwen3LlamaCppWrapper` has `generate_with_beam_search()` — not available for other models.

### Experiment 3: Prompting Strategy (Zero/One/Few-Shot)
- **Status: Requires new code.**
- Currently prompting is binary (on/off). When on, `_add_simplification_context()` at `base_model.py:117-133` prepends a fixed zero-shot context string.
- No mechanism for 1-shot or few-shot examples exists anywhere in the codebase.
- To implement: add `num_shots` field to `ExperimentConfig` (`data_models.py:14-47`), modify `_add_simplification_context()` to accept a shot count and inject example Q&As, add config factory in `experiment_configs.py`, add runner method in `factorial_experiment.py`.

### Architecture Snapshot
```
ExperimentRunner (public API, experiment_runner.py)
  → FactorialExperiment (loops, factorial_experiment.py)
    → lazy-loads model wrappers (_model_classes dict)
    → model_wrapper.generate_response(prompt, config)
      → _add_simplification_context() if config_prompting
      → _format_prompt() (ChatML/Llama2/etc)
      → _create_logit_bias() if config_weighting
      → llama.cpp call with logit_bias
    → TextEvaluator → ExperimentResult → ExperimentDataManager
  → save_results() → results/{ModelName}/
```

## Artifacts

- `Tesis/thoughts/shared/research/2026-03-04-hyperparameter-experiments-mapping.md` — Full mapping research doc

## Action Items & Next Steps

1. **Clarify logit bias formula** — Decide whether `weight_factor` should be applied as-is or as `log(weight_factor)` (matches the doc). If it should be log, change `llamacpp_base.py:207`.

2. **Run logit bias weight experiment** — Already works. Just run:
   ```
   python scripts/run_experiment.py --experiment multi_weight --weights 1.0,1.3,1.5,2.0,2.5,3.0,4.0 --prompts 5
   ```

3. **Create beam width sweep script** — Modify `scripts/run_beam_search_experiment.py` (or create a new script) to loop over `beam_width=[4, 8, 10]` calling `experiment.run_beam_search_experiment(beam_width=N, ...)` for each, and save results with width in the filename.

4. **Implement prompting strategy experiment** — This is the only experiment requiring new code:
   - Add `num_shots: int = 0` to `ExperimentConfig` (`data_models.py`)
   - Modify `_add_simplification_context()` (`base_model.py:117-133`) to accept shot count and embed example Q&As
   - Add `create_prompting_experiment_configs()` to `experiment_configs.py`
   - Add `run_prompting_experiment()` to `factorial_experiment.py`

5. **Decide on multi-weight + prompting** — The doc's logit bias experiment implies both interventions enabled. Update `create_multi_weight_configs()` (`experiment_configs.py:214-254`) to set `config_prompting=True` if that's the intent.

## Other Notes

- Results directory structure: `results/{ModelName}/{prefix}_specification_{MMDD_HHMM}.csv`. The `ModelName` is inferred from the `filename_prefix` argument at `factorial_experiment.py:340`.
- Beam search results already exist in `results/Qwen3/` from previous runs (files dated 1111, 1120, 0301).
- The `ExperimentResult` dataclass already has all beam fields (`beam_width`, `beam_a1_ratio`, `beam_a1_count`, `beam_content_word_count`, `beam_cumulative_logprob`, `beam_selection_method`) at `data_models.py:96-102` — no schema changes needed for the beam width experiment.
- All models currently use llama.cpp GGUF backends (Qwen2, Qwen3, TinyLlama, Phi3). The `SmolLM` model name appears in the runner but has no wrapper class defined.
