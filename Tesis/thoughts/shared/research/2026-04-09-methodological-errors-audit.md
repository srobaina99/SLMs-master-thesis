---
date: 2026-04-09T00:00:00-03:00
researcher: Claude
git_commit: 18277f7216381a9944764e6009c116b87c4e62e4
branch: feature/refactor
repository: SLMs-master-thesis
topic: "Methodological errors audit of the experiment framework"
tags: [research, codebase, methodology, readability-metrics, logit-bias, reproducibility]
status: complete
last_updated: 2026-04-09
last_updated_by: Claude
---

# Research: Methodological Errors Audit

**Date**: 2026-04-09
**Git Commit**: 18277f7
**Branch**: feature/refactor

## Research Question

Are there methodological errors in the experimentation/research framework that could compromise the validity of results?

## Summary

Six issues found, two of which are **critical** and directly contaminate the primary experiment results used for the paper. The rest range from moderate (affects statistical validity) to minor (best-practice gaps).

## Critical Issues

### 1. Thinking Chain-of-Thought Contamination in Qwen3 Results

**Severity: CRITICAL** — Directly corrupts measured readability metrics.

In the main Qwen3 experiment file (`Qwen3_full_experiment_full_1023_0027.csv`), **19 out of 60 rows (32%)** have the model's internal chain-of-thought reasoning in the `cleaned_response` field — the exact text that readability metrics are computed on.

**Root cause**: Qwen3 produces `<think>...</think>` blocks. The `_extract_response()` method (`qwen3_llamacpp_wrapper.py:112-116`) only strips thinking if BOTH `<think>` and `</think>` tags are present. When the model hits `max_tokens=200` during the thinking phase (before generating `</think>`), the incomplete thinking text becomes the response. The `ResponseFormatter` (`response_formatter.py`) does not handle `<think>` tags at all.

**Impact on data**: All 19 contaminated rows have **zero** `</think>` tags — every one is an incomplete thinking block:

| Metric | Think-contaminated (n=19) | Clean (n=41) |
|--------|--------------------------|--------------|
| FK Grade | mean=5.11 | mean=4.29 |
| Gunning Fog | mean=6.33 | mean=5.46 |
| SMOG | mean=8.06 | mean=7.36 |
| Spache | mean=3.75 | mean=3.08 |
| Word count | mean=154.68 | mean=26.37 |

The contaminated rows inflate Qwen3's readability scores by ~0.7-1.0 grade levels and inflate word counts by 6x (because internal reasoning is verbose and uses complex language). This affects **all four intervention conditions** (control: 6, weighted: 6, prompted: 4, both: 3).

**Evidence**: `results/Qwen3/full_data/Qwen3_full_experiment_full_1023_0027.csv`, rows with `<think>` in cleaned_response.

**Note**: The `/nothink` fix was applied in the current working tree but was NOT present when the 1023 experiment data was generated. The post-fix runs (0301 files) all produced empty responses (model may need different parameters with `/nothink`).

### 2. Failed Generations Scored as Perfect Readability

**Severity: CRITICAL** — Introduces systematic bias toward interventions that cause failures.

When generation fails (timeout, error, empty response), `text_evaluator.py:169-174` returns `0.0` for all grade-level metrics. A grade level of 0.0 represents the easiest possible text — effectively "perfect" for A1 learners.

**Impact**: In `multi_weight_experiment_full_1007_0329.csv`, **86 out of 270 rows (32%)** have empty responses with all-zero metrics. If these rows are included in aggregate statistics (mean FK grade, etc.), they artificially lower the measured complexity, making failed configurations appear to produce simpler text.

**Evidence**: `results/multi/full_data/multi_weight_experiment_full_1007_0329.csv`, rows where `word_count == 0`.

## Moderate Issues

### 3. No Random Seed — Experiments Not Reproducible

**Severity: MODERATE** — Standard requirement for published research.

No random seed is set anywhere in the framework. The `Llama()` constructor (`llamacpp_base.py:100-106`) does not pass a `seed` parameter. The generation call (`llamacpp_base.py:250-259`) does not pass one either. With `temperature=0.7`, `top_k=50`, `top_p=0.95`, each run produces different outputs.

**Impact**: Results cannot be independently reproduced. Re-running the same experiment produces different responses and different readability scores. For a published factorial experiment, this is a significant methodological gap.

**Evidence**: Searched for `seed`, `random.seed`, `np.random`, `torch.manual_seed` across all `.py` files — zero matches.

### 4. SMOG Index Invalid for Short Texts

**Severity: MODERATE** — Metric reliability concern.

The SMOG formula was designed and validated for texts with **30+ sentences**. Generated responses typically have 2-10 sentences. textstat's implementation applies the formula unconditionally with no minimum-sentence guard.

The formula `1.043 * (30 * polysyllables/sentences)^0.5 + 3.1291` extrapolates short-text polysyllable density to a 30-sentence equivalent. With few sentences, a single polysyllabic word causes large score swings.

**Observed in data**: SMOG values range from 3.13 (floor for 0 polysyllables, due to the +3.1291 constant) to 13.56, with high variance on short texts.

### 5. Multi-Token Logit Bias Multiplicative Effect

**Severity: MODERATE** — Creates uneven vocabulary boosting.

In `_create_logit_bias()` (`llamacpp_base.py:199-208`), every sub-token of a word receives the same `math.log(weight_factor)` bias independently at each decoding step. For multi-token words, this creates a multiplicative joint probability boost:

- 1-token word: 1.5x boost
- 2-token word: 1.5^2 = 2.25x boost
- 3-token word: 1.5^3 = 3.375x boost

This means longer/rarer A1 vocabulary words (which tend to require more sub-tokens) receive disproportionately stronger boosting than short/common A1 words. The experiment specification does not acknowledge this asymmetry.

## Minor Issues

### 6. Negative Readability Scores on Very Short Texts

**Severity: LOW** — Edge case, few rows affected.

The Qwen3 control condition sometimes produces very short responses (e.g., 2 words: `The word ""`). textstat computes FK Grade = -3.01 for such texts. Negative grade levels are mathematically possible in the FK formula but semantically nonsensical.

**Evidence**: Row 2 of `Qwen3_full_experiment_full_1023_0027.csv`.

## Architecture Insights

- The pipeline flow is: raw output -> `_extract_response()` (model-specific) -> `response_formatter.clean_response_for_evaluation()` -> `text_evaluator.evaluate_text_comprehensive()`. The thinking-tag gap exists between steps 1 and 2.
- The context prompting intervention (`_add_simplification_context`) prepends context to the user message, not the system prompt. The system prompt is identical across all conditions. This is a design choice, not an error, but worth documenting.
- Config objects are mutated in-place (`config.prompt_id = prompt_id`), but since `ExperimentResult` captures the value at creation time and the loop is sequential, this does not cause data corruption.

## Recommendations

1. **Thinking contamination**: Already partially addressed by the `/nothink` fix. Additionally, `_extract_response()` should handle incomplete thinking blocks (strip everything from `<think>` to end of string if no `</think>` found). Also consider adding `<think>` tag handling to `ResponseFormatter`.
2. **Empty response scoring**: Use `NaN` instead of `0.0` for failed generations, or add a `generation_successful` column to the output CSV and filter in analysis.
3. **Reproducibility**: Add a `seed` parameter to `ExperimentConfig` and pass it to `Llama()` and/or the generation call.
4. **SMOG validity**: Either add a minimum sentence count check (returning NaN if below threshold), or document the limitation and prefer FK Grade and Gunning Fog for short texts.
5. **Multi-token bias**: Document the multiplicative effect. Consider normalizing the bias by token count: `math.log(weight_factor) / len(tokens)` per token.
6. **Negative scores**: Clamp readability metrics to `max(0, score)` or filter in analysis.
