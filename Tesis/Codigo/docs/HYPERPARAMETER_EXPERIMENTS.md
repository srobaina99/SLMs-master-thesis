# Hyperparameter Experiments - Text Complexity Control

## Overview

This document outlines a systematic exploration of three intervention strategies for controlling text complexity in Small Language Models (SLMs). Each intervention group tests a specific hyperparameter to identify optimal configurations for A1-level text generation.

**Objective:** Determine the best combination of intervention type and hyperparameter value for generating pedagogically appropriate text for A1 English learners.

**Model:** Qwen3 (0.6B parameters)
**Test Set:** First 5 prompts from STANDARD_PROMPTS (P1-P5)
**Target Metrics:** Flesch-Kincaid Grade ≤5.0, Gunning Fog ≤6.0, SMOG ≤7.0, Spache ≤4.0

---

## Experiment Groups

### 1. Prompting Strategy Experiment

**Intervention:** Contextual prompting with varying levels of example guidance
**Hyperparameter:** Number of examples (shots)
**Configurations:**

| Configuration       | Description                         | Examples Provided |
| ------------------- | ----------------------------------- | ----------------- |
| **Zero-shot** | Contextual instruction only         | 0                 |
| **One-shot**  | Contextual instruction + 1 example  | 1                 |
| **Few-shot**  | Contextual instruction + 3 examples | 3                 |

**Prompt Structure:**

**Zero-shot:**

```
# Context
Please respond using simple words that a young non-English speaking student can understand. 
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.

[Question]
```

**One-shot:**

```
# Context
Please respond using simple words that a young non-English speaking student can understand. 
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.

# Example
Question: What is a cat?
Answer: A cat is a small animal. It is soft and likes to play and sleep.

[Question]
```

**Few-shot:**

```
# Context
Please respond using simple words that a young non-English speaking student can understand. 
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.

# Examples
Question: What is a cat?
Answer: A cat is a small animal. It is soft and likes to play and sleep.

Question: What does 'happy' mean?
Answer: Happy means you feel good. You smile when you are happy. Happy is a nice feeling.

Question: What is water?
Answer: Water is a drink. We need water every day.

[Question]
```

**Expected Results:**

- Zero-shot: Baseline performance using instruction alone
- One-shot: Improved consistency with concrete example
- Few-shot: Best performance with multiple reference patterns

---

### 2. Beam Search Width Experiment

**Intervention:** Beam search with A1 vocabulary ratio selection
**Hyperparameter:** Number of beams (beam width)
**Configurations:**

| Beam Width     | Candidates Generated | Selection Method |
| -------------- | -------------------- | ---------------- |
| **n=4**  | 4 sequences          | Highest A1 ratio |
| **n=8**  | 8 sequences          | Highest A1 ratio |
| **n=10** | 10 sequences         | Highest A1 ratio |

**Generation Parameters:**

- Temperature: 0.7
- Top-p: 0.95
- Top-k: 50
- Contextual prompting: Enabled (zero-shot)
- Logit bias: Disabled

**A1 Ratio Formula:**

```
A1_ratio = (Count of A1 words × 1.5) / Count of content words
```

**Expected Results:**

- n=4: Baseline beam search performance
- n=8: Improved A1 vocabulary selection (proven: FK Grade 2.14 vs 3.39)
- n=10: Potential further improvement or diminishing returns

**Trade-offs:**

- More beams = Better quality but longer generation time
- Linear time increase: ~76s (n=4) → ~150s (n=8) → ~190s (n=10 estimated)

---

### 3. Logit Bias Weight Experiment

**Intervention:** Vocabulary weighting + contextual prompting ("Both")
**Hyperparameter:** Weight factor for A1 vocabulary tokens
**Configurations:**

| Weight Factor | Logit Bias Applied | Description            |
| ------------- | ------------------ | ---------------------- |
| **1.0** | +1.0 (2.7x prob)   | Light weighting        |
| **1.3** | +1.3 (3.7x prob)   | Moderate weighting     |
| **1.5** | +1.5 (4.5x prob)   | Current baseline       |
| **2.0** | +2.0 (7.4x prob)   | Strong weighting       |
| **2.5** | +2.5 (12.2x prob)  | Very strong weighting  |
| **3.0** | +3.0 (20.1x prob)  | Heavy weighting        |
| **4.0** | +4.0 (54.6x prob)  | Maximum weighting      |

**Logit Bias Calculation:**

> **Note:** The spec originally described `logit_bias = log(weight_factor)`, but the actual implementation at `llamacpp_base.py:203` applies `weight_factor` directly as the logit bias (no `log()` transform). This means the actual probability multipliers are much stronger than a `log()` formula would produce. See `docs/WEIGHTING_MECHANISM.md` for the full analysis.

```python
logit_bias = weight_factor  # applied directly, no log() transform
```

**Generation Parameters:**

- Temperature: 0.7
- Top-p: 0.95
- Top-k: 50
- Contextual prompting: Enabled (zero-shot)
- Beam search: Disabled (greedy decoding)

**Expected Results:**

- 1.0: Baseline with prompting only
- 1.3-1.5: Subtle vocabulary guidance
- 2.0-2.5: Strong A1 vocabulary preference
- 3.0-4.0: Maximum simplification (risk: unnatural language)

**Hypothesis:**

- Optimal weight likely between 1.5-2.5
- Higher weights may sacrifice fluency for simplicity
- Diminishing returns or quality degradation above 3.0

---

## Experimental Design

### Common Configuration

**Model:** Qwen3 (ggml-org/Qwen3-0.6B-GGUF)**System Prompt:** "You are a helpful English teacher for beginner students. Answer with a paragraph only with plain text"**Max Tokens:** 200**Test Prompts:**

- P1: "What does the word 'library' mean?"
- P2: "How do I introduce myself in English?"
- P3: "What is a dog?"
- P4: "Can you explain what 'breakfast' is?"
- P5: "What is the difference between 'big' and 'large'?"

### Evaluation Metrics

**Primary Metrics:**

- Flesch-Kincaid Grade Level (target: ≤5.0)
- Gunning Fog Index (target: ≤6.0)
- SMOG Index (target: ≤7.0)
- Spache Readability (target: ≤4.0)

**Secondary Metrics:**

- Word count (consistency indicator)
- Difficult words count
- Response time (efficiency)

### Results Format

Each experiment will generate:

1. **Specification CSV:** Summary metrics per configuration
2. **Full CSV:** Complete response data with metadata
3. **Summary JSON:** Aggregated statistics by configuration
4. **Visualization:** Boxplots comparing all configurations

---

## Expected Outcomes

### Research Questions

1. **Prompting Strategy:**

   - Does providing examples improve complexity control?
   - What is the optimal number of shots for A1 text generation?
2. **Beam Search:**

   - Does increasing beam width continue to improve A1 ratio selection?
   - What is the point of diminishing returns for beam width?
3. **Logit Bias Weighting:**

   - What is the optimal weight factor for A1 vocabulary?
   - At what point does weighting degrade fluency?

### Success Criteria

**Optimal Configuration:**

- Consistently meets all A1 target thresholds
- Low variance across prompts (predictable performance)
- Reasonable generation time (<5 seconds per prompt)
- Natural, pedagogically appropriate language

---

## Implementation Plan

### Phase 1: Prompting Strategy (Estimated: 30 minutes)

> **Note:** No dedicated prompting experiment script exists yet. This would require a new script.

### Phase 2: Beam Search Width (Estimated: 10 minutes)

```bash
python scripts/run_beam_search_experiment.py
```

### Phase 3: Logit Bias Weighting (Estimated: 15 minutes)

```bash
python scripts/run_experiment.py --experiment multi_weight --weights 1.5,2.0,4.0 --prompts 5
```

The default weight factors are `[1.5, 2.0, 4.0]`.

### Phase 4: Comparative Analysis

```bash
python scripts/analysis/visualize_weights_comparison.py
python scripts/analysis/visualize_beam_search_comparison.py
```

---

## Previous Results (Baseline)

### Beam Search (Width=4 vs Width=8)

| Method   | Beam Width | FK Grade               | Gunning Fog    | SMOG           | Spache         |
| -------- | ---------- | ---------------------- | -------------- | -------------- | -------------- |
| A1 Ratio | 4          | 3.39 ± 2.27           | 5.11 ± 1.84   | 6.78 ± 2.28   | 2.98 ± 1.08   |
| A1 Ratio | 8          | **2.14 ± 0.93** | **~4.0** | **~6.8** | **~2.2** |
| Max Prob | 4          | 4.49 ± 2.08           | 6.08 ± 2.23   | 8.27 ± 1.55   | 3.28 ± 0.69   |
| Max Prob | 8          | 6.15 ± 1.55           | ~7.0           | ~8.7           | ~3.4           |

**Key Finding:** Beam-8 (A1 Ratio) achieved best performance with lowest variance.

### Baseline "Both" Intervention (Weight=1.5)

| Metric      | Mean ± Std  | Range       | Target Met? |
| ----------- | ------------ | ----------- | ----------- |
| FK Grade    | 3.05 ± 2.86 | 0.04 - 6.17 | ✅          |
| Gunning Fog | ~4.5         | 3.6 - 5.6   | ✅          |
| SMOG        | ~6.0         | 3.3 - 9.1   | ✅          |
| Spache      | ~2.5         | 2.1 - 3.7   | ✅          |

**Key Finding:** High variance indicates inconsistent performance despite meeting targets on average.

---

## Files and Scripts

### Experiment Scripts

- `scripts/run_experiment.py` - Main experiment runner (supports `--experiment multi_weight`)
- `scripts/run_beam_search_experiment.py` - Beam width variations

### Visualization Scripts

- `scripts/analysis/visualize_weights_comparison.py` - Weight factor comparison
- `scripts/analysis/visualize_beam_search_comparison.py` - Beam search comparison
- `scripts/analysis/visualize_multi_weight.py` - Per-model weight comparison
- `scripts/analysis/visualize_multi_weight_combined.py` - All models by weight factor

### Results Location

- `results/Qwen3/` - Model-specific results
- `results/multi/` - Multi-weight experiment results

---

## Notes

- All experiments use the same 5 test prompts for direct comparability
- Content word identification uses NLTK POS tagging with fallback heuristic
- Results stored with full metadata for post-hoc analysis
- Beam search experiments already partially completed (width=4, width=8)
- Weight factor 1.5 is current production baseline

---

## Future Extensions

1. **Combined Interventions:** Test beam search + optimal weight factor
2. **Model Comparison:** Replicate experiments on Phi3, Qwen2, TinyLlama
3. **Prompt Engineering:** Test alternative contextual prompts
4. **Dynamic Weighting:** Adaptive weight factors based on prompt complexity
5. **Human Evaluation:** Pedagogical appropriateness assessment beyond metrics
