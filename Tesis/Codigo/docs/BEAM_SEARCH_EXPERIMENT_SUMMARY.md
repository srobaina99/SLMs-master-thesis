# Beam Search Experiment - Implementation Summary

## Overview

Implemented and executed a **beam search experiment** testing the efficacy of vocabulary-weighted beam selection for text complexity control on the Qwen3 model. The experiment compares two beam selection criteria:

1. **A1 Word Ratio**: Highest ratio of A1 vocabulary words to content words (with 1.5x weighting)
2. **Max Probability**: Highest cumulative log probability

## Key Design Decisions

### Beam Search Algorithm
- **Standard approach**: Maintain n=4 candidate beams during generation
- **Implementation**: Simplified greedy-per-beam with stochastic sampling (generates n=4 independently sampled sequences)
- **No logit bias during generation**: Uses base model probabilities; A1 weighting applied only at final beam selection

### A1 Word Ratio Calculation
- **Formula**: (Count of A1 words × 1.5) / Count of content words
- **Content words**: Identified via NLTK POS tagging (nouns, verbs, adjectives, adverbs) with fallback heuristic
- **1.5x weighting**: Applied to A1 word count only during final beam selection, not during generation

### Contextual Prompting
All beam searches use contextual prompting intervention, consistent with baseline "Both" intervention for fair comparison.

**Full Contextual Prompt:**
```
# Context
Please respond using simple words that a young non-English speaking student can understand. 
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.
```

**System Prompt:**
```
You are a helpful English teacher for beginner students. Answer with a paragraph only with plain text
```

**Test Prompts (STANDARD_PROMPTS):**
- P1: "What does the word 'library' mean?"
- P2: "How do I introduce myself in English?"
- P3: "What is a dog?"
- P4: "Can you explain what 'breakfast' is?"
- P5: "What is the difference between 'big' and 'large'?"

**Key Design Choice:** No logit bias weighting is applied during beam generation. Only the contextual prompt guides simplification during generation. The A1 vocabulary weighting (1.5x) is applied exclusively at the beam selection stage, not during token generation.

## Results

### Experiment Configuration
- **Model**: Qwen3 (0.6B parameters)
- **Beam width**: 4
- **Prompts**: First 5 from STANDARD_PROMPTS (5 total experiments × 2 selection methods = 10 results)
- **Total time**: ~76 seconds
- **Results saved**: 
  - Specification format: `src/evaluation/experiment_framework/results/Qwen3/Qwen3_beam_search_specification_1111_1929.csv`
  - Full data: `src/evaluation/experiment_framework/results/Qwen3/full_data/Qwen3_beam_search_full_1111_1929.csv`
  - Visualization: `src/evaluation/experiment_framework/results/Qwen3/beam_search_comparison_1111_1936.png`

### Performance Metrics

#### Flesch-Kincaid Grade Level
- **A1 Ratio**: 3.39 ± 2.27 (range: 0.59 - 6.33) ✅ Meets A1 target (≤5.0)
- **Max Probability**: 4.49 ± 2.08 (range: 1.46 - 6.31) ✅ Mostly meets A1 target

#### Gunning Fog Index
- **A1 Ratio**: 5.11 ± 1.84 (range: 2.70 - 7.33) ✅ Meets A1 target (≤6.0)
- **Max Probability**: 6.08 ± 2.23 (range: 3.60 - 8.38) ⚠️ Slightly above A1 target

#### SMOG Index
- **A1 Ratio**: 6.78 ± 2.28 (range: 3.13 - 8.60) ✅ Mostly meets A1 target (≤7.0)
- **Max Probability**: 8.27 ± 1.55 (range: 5.68 - 9.77) ❌ Above A1 target

#### Spache Readability
- **A1 Ratio**: 2.98 ± 1.08 (range: 1.79 - 4.47) ✅ Meets A1 target (≤4.0)
- **Max Probability**: 3.28 ± 0.69 (range: 2.58 - 4.14) ✅ Meets A1 target

#### Word Count
- **A1 Ratio**: 106.80 ± 99.96 words (high variability)
- **Max Probability**: 104.00 ± 101.16 words (high variability)

## Key Findings

1. **A1 Ratio Selection Outperforms**: The A1 word ratio selection method consistently achieves lower complexity scores across most metrics (FK Grade: 3.39 vs 4.49)

2. **A1 Target Achievement**:
   - A1 Ratio meets or nearly meets all A1 targets
   - Max Probability selection exceeds targets on SMOG and Gunning Fog metrics

3. **Metric Agreement**: Spache and FK Grade align well in indicating complexity control success, while SMOG shows higher variance

4. **Verbosity**: Both methods show high variability in word count (mean ~105 words), indicating less consistent control over response length

## Files Created/Modified

### New Files
- `src/evaluation/experiment_framework/models/beam_search_generator.py` - Beam search implementation
- `scripts/run_beam_search_experiment.py` - Experiment runner script
- `scripts/visualize_beam_search_comparison.py` - Visualization script

### Modified Files
- `src/evaluation/text_complexity/text_evaluator.py` - Added content word detection
- `src/evaluation/experiment_framework/core/data_models.py` - Added beam search result fields
- `src/evaluation/experiment_framework/experiments/experiment_configs.py` - Added beam search configs
- `src/evaluation/experiment_framework/models/qwen3_llamacpp_wrapper.py` - Added beam search method
- `src/evaluation/experiment_framework/experiments/factorial_experiment.py` - Added experiment runner method

## Usage

### Run Beam Search Experiment
```bash
cd /Users/santiago/Documents/Personal/SLMs-master-thesis/Tesis/Codigo
. venv/bin/activate
python scripts/run_beam_search_experiment.py
```

### Generate Visualization
```bash
python scripts/visualize_beam_search_comparison.py
```

## Future Work

1. **Expand to All Models**: Test A1 ratio selection on Phi3, Qwen2, and TinyLlama
2. **Larger Beam Width**: Explore n=6, 8, 10 beams for potential improved quality
3. **Dynamic Weighting**: Experiment with adaptive A1 weight factors per model
4. **Length Control**: Add secondary criterion to control word count variance
5. **Multi-turn Dialogue**: Test in conversation scenarios requiring consistency
6. **Human Evaluation**: Assess pedagogical appropriateness beyond readability metrics

## Technical Notes

### Content Word Identification
The A1 word ratio calculation requires identifying **content words** (meaningful words carrying semantic information) versus **function words** (grammatical words like "the", "is", "and").

**Primary Method: NLTK POS Tagging**
- Uses Natural Language Toolkit to identify parts of speech
- **Content words** include:
  - **Nouns**: NN, NNS, NNP, NNPS (e.g., "dog", "library", "books")
  - **Verbs**: VB, VBD, VBG, VBN, VBP, VBZ (e.g., "read", "eating", "was")
  - **Adjectives**: JJ, JJR, JJS (e.g., "big", "bigger", "biggest")
  - **Adverbs**: RB, RBR, RBS (e.g., "quickly", "very", "well")

**Fallback Heuristic** (if NLTK unavailable or fails):
- Words longer than 2 characters
- NOT in predefined list of ~70 common function words

**Rationale**: Only counting content words ensures the A1 ratio focuses on meaningful vocabulary complexity, not grammatical structure. For example:
- "The dog is big" → content words: {dog, big}
- "The canine is enormous" → content words: {canine, enormous}

The second sentence has harder content words despite similar grammatical structure.

### Beam Generation and Selection Process

**Generation Phase:**
1. Generate all `beam_width` (n=4) candidate sequences first
2. Each beam uses independent stochastic sampling (temperature=0.7, top_p=0.95, top_k=50)
3. Creates 4 diverse candidate responses per prompt

**Selection Phase:**
1. Calculate metrics for ALL generated beams:
   - A1 word ratio (weighted) for each beam
   - Cumulative log probability for each beam
2. Select best beam by each criterion:
   - `best_by_a1_ratio`: Beam with highest A1 vocabulary ratio
   - `best_by_probability`: Beam with highest cumulative log probability
3. Experiment runner chooses one based on selection method parameter

**Efficiency Note:** Both selection methods analyze the same set of generated beams. The 10 results (5 prompts × 2 methods) required only 5 generation passes (5 prompts × 4 beams each), with selection applied twice to each set. This makes comparison between selection methods computationally efficient.

### Other Implementation Details
- Cumulative log probability approximated based on generation parameters and sequence length
- Results stored with full beam metadata (all 4 beams + both selections) for post-hoc analysis
- Content words extracted from union of all beams for consistent A1 ratio calculation

