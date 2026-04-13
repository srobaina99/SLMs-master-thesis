# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Small Language Model (SLM) Evaluation Framework** that tests whether inference-time interventions can make SLMs (0.5B-3.8B parameters) produce text at CEFR A1 level (beginner English).

**Research design**: A 2×2 factorial experiment crossing two interventions — probability weighting (logit bias on A1 vocabulary) and contextual prompting — across 4 models and 16 prompts (256 total observations). Success = meeting readability thresholds: Flesch-Kincaid Grade ≤5, Gunning Fog ≤6, SMOG ≤7, Spache ≤4.

**Paper**: There is an LREC 2026 submission in `paper/`. **Do not edit anything under `paper/`** — the paper has been submitted and its contents are frozen.

For the full project layout, see `PROJECT_STRUCTURE.md`.

## Build & Run Commands

The project has a Python 3.11 virtualenv at `Codigo/venv/`. Always activate it before running anything.

```bash
# Environment setup
source Codigo/venv/bin/activate
pip install -r requirements.txt

# Run full factorial experiment (4 models × 4 configs × 16 prompts = 256 observations)
python scripts/run_experiment.py --experiment all --prompts all

# Run single model experiments
python scripts/run_experiment.py --experiment Qwen3 --prompts 5
python scripts/run_experiment.py --experiment Phi3 --prompts all

# Run pytest test suite
pytest

# Integration tests for model wrappers
python scripts/legacy_tests/test_qwen3_llamacpp_integration.py

# Visualize results
python scripts/analysis/visualize_multi_weight_combined.py
python scripts/analysis/visualize_weights_comparison.py
```

## Source Layout (post-refactor)

The source was restructured in the `feature/refactor` branch. Old paths under `src/evaluation/` are now:

```
src/
├── framework/                    # Experiment framework (was src/evaluation/experiment_framework/)
│   ├── core/
│   │   ├── data_models.py           # ExperimentConfig, ExperimentResult, ExperimentDataManager
│   │   └── experiment_runner.py     # ExperimentRunner (simplified public API)
│   ├── experiments/
│   │   ├── experiment_configs.py    # Config factories, prompts, model registry
│   │   ├── factorial_experiment.py  # FactorialExperiment (main experiment loop)
│   │   └── test_multi_weight.py     # Multi-weight experiment test
│   ├── models/
│   │   ├── base_model.py           # BaseModelWrapper (abstract)
│   │   ├── llamacpp_base.py        # LlamaCppBaseWrapper (shared llama.cpp logic)
│   │   ├── qwen3_llamacpp_wrapper.py   # Qwen3 0.6B (ChatML)
│   │   ├── qwen2_llamacpp_wrapper.py   # Qwen2 0.5B (ChatML)
│   │   ├── phi3_llamacpp_wrapper.py    # Phi3 3.8B (custom template, GPU)
│   │   ├── tinyllama_llamacpp_wrapper.py # TinyLlama 1.1B
│   │   └── beam_search_generator.py # Beam search with A1-ratio selection
│   │
│
├── text_complexity/              # Text evaluation (was src/evaluation/text_complexity/)
│   ├── text_evaluator.py           # Readability metrics (textstat + NLTK)
│   ├── response_formatter.py       # Regex-based response cleaning
│   └── obsidian_logger.py          # Logging utility for Obsidian vault
│
└── evaluation/experiment_framework/results/  # Legacy placeholder (mostly empty)
```

### Model Wrapper Hierarchy

```
BaseModelWrapper (abstract)
└── LlamaCppBaseWrapper (src/framework/models/llamacpp_base.py)
    ├── Qwen3LlamaCppWrapper (0.6B, ChatML template)
    ├── Qwen2LlamaCppWrapper (0.5B, ChatML template)
    ├── Phi3LlamaCppWrapper (3.8B, Phi3 template)
    └── TinyLlamaLlamaCppWrapper (1.1B, TinyLlama template)
```

Legacy models (in `legacy/`): TinyStoriesWrapper (33M, Transformers backend), ProbabilityWeightingLogitsProcessor.

### Other Key Directories

- **tests/**: Pytest suite (conftest, imports, data models, configs, evaluator, formatter, pipeline integration)
- **results/**: Main experiment output (CSVs, JSONs, plots) organized by model + aggregate/multi/quick/human_tagged
- **legacy/**: Archived pre-refactor code (old wrappers, utilities, small_models/)
- **scripts/analysis/**: Visualization and utility scripts
- **scripts/legacy_tests/**: Per-model integration tests

## Intervention Configurations

The framework tests 4 intervention combinations:
1. **Control**: No interventions
2. **Weighting Only**: Probability boosting of A1 vocabulary (493 words, 1.5× factor)
3. **Prompting Only**: System prompt for simplification
4. **Both**: Combined weighting + prompting

## Model Weights Location

GGUF quantized models are stored in `models/gguf/`:
- `Qwen3-0.6B-Q4_0.gguf`
- `qwen2.5-0.5b-instruct-q4_0.gguf`
- `Phi-3-mini-4k-instruct-q4.gguf`
- `TinyLlama-1.1b-chat-v1.0.Q4_0.gguf`

## Key Files

- `ExperimentSpecification.md`: Formal experiment design specification
- `docs/LLAMACPP_MIGRATION_GUIDE.md`: Patterns for adding new llama.cpp model wrappers
- `docs/text_metrics.md`: Comprehensive readability metrics reference
- `data/vocabularies/filtered_starters_vocab.txt`: A1 English vocabulary list (493 words)
- `pytest.ini`: Pytest configuration

## Generation Parameters (defaults)

- Temperature: 0.7
- Top-K: 50
- Top-P: 0.95
- Max Tokens: 200
- Context Window: 2048 tokens (4096 for Phi3)

## Output Formats

Results export to `results/`:
- **CSV (full)**: All columns, standard format
- **CSV (specification)**: Reduced columns, European decimal format (`decimal=','`) for Google Sheets
- **JSON**: Summary statistics

## Adding New Models

1. Create wrapper extending `LlamaCppBaseWrapper` in `src/framework/models/`
2. Implement `_format_prompt()` for model-specific chat template
3. Register in `src/framework/experiments/experiment_configs.py`
4. Add GGUF model file to `models/gguf/`

See `docs/LLAMACPP_MIGRATION_GUIDE.md` for detailed patterns and chat template examples.
