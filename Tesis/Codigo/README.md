# Small Language Model Evaluation Framework

A framework for evaluating small language models in English teaching scenarios, with focus on text complexity control for A1-level learners.

## Project Structure

```
Codigo/
├── src/
│   ├── framework/                  # Experiment framework
│   │   ├── core/
│   │   │   ├── data_models.py         # ExperimentConfig, ExperimentResult, ExperimentDataManager
│   │   │   └── experiment_runner.py   # ExperimentRunner (public API)
│   │   ├── experiments/
│   │   │   ├── experiment_configs.py      # Config factories, prompts, model registry
│   │   │   ├── factorial_experiment.py    # FactorialExperiment (main experiment loop)
│   │   │   └── test_multi_weight.py       # Multi-weight experiment test
│   │   └── models/
│   │       ├── base_model.py             # BaseModelWrapper (abstract)
│   │       ├── llamacpp_base.py          # LlamaCppBaseWrapper (shared llama.cpp logic)
│   │       ├── qwen3_llamacpp_wrapper.py # Qwen3 0.6B (ChatML)
│   │       ├── qwen2_llamacpp_wrapper.py # Qwen2 0.5B (ChatML)
│   │       ├── phi3_llamacpp_wrapper.py  # Phi3 3.8B (Phi3 template, GPU)
│   │       ├── tinyllama_llamacpp_wrapper.py # TinyLlama 1.1B
│   │       └── beam_search_generator.py  # Beam search with A1-ratio selection
│   └── text_complexity/            # Text evaluation
│       ├── text_evaluator.py         # Readability metrics (textstat + NLTK)
│       ├── response_formatter.py     # Regex-based response cleaning
│       └── obsidian_logger.py        # Logging utility
├── scripts/
│   ├── run_experiment.py              # Main experiment runner CLI
│   ├── run_beam_search_experiment.py  # Beam search experiment (Qwen3 only)
│   ├── run_all_models_safe.sh         # Shell script to run all models sequentially
│   ├── analysis/                      # Visualization & utility scripts
│   │   ├── visualize_multi_weight.py
│   │   ├── visualize_multi_weight_combined.py
│   │   ├── visualize_weights_comparison.py
│   │   ├── visualize_beam_search_comparison.py
│   │   └── recover_weight_factors.py
│   └── legacy_tests/                  # Integration tests per model wrapper
├── results/                       # Experiment output (CSVs, JSONs, plots)
│   ├── aggregate/plots/
│   ├── multi/
│   ├── Phi3/, Qwen2/, Qwen3/, TinyLlama/
│   └── human_tagged/
├── legacy/                        # Archived pre-refactor code
├── data/vocabularies/             # A1 vocabulary lists
├── models/gguf/                   # GGUF model weights (not in git)
├── docs/                          # Technical documentation
├── notebooks/legacy/              # Archived notebooks
├── ExperimentSpecification.md     # Experiment design spec
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run full factorial experiment (4 models x 4 configs x 16 prompts = 256 observations)
python scripts/run_experiment.py --experiment all --prompts all

# Run single model experiment
python scripts/run_experiment.py --experiment Qwen3 --prompts 5

# Run multi-weight sweep
python scripts/run_experiment.py --experiment multi_weight --weights 1.5,2.0,4.0

# Visualize results
python scripts/analysis/visualize_multi_weight_combined.py
python scripts/analysis/visualize_weights_comparison.py

# Integration tests
python scripts/legacy_tests/test_qwen3_llamacpp_integration.py
```

## Experiment Framework

### Features
- **Text Complexity Analysis**: Flesch-Kincaid, Gunning Fog, SMOG, Spache metrics
- **Performance Tracking**: Response time, generation success rates
- **Parameter Sweeps**: Multi-weight factor testing across models
- **Data Export**: CSV (standard + European decimal format) and JSON summary statistics

### Usage Example

```python
from src.framework.core.experiment_runner import ExperimentRunner

# Create experiment runner and run factorial experiment
runner = ExperimentRunner()
runner.run_factorial_experiment(num_prompts=5)

# Or run a single model
runner.run_single_model_experiment(model_name="Qwen3", num_prompts=5)
```

### Text Complexity Evaluation

```python
from src.text_complexity.text_evaluator import TextEvaluator

evaluator = TextEvaluator()
text = "The library is a place where you can borrow books."
analysis = evaluator.evaluate_text_comprehensive(text)

print(f"Grade level: {analysis['flesch_kincaid_grade']}")
```

## Model Implementations

All 4 models use the **llama.cpp GGUF backend** for efficient inference:

| Model | Wrapper | GGUF File | Template |
|-------|---------|-----------|----------|
| **Qwen3-0.6B** | `src/framework/models/qwen3_llamacpp_wrapper.py` | `Qwen3-0.6B-Q4_0.gguf` | ChatML |
| **Qwen2-0.5B** | `src/framework/models/qwen2_llamacpp_wrapper.py` | `qwen2.5-0.5b-instruct-q4_0.gguf` | ChatML |
| **Phi3-3.8B** | `src/framework/models/phi3_llamacpp_wrapper.py` | `Phi-3-mini-4k-instruct-q4.gguf` | Phi3 custom |
| **TinyLlama-1.1B** | `src/framework/models/tinyllama_llamacpp_wrapper.py` | `TinyLlama-1.1b-chat-v1.0.Q4_0.gguf` | TinyLlama |

Key features:
- 4-bit quantization (Q4_0/Q4_K_M) for memory efficiency
- Probability weighting via `logit_bias` for vocabulary control
- On-device inference with Metal GPU acceleration on Apple Silicon

## Experimental Design

### Interventions (4 configurations)
1. **Control**: No interventions
2. **Weighting Only**: Probability boosting (A1 vocabulary list, 493 words, 1.5x logit bias)
3. **Prompting Only**: Simplification context instructions
4. **Both**: Weighting + Prompting combined

### Evaluation Prompts (16 diverse questions)
Covering vocabulary definitions, grammar explanations, conversation scenarios, and cultural context.

### Models Evaluated (4 total)
Phi3 (3.8B), TinyLlama (1.1B), Qwen3 (0.6B), Qwen2 (0.5B)

**Total Observations:** 4 models x 4 configs x 16 prompts = **256 observations**

## Data Analysis

Results are exported to `results/`:
- **CSV (full)**: All columns, standard format
- **CSV (specification)**: Reduced columns, European decimal format for Google Sheets
- **JSON**: Summary statistics

### Key Metrics Collected
- Text complexity (Flesch-Kincaid Grade, Gunning Fog, SMOG, Spache)
- Word count, difficult words count
- Response time, generation success

## Generation Parameters (defaults)

- **Temperature**: 0.7
- **Top-K**: 50
- **Top-P**: 0.95
- **Max Tokens**: 200
- **Context Window**: 2048 tokens (4096 for Phi3)

## Development

### Adding New Models
1. Create wrapper extending `LlamaCppBaseWrapper` in `src/framework/models/`
2. Implement `_format_prompt()`, `_get_stop_tokens()`, `_extract_response()`
3. Register in `src/framework/experiments/experiment_configs.py` and `factorial_experiment.py`
4. Add GGUF model file to `models/gguf/`

See `docs/LLAMACPP_MIGRATION_GUIDE.md` for detailed patterns.

### Adding New Prompts
Edit `STANDARD_PROMPTS` in `src/framework/experiments/experiment_configs.py`.

## Research Context

This codebase supports thesis research on **text complexity control in Small Language Models** for A1 English learners, targeting deployment through Uruguay's Ceibal initiative (~550,000 students).
