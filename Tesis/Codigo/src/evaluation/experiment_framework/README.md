# Factorial Experiment Framework

A clean, focused framework for running factorial experiments with small language models, implementing the 4×4×N experimental design from `ExperimentSpecification.md`.

## Purpose

Evaluate the effectiveness of two interventions on small language model outputs:
1. **Probability Weighting**: Boost vocabulary from beginner English word lists
2. **Context Prompting**: Add simplification instructions to prompts

## Structure

```
experiment_framework/
├── core/                          # Core framework components
│   ├── data_models.py            # Data structures and CSV export
│   └── experiment_runner.py      # Main experiment interface
├── experiments/                   # Experiment logic
│   ├── factorial_experiment.py   # 4×4×N factorial design
│   └── experiment_configs.py     # Standard configurations and prompts
├── models/                        # Model wrappers
│   ├── base_model.py            # Abstract base class
│   ├── qwen2_wrapper.py         # Qwen2 integration
│   ├── qwen3_wrapper.py         # Qwen3 integration
│   ├── tinyllama_wrapper.py     # TinyLlama integration
│   └── tinystories_wrapper.py   # TinyStories integration
├── demo_factorial_experiment.py  # Interactive demo
└── README.md                     # This file
```

## Quick Start

### Basic Usage

```python
from experiment_framework import ExperimentRunner

# Initialize runner
runner = ExperimentRunner()

# Run complete factorial experiment (5 models × 4 configs × 10 prompts = 200 experiments)
results_file = runner.run_factorial_experiment()

# Run experiment for single model only
results_file = runner.run_single_model_experiment("Qwen3")

# Check which models are loaded
status = runner.get_model_status()
print(status)
```

### Quick Tests

```python
from experiment_framework import run_quick_factorial_test, run_single_model_test

# Quick test with 3 prompts (48 total experiments)
results_file = run_quick_factorial_test()

# Test single model with 2 prompts (8 experiments)
results_file = run_single_model_test("TinyStories")
```

### Interactive Demo

```bash
cd experiment_framework
python demo_factorial_experiment.py
```

## Experimental Design

### Models (4)
- **Qwen2**: Qwen2.5-0.5B-Instruct
- **Qwen3**: Qwen3-0.6B  
- **TinyLlama**: TinyLlama-1.1B-Chat-v1.0
- **TinyStories**: TinyStories-33M

### Interventions (4 combinations)
1. **Control**: No interventions
2. **Weighting Only**: Boost probability of simple vocabulary
3. **Prompting Only**: Add simplification context
4. **Both**: Weighting + Prompting

### Standard Prompts (10)
- Basic English learning questions
- Vocabulary explanations
- Grammar usage
- Conversational scenarios

## Output Format

Results are saved in the exact format specified in `ExperimentSpecification.md`:

| model | config_weighting | config_prompting | prompt_id | answer | time_spent | flesch_kincaid_grade | ... |
|-------|------------------|------------------|-----------|--------|------------|---------------------|-----|
| Qwen2 | True | False | P1 | "Hello there!" | 2.3 | 4.2 | ... |
| Qwen3 | False | True | P1 | "Hi friend!" | 1.8 | 3.8 | ... |

## Configuration

All configurations are automatically generated:

```python
from experiment_framework import create_factorial_configs, STANDARD_PROMPTS

# Get all 20 configurations (5 models × 4 intervention combinations)
configs = create_factorial_configs()

# Get standard prompts
prompts = STANDARD_PROMPTS
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- All dependencies from `requirements.txt`

## Model Status

Check which models are available:

```python
runner = ExperimentRunner()
status = runner.get_model_status()

for model_name, info in status.items():
    print(f"{model_name}: {'✅ LOADED' if info['loaded'] else '❌ NOT LOADED'}")
```

## Notes

- Models are loaded on-demand when first used
- Results are automatically saved in multiple formats (CSV, Parquet, JSON)
- All text complexity metrics are calculated using the `textstat` library
- Vocabulary weighting uses words from `data/vocabularies/filtered_starters_vocab.txt`