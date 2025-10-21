# Small Language Model Evaluation Framework

A comprehensive framework for evaluating small language models in English teaching scenarios, with focus on conversation quality and text complexity analysis.

## Project Structure

```
Codigo/
├── venv/                          # Virtual environment (Python 3.11)
├── src/                           # Source code
│   ├── models/                    # Model implementations (legacy)
│   │   ├── small_models/         # TinyStories implementations
│   │   ├── probability_processor.py  # Probability weighting logic
│   │   └── legacy_notebooks/     # Archived model experiments
│   ├── evaluation/               # Evaluation frameworks
│   │   ├── text_complexity/      # Text readability analysis
│   │   │   └── text_evaluator.py # Comprehensive readability metrics
│   │   └── experiment_framework/ # LLM experiment framework
│   │       ├── core/             # Experiment runner, data models
│   │       ├── experiments/      # Pre-configured experiments
│   │       ├── models/           # Model wrappers
│   │       │   ├── llamacpp_base.py            # Base llama.cpp wrapper
│   │       │   ├── qwen2_llamacpp_wrapper.py   # Qwen2 (0.5B) - llama.cpp
│   │       │   ├── qwen3_llamacpp_wrapper.py   # Qwen3 (0.6B) - llama.cpp
│   │       │   ├── phi3_llamacpp_wrapper.py    # Phi3 (3.8B) - llama.cpp
│   │       │   ├── smollm_llamacpp_wrapper.py  # SmolLM (1.7B) - llama.cpp
│   │       │   ├── tinyllama_llamacpp_wrapper.py # TinyLlama (1.1B) - llama.cpp
│   │       │   └── tinystories_wrapper.py      # TinyStories (33M) - Transformers
│   │       └── results/          # Experiment outputs (CSV, PNG, JSON)
│   └── utils/                    # Shared utilities
├── models/                       # Model weights
│   └── gguf/                     # GGUF quantized models
│       ├── Qwen3-0.6B-Q4_0.gguf
│       ├── qwen2.5-0.5b-instruct-q4_0.gguf
│       ├── Phi-3-mini-4k-instruct-q4.gguf
│       ├── SmolLM-1.7B-Instruct.Q4_K_M.gguf
│       └── tinyllama-1.1b-chat-v1.0.Q4_0.gguf
├── data/                         # Data storage
│   └── vocabularies/             # English learning vocabularies (A1 Starters)
├── paper/                        # Research paper materials
│   ├── BRAINSTORMING.md         # Paper planning document
│   ├── EXECUTIVE_SUMMARY.md     # Research summary
│   ├── slm_complexity_control.tex # LaTeX paper draft
│   ├── combined_analysis.py     # Results analysis script
│   ├── figures/                 # Generated visualizations
│   ├── results/                 # Aggregated experiment results
│   ├── SOTA/                    # State-of-the-art literature review
│   └── LREC2026 Author's kit/  # Conference submission template
├── docs/                         # Technical documentation
│   ├── LLAMACPP_MIGRATION_GUIDE.md  # Migration from Transformers
│   ├── text_metrics.md          # Readability metrics reference
│   └── weekly_progress/         # Development logs
├── scripts/                      # Entry point scripts
│   ├── run_experiment.py        # Experiment runner
│   ├── visualize_*.py           # Result visualization scripts
│   └── legacy_tests/            # Integration tests
├── notebooks/                    # Jupyter notebooks
│   └── legacy/                  # Archived explorations
├── requirements.txt              # Python dependencies
├── ExperimentSpecification.md   # Experiment design documentation
└── README.md                     # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Run factorial experiment (all 6 models × 4 configs × 8 prompts = 192 observations)
python scripts/run_experiment.py

# Visualize results
python scripts/visualize_multi_weight_combined.py
python scripts/visualize_weights_comparison.py
```

### 3. Model Integration Tests

```bash
# Test individual model wrappers
python scripts/legacy_tests/test_qwen3_llamacpp_integration.py
python scripts/legacy_tests/test_phi3_llamacpp_integration.py
python scripts/legacy_tests/benchmark_qwen3_llamacpp.py
```

## Experiment Framework

The experiment framework provides comprehensive evaluation of LLM responses:

### Features
- **Text Complexity Analysis**: Flesch-Kincaid, Gunning Fog, SMOG, and additional metrics
- **Readability Metrics**: Reading ease, vocabulary difficulty, sentence complexity
- **Performance Tracking**: Response time, generation success rates
- **Parameter Sweeps**: Systematic testing of model configurations
- **Data Export**: Parquet and CSV files for statistical analysis

### Usage Example

```python
from src.evaluation.experiment_framework.core.experiment_runner import ExperimentRunner
from src.evaluation.experiment_framework.core.data_models import ExperimentConfig

# Create experiment runner
runner = ExperimentRunner()

# Configure experiment
config = ExperimentConfig(
    system_prompt="You are an English teacher for beginners.",
    weighted_words_enabled=True,
    weight_factor=2.0,
    temperature=0.7
)

# Run experiment
results = runner.run_batch_experiment(
    prompts=["What does 'library' mean?", "How do I say hello?"],
    config=config,
    experiment_name="vocabulary_test"
)

# Save results
runner.save_results("my_experiment")
```

## Text Complexity Evaluation

The text evaluator provides comprehensive readability analysis:

```python
from src.evaluation.text_complexity.text_evaluator import TextEvaluator

evaluator = TextEvaluator()

# Analyze text
text = "The library is a place where you can borrow books."
analysis = evaluator.evaluate_text_comprehensive(text)

print(f"Grade level: {analysis['grade_level_indices']['flesch_kincaid_grade']}")
print(f"Reading ease: {analysis['readability_scores']['flesch_reading_ease']}")
```

## Model Implementations

All models use **llama.cpp GGUF backend** for efficient inference:

### Qwen3-0.6B
- **Wrapper**: `src/evaluation/experiment_framework/models/qwen3_llamacpp_wrapper.py`
- **Model File**: `models/gguf/Qwen3-0.6B-Q4_0.gguf`
- **Performance**: 98 tok/s, 491MB memory
- **Template**: ChatML format

### Qwen2-0.5B
- **Wrapper**: `src/evaluation/experiment_framework/models/qwen2_llamacpp_wrapper.py`
- **Model File**: `models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf`
- **Template**: ChatML format

### Phi3-3.8B
- **Wrapper**: `src/evaluation/experiment_framework/models/phi3_llamacpp_wrapper.py`
- **Model File**: `models/gguf/Phi-3-mini-4k-instruct-q4.gguf`
- **Template**: Phi3 chat format

### SmolLM-1.7B
- **Wrapper**: `src/evaluation/experiment_framework/models/smollm_llamacpp_wrapper.py`
- **Model File**: `models/gguf/SmolLM-1.7B-Instruct.Q4_K_M.gguf`
- **Template**: ChatML format

### TinyLlama-1.1B
- **Wrapper**: `src/evaluation/experiment_framework/models/tinyllama_llamacpp_wrapper.py`
- **Model File**: `models/gguf/tinyllama-1.1b-chat-v1.0.Q4_0.gguf`
- **Template**: Zephyr/ChatML format

### TinyStories-33M
- **Wrapper**: `src/evaluation/experiment_framework/models/tinystories_wrapper.py`
- **Model**: `roneneldan/TinyStories-33M` (Hugging Face)
- **Template**: Plain text (no chat template)
- **Backend**: Transformers (native probability weighting support)

**Key Features (llama.cpp models):**
- 4-bit quantization (Q4_0/Q4_K_M) for memory efficiency
- Probability weighting via `logit_bias` for vocabulary control
- On-device inference (no API calls required)
- Metal GPU acceleration on Apple Silicon

**Key Features (TinyStories):**
- Native Transformers probability weighting via `LogitsProcessor`
- Smallest model (33M parameters) for baseline comparison
- Direct vocabulary token manipulation

## Experimental Design

The framework implements a **factorial experiment** for text complexity control:

### Interventions (4 configurations)
1. **Control**: No interventions
2. **Weighting Only**: Probability boosting (A1 vocabulary list, 493 words, 2.0× factor)
3. **Prompting Only**: Context instructions ("You are an English teacher for beginners...")
4. **Both**: Weighting + Prompting combined

### Evaluation Prompts (8 diverse questions)
- Vocabulary definitions ("What does 'library' mean?")
- Grammar explanations ("When do we use 'have' vs 'has'?")
- Conversation scenarios
- Cultural context questions

### Models Evaluated (6 total)
- **llama.cpp**: Phi3 (3.8B), SmolLM (1.7B), TinyLlama (1.1B), Qwen3 (0.6B), Qwen2 (0.5B)
- **Transformers**: TinyStories (33M)

**Total Observations:** 6 models × 4 configs × 8 prompts = **192 observations**

## Data Analysis

Results are exported in multiple formats:
- **Parquet**: Optimized for Google Sheets import
- **CSV**: Human-readable format
- **JSON**: Summary statistics

### Key Metrics Collected
- Response time and generation success
- Text complexity (grade level, readability)
- Vocabulary difficulty and sentence structure
- Word count, syllable count, reading time

## Development

### Adding New Models
1. Create directory under `src/models/`
2. Implement model interface
3. Add to experiment framework

### Adding New Metrics
1. Extend `TextEvaluator` class
2. Update `ExperimentResult` data model
3. Modify export functions

### Adding New Prompts
1. Edit `src/evaluation/experiment_framework/prompts/english_learning_prompts.py`
2. Add to appropriate category
3. Update standard experiment configurations

## Configuration

### Virtual Environment
- **Python**: 3.11
- **Key Dependencies**: llama-cpp-python, textstat, pandas, matplotlib, seaborn
- **Backend**: llama.cpp with Metal GPU acceleration (Apple Silicon)

### Generation Parameters
- **Temperature**: 0.7
- **Top-K**: 50
- **Top-P**: 0.95
- **Max Tokens**: 512
- **Context Window**: 2048 tokens (model-dependent)

## Research Context

This codebase supports thesis research on **text complexity control in Small Language Models** for A1 English learners:

### Research Questions
- Can SLMs be controlled to produce appropriately simple text for beginner learners?
- How effective is probability weighting vs. contextual prompting?
- What are the trade-offs between simplicity and response latency?
- Which models naturally produce beginner-appropriate text?

### Key Contributions
- **Novel approach**: Real-time complexity control via logits manipulation + prompt engineering
- **Comprehensive evaluation**: 6 models (33M-3.8B parameters) across 18 readability metrics
- **Factorial design**: Isolates individual and interaction effects of interventions
- **Deployment focus**: On-device inference for resource-constrained educational settings

### Target Application
Adaptive AI tutors for **550,000 Uruguayan students** through the Ceibal initiative, enabling offline-first, cost-efficient educational technology at scale.