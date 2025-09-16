# Thesis Code: Small Language Model Evaluation Framework

A comprehensive framework for evaluating small language models in English teaching scenarios, with focus on conversation quality and text complexity analysis.

## 📁 Project Structure

```
thesis_code/
├── .venv/                          # Virtual environment (Python 3.10)
├── src/                           # Source code
│   ├── models/                    # Model implementations
│   │   ├── qwen/                  # Qwen model family
│   │   │   ├── qwen2/            # Qwen2 implementation
│   │   │   ├── qwen3/            # Qwen3 implementation (main)
│   │   │   └── shared/           # Shared utilities (probability_processor)
│   │   └── small_models/         # Other small models
│   │       ├── tinylama/         # TinyLama implementation
│   │       └── tinystories/      # TinyStories implementation
│   ├── evaluation/               # Evaluation frameworks
│   │   ├── text_complexity/      # Text readability analysis
│   │   └── experiment_framework/ # LLM experiment framework
│   └── utils/                    # Shared utilities
├── data/                         # Data storage
│   ├── vocabularies/             # English learning vocabularies
│   ├── model_weights/            # Model weight files (.gguf)
│   └── results/                  # Experiment results
├── notebooks/                    # Jupyter notebooks for exploration
├── scripts/                      # Entry point scripts
│   ├── run_qwen3_chat.py        # Interactive chat interface
│   └── run_experiment.py        # Experiment runner
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Run Interactive Chat

```bash
# Basic English teacher chat
python scripts/run_qwen3_chat.py

# With custom system prompt
python scripts/run_qwen3_chat.py --system "You are a patient English tutor."

# With weighted words
python scripts/run_qwen3_chat.py --words "simple,clear,easy" --factor 1.5
```

### 3. Run Experiments

```bash
# Quick test experiment
python scripts/run_experiment.py --experiment quick_test

# Weighted words comparison
python scripts/run_experiment.py --experiment weighted_comparison

# Full demo
python scripts/run_experiment.py --experiment demo
```

## 🧪 Experiment Framework

The experiment framework provides comprehensive evaluation of LLM responses:

### Features
- **Text Complexity Analysis**: Flesch-Kincaid, Gunning Fog, SMOG, and more
- **Readability Metrics**: Reading ease, vocabulary difficulty, sentence complexity
- **Performance Tracking**: Response time, generation success rates
- **Parameter Sweeps**: Systematic testing of model configurations
- **Data Export**: Parquet files for Google Sheets analysis

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

## 📊 Text Complexity Evaluation

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

## 🤖 Model Implementations

### Qwen3 (Primary Model)
- **Location**: `src/models/qwen/qwen3/`
- **Features**: Weighted word generation, thinking mode, MPS support
- **Usage**: Interactive chat, experiment framework integration

### Qwen2
- **Location**: `src/models/qwen/qwen2/`
- **Features**: Basic chat interface, MPS optimization

### TinyLama & TinyStories
- **Location**: `src/models/small_models/`
- **Features**: Lightweight models for comparison studies

## 📝 English Learning Prompts

The framework includes 50+ standardized prompts across categories:

- **Vocabulary Questions**: Word definitions and usage
- **Grammar Questions**: Grammar rules and patterns
- **Conversation Scenarios**: Real-world communication
- **Cultural Questions**: Social context and norms
- **Error Correction**: Common ESL mistakes

## 📈 Data Analysis

Results are exported in multiple formats:
- **Parquet**: Optimized for Google Sheets import
- **CSV**: Human-readable format
- **JSON**: Summary statistics

### Key Metrics Collected
- Response time and generation success
- Text complexity (grade level, readability)
- Vocabulary difficulty and sentence structure
- Word count, syllable count, reading time

## 🛠️ Development

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

## 🔧 Configuration

### Virtual Environment
- **Python**: 3.10
- **Key Dependencies**: torch, transformers, textstat, pandas
- **GPU Support**: MPS (Apple Silicon), CUDA (optional)

### Model Configuration
- **Default Model**: Qwen3-0.6B (unsloth version)
- **Generation**: Temperature 0.7, top-k 50, top-p 0.95
- **Max Tokens**: 1024

## 📋 Experiment Types

### Standard Experiments
- **quick_test**: 5 diverse prompts, basic evaluation
- **grammar_focus**: Grammar-specific teaching scenarios
- **conversation_practice**: Dialogue-based interactions
- **error_correction**: ESL mistake correction
- **weighted_comparison**: Parameter sweep analysis

### Custom Experiments
Create your own experiments using the `ExperimentRunner` class with custom configurations and prompt sets.

## 🤝 Contributing

1. Follow the established folder structure
2. Add comprehensive docstrings
3. Update relevant README files
4. Test with the experiment framework

## 📚 Research Context

This codebase supports thesis research on:
- Small language model evaluation in educational contexts
- Text complexity analysis for English learning
- Parameter optimization for teaching-focused LLMs
- Comparative analysis of model architectures

---

**Happy Researching! 🎓**