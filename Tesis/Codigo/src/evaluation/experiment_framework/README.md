# English Learning Conversation Experiment Framework

A comprehensive framework for evaluating small language models in English teaching scenarios. Integrates with existing Qwen3 implementation and provides automated text complexity evaluation.

## 📁 Folder Structure

```
experiment_framework/
├── core/                          # Core framework components
│   ├── __init__.py
│   ├── data_models.py            # Data structures and Parquet export
│   └── experiment_runner.py      # Main experiment orchestration
├── data/                         # Data storage (auto-created)
├── prompts/                      # Prompt templates and scenarios
│   ├── __init__.py
│   └── english_learning_prompts.py
├── results/                      # Experiment results (auto-created)
├── utils/                        # Utility functions
│   ├── __init__.py
│   └── qwen3_wrapper.py         # Qwen3 model integration
├── demo_experiment.py           # Demo script to test framework
├── __init__.py
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have your Qwen3 model loaded:

```bash
cd /Users/santiago/Documents/Personal/Tesis/Codigo
python -c "from qwen3_mps.qwen3_weighted import *"
```

### 2. Run Demo

```bash
cd experiment_framework
python demo_experiment.py
```

### 3. Run Quick Test

```python
from experiment_framework import ExperimentRunner

# Run a quick test experiment
runner = ExperimentRunner()
runner.run_standard_experiment("quick_test")
results_file = runner.save_results("my_test")
print(f"Results saved to: {results_file}")
```

## 📊 Usage Examples

### Single Experiment

```python
from experiment_framework.core.experiment_runner import ExperimentRunner
from experiment_framework.core.data_models import ExperimentConfig

runner = ExperimentRunner()

config = ExperimentConfig(
    system_prompt="You are an English teacher for beginner students.",
    weighted_words_enabled=True,
    weight_factor=2.0
)

result = runner.run_single_experiment(
    prompt="What does the word 'library' mean?",
    config=config
)

print(f"Response: {result.response}")
print(f"Grade level: {result.flesch_kincaid_grade}")
```

### Batch Experiments

```python
prompts = [
    "What does the word 'library' mean?",
    "How do I introduce myself?",
    "When do I use 'a' vs 'an'?"
]

results = runner.run_batch_experiment(
    prompts=prompts,
    config=config,
    experiment_name="vocabulary_test"
)
```

### Parameter Sweep

```python
base_config = ExperimentConfig(
    system_prompt="You are an English teacher."
)

parameter_variations = {
    'weighted_words_enabled': [False, True],
    'weight_factor': [1.0, 1.5, 2.0],
    'temperature': [0.5, 0.7, 0.9]
}

results = runner.run_parameter_sweep(
    prompts=prompts,
    base_config=base_config,
    parameter_variations=parameter_variations
)
```

### Standard Experiments

```python
# Available: 'basic_teaching', 'grammar_focus', 'conversation_practice', 
#           'error_correction', 'quick_test'

runner.run_standard_experiment("grammar_focus")
runner.save_results("grammar_experiment")
```

## 📈 Data Export

Results are automatically saved in multiple formats:

- **Parquet**: Optimized for Google Sheets import
- **CSV**: Human-readable backup
- **JSON Summary**: Statistical overview

```python
# Save results
results_file = runner.save_results("my_experiment")

# Get summary statistics
summary = runner.get_results_summary()
print(f"Total experiments: {summary['total_experiments']}")
print(f"Average grade level: {summary['flesch_kincaid_grade']['mean']}")
```

## 🔧 Configuration Options

### ExperimentConfig Parameters

- **model_id**: Model identifier (default: "unsloth/Qwen3-0.6B")
- **system_prompt**: System prompt for the model
- **weighted_words_enabled**: Boolean flag for word weighting
- **weight_factor**: Multiplier for word weights (>1 increases probability)
- **enable_thinking**: Enable Qwen3's thinking mode
- **temperature**: Generation temperature (0.1-1.0)
- **top_k**: Top-k sampling parameter
- **top_p**: Top-p (nucleus) sampling parameter

### Collected Metrics

**Text Complexity:**
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- Automated Readability Index
- Coleman-Liau Index
- Dale-Chall Readability Score

**Readability Scores:**
- Flesch Reading Ease
- Linsear Write Formula
- Spache Readability
- McAlpine EFLAW

**Text Statistics:**
- Word count, sentence count, character count
- Syllable count, polysyllable count
- Difficult words count
- Reading time estimates

**Performance:**
- Response time in seconds
- Generation success/failure status

## 🎯 English Learning Prompts

The framework includes 50+ standardized prompts across categories:

- **Vocabulary Questions**: Basic word definitions and usage
- **Grammar Questions**: Grammar rules and usage patterns
- **Conversation Scenarios**: Real-world communication situations
- **Cultural Questions**: Cultural context and social norms
- **Error Correction**: Common ESL mistakes for correction

## 📋 System Prompt Variations

- **basic_teacher**: Simple, encouraging responses
- **detailed_teacher**: Thorough explanations with examples
- **conversational_teacher**: Natural dialogue approach
- **grammar_focused**: Grammar-centric teaching
- **vocabulary_focused**: Vocabulary building emphasis

## 🔍 Analysis and Visualization

After running experiments:

1. **Upload Parquet files to Google Sheets**
2. **Create pivot tables** to analyze parameter effects
3. **Visualize trends** in response complexity and quality
4. **Compare configurations** across different metrics

## 🛠️ Troubleshooting

### Model Not Loaded Error
```
❌ ERROR: Qwen3 model not loaded!
```
**Solution**: Load Qwen3 first:
```bash
python -c "from qwen3_mps.qwen3_weighted import *"
```

### Import Errors
Make sure you're running from the project root directory and all dependencies are installed.

### Memory Issues
For large parameter sweeps, consider:
- Reducing batch sizes
- Running experiments in smaller chunks
- Clearing results periodically with `runner.clear_results()`

## 📝 Example Output

```
Running experiment: vocabulary_test
Prompt: What does the word 'library' mean?...

Generating response...
Response time: 3.45 seconds

Result Summary:
  Response: A library is a place where you can borrow books...
  Response time: 3.45 seconds
  Word count: 42
  Flesch-Kincaid Grade: 6.2
  Reading ease: 78.5

Results saved to: experiment_framework/results/vocabulary_test_20240115_143022.parquet
```

## 🤝 Contributing

To extend the framework:

1. **Add new prompts** in `prompts/english_learning_prompts.py`
2. **Create new metrics** by extending `TextEvaluator`
3. **Add experiment types** in `STANDARD_EXPERIMENT_CONFIGS`
4. **Customize data models** in `core/data_models.py`

---

**Happy Experimenting! 🎉**
