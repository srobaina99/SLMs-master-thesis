# Codigo/ Folder Restructure Implementation Plan

## Overview

Restructure `Codigo/` to remove the over-nested `src/evaluation/experiment_framework/` hierarchy, move result data out of `src/`, group analysis scripts together, and consolidate all legacy code (TinyStories, Transformers-based weighting) in one place.

---

## Current State Analysis

```
Codigo/src/
├── evaluation/
│   ├── experiment_framework/    ← active code buried 3 levels deep
│   │   ├── core/
│   │   ├── experiments/
│   │   ├── models/
│   │   └── results/             ← DATA inside src/ — wrong
│   └── text_complexity/
└── models/                      ← confusingly parallel to framework/models/
    ├── probability_processor.py
    └── small_models/
```

Two bugs exist today that the restructure will fix:
- `experiment_configs.py` and all `models/*.py` wrappers navigate 5 levels up (`Tesis/`) instead of 4 (`Codigo/`) when computing `project_root`, causing an off-by-one inconsistency.
- Model wrappers then do `os.path.join(project_root, "Codigo", "models", "gguf", ...)` to compensate — this double-indirection disappears after the restructure.

### Legacy code (not moved into active src)
- `tinystories_wrapper.py` — TinyStories 33M, Transformers backend, not used in final experiments
- `probability_processor.py` — `LogitsProcessor` for Transformers pipeline, only used by TinyStories
- `src/models/small_models/` — early standalone scripts
- `src/utils/` — early utilities (`run_local_llm.py`, `weighted_stories.py`)

---

## Desired End State

```
Codigo/
├── src/
│   ├── framework/               # was: src/evaluation/experiment_framework/
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── experiments/
│   │   ├── models/              # only llama.cpp wrappers (no TinyStories)
│   │   └── demo_factorial_experiment.py
│   └── text_complexity/         # was: src/evaluation/text_complexity/
│       ├── __init__.py
│       ├── text_evaluator.py
│       ├── response_formatter.py
│       └── obsidian_logger.py
├── scripts/
│   ├── run_experiment.py
│   ├── run_beam_search_experiment.py
│   ├── run_all_models_safe.sh
│   └── analysis/                # was: loose visualize_*.py + recover_weight_factors.py
│       ├── visualize_multi_weight.py
│       ├── visualize_multi_weight_combined.py
│       ├── visualize_weights_comparison.py
│       ├── visualize_beam_search_comparison.py
│       └── recover_weight_factors.py
├── results/                     # was: src/evaluation/experiment_framework/results/
│   ├── visualize_results.py
│   ├── aggregate/
│   ├── multi/
│   ├── quick/
│   ├── Phi3/
│   ├── Qwen2/
│   ├── Qwen3/
│   └── TinyLlama/
├── legacy/                      # new: all superseded code
│   ├── probability_processor.py
│   ├── tinystories_wrapper.py
│   ├── small_models/
│   ├── run_local_llm.py
│   └── weighted_stories.py
├── data/vocabularies/
├── models/gguf/
├── notebooks/legacy/
└── docs/
```

### Verification
- `python scripts/run_experiment.py --experiment Qwen3 --prompts 1` completes without ImportError
- `python scripts/run_beam_search_experiment.py` resolves paths correctly
- All 4 visualization scripts in `scripts/analysis/` find their default input CSVs
- `results/visualize_results.py` locates model subdirectories correctly

---

## What We're NOT Doing

- Not changing any logic, generation parameters, or experiment design
- Not touching `scripts/legacy_tests/` (already broken/legacy, leave as-is)
- Not changing `data/`, `models/gguf/`, `docs/`, or `notebooks/`
- Not installing a proper Python package (`setup.py`/`pyproject.toml`) — keeping the `sys.path` append pattern
- Not fixing `obsidian_logger.py`'s bare `from text_evaluator import` or hardcoded vault path (legacy-ish file)
- Not updating `CLAUDE.md` or `README.md` paths (addressed separately)

---

## Phase 1: Move Files with git mv

All moves preserve git history.

### 1a. Active source code

```bash
# Create new package directories
mkdir -p src/framework
mkdir -p src/text_complexity
mkdir -p scripts/analysis

# Move experiment framework
git mv src/evaluation/experiment_framework/__init__.py      src/framework/__init__.py
git mv src/evaluation/experiment_framework/core             src/framework/core
git mv src/evaluation/experiment_framework/experiments      src/framework/experiments
git mv src/evaluation/experiment_framework/models           src/framework/models
git mv src/evaluation/experiment_framework/demo_factorial_experiment.py  src/framework/demo_factorial_experiment.py

# Move text complexity
git mv src/evaluation/text_complexity/__init__.py           src/text_complexity/__init__.py
git mv src/evaluation/text_complexity/text_evaluator.py     src/text_complexity/text_evaluator.py
git mv src/evaluation/text_complexity/response_formatter.py src/text_complexity/response_formatter.py
git mv src/evaluation/text_complexity/obsidian_logger.py    src/text_complexity/obsidian_logger.py

# Move results out of src/
git mv src/evaluation/experiment_framework/results          results

# Move analysis scripts to subdirectory
git mv scripts/visualize_multi_weight.py            scripts/analysis/visualize_multi_weight.py
git mv scripts/visualize_multi_weight_combined.py   scripts/analysis/visualize_multi_weight_combined.py
git mv scripts/visualize_weights_comparison.py      scripts/analysis/visualize_weights_comparison.py
git mv scripts/visualize_beam_search_comparison.py  scripts/analysis/visualize_beam_search_comparison.py
git mv scripts/recover_weight_factors.py            scripts/analysis/recover_weight_factors.py
```

### 1b. Legacy code consolidation

```bash
mkdir -p legacy

# TinyStories wrapper and its Transformers-based weighting processor
git mv src/evaluation/experiment_framework/models/tinystories_wrapper.py  legacy/tinystories_wrapper.py
git mv src/models/probability_processor.py                                 legacy/probability_processor.py

# Early standalone scripts
git mv src/models/small_models  legacy/small_models
git mv src/utils/run_local_llm.py   legacy/run_local_llm.py
git mv src/utils/weighted_stories.py legacy/weighted_stories.py
```

### 1c. Clean up now-empty directories

```bash
# Remove empty intermediate directories
rm -rf src/evaluation   # now empty after moves
rm -rf src/models       # now empty after moves
rm -rf src/utils        # now empty after moves
```

### Success Criteria:
#### Automated:
- [ ] `git status` shows all moves as renames (not delete+add)
- [ ] `ls src/` shows only `__init__.py`, `framework/`, `text_complexity/`
- [ ] `ls results/` shows `visualize_results.py`, `aggregate/`, `multi/`, `Phi3/`, `Qwen2/`, `Qwen3/`, `TinyLlama/`
- [ ] `ls scripts/` shows `run_experiment.py`, `run_beam_search_experiment.py`, `run_all_models_safe.sh`, `analysis/`
- [ ] `ls legacy/` shows 5 items

---

## Phase 2: Update sys.path in All Active Files

Every file that computes `project_root` via `__file__` traversal needs updating. The new uniform rule: **all active source files navigate to `Codigo/`**, and the level count is now consistent across the whole codebase.

New depth from each location to `Codigo/`:
- `src/framework/core/` → 3 levels up
- `src/framework/experiments/` → 3 levels up
- `src/framework/models/` → 3 levels up
- `src/framework/` (demo) → 2 levels up
- `results/` → 1 level up
- `scripts/` → 1 level up
- `scripts/analysis/` → 2 levels up

### Files to update:

**`src/framework/core/experiment_runner.py`**
```python
# OLD (4 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
# NEW (3 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
```
Also fix default results_dir computation (see Phase 3).

**`src/framework/experiments/factorial_experiment.py`**
```python
# OLD (4 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
# NEW (3 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
```

**`src/framework/experiments/experiment_configs.py`**
```python
# OLD (5 levels up → Tesis/ — was a bug)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
# NEW (3 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
```

**`src/framework/experiments/test_multi_weight.py`**
```python
# OLD (4 levels up → Codigo/)  →  NEW (3 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
```

**`src/framework/demo_factorial_experiment.py`**
```python
# OLD (4 levels up → Codigo/)  →  NEW (2 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(current_dir))
```

**`src/framework/models/base_model.py`**
```python
# OLD (4 levels up → Codigo/)  →  NEW (3 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# vocab path stays the same:
# os.path.join(project_root, "data", "vocabularies", "filtered_starters_vocab.txt")
```

**`src/framework/models/llamacpp_base.py`**
```python
# OLD (5 levels up → Tesis/ — was a bug)  →  NEW (3 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
```

**`src/framework/models/qwen3_llamacpp_wrapper.py`**
**`src/framework/models/qwen2_llamacpp_wrapper.py`**
**`src/framework/models/phi3_llamacpp_wrapper.py`**
**`src/framework/models/tinyllama_llamacpp_wrapper.py`**
```python
# OLD (5 levels up → Tesis/ — was a bug)  →  NEW (3 levels up → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
```

**`scripts/run_experiment.py`** — already correct (2 levels up → Codigo/), no change.

**`scripts/run_beam_search_experiment.py`** — already correct (2 levels up → Codigo/), no change.

**`scripts/visualize_beam_search_comparison.py`** (now in `scripts/analysis/`)
```python
# OLD (2 levels up from scripts/ → Codigo/)
project_root = os.path.dirname(os.path.dirname(current_dir))
# NEW (3 levels up from scripts/analysis/ → Codigo/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
```

**`scripts/analysis/visualize_multi_weight.py`** (no sys.path manipulation)
**`scripts/analysis/visualize_multi_weight_combined.py`** (no sys.path manipulation)
**`scripts/analysis/visualize_weights_comparison.py`** (no sys.path manipulation)
**`scripts/analysis/recover_weight_factors.py`** (no sys.path manipulation)
→ Only the `Path(__file__)` anchor changes (see Phase 3).

### Success Criteria:
#### Automated:
- [ ] `python -c "import src.framework.core.experiment_runner"` succeeds (run from Codigo/)
- [ ] `python -c "import src.framework.models.qwen3_llamacpp_wrapper"` succeeds

---

## Phase 3: Update All Absolute Import Strings and Hardcoded Paths

### 3a. Import strings (`src.evaluation.X` → `src.framework.X` or `src.text_complexity.X`)

Files with `from src.evaluation.experiment_framework.X import Y`:
- `src/framework/core/experiment_runner.py` — 3 imports
- `src/framework/experiments/factorial_experiment.py` — 4 imports
- `src/framework/experiments/experiment_configs.py` — 1 import
- `src/framework/experiments/test_multi_weight.py` — 1 import
- `src/framework/demo_factorial_experiment.py` — 3 imports
- `src/framework/models/base_model.py` — 1 import
- `src/framework/models/llamacpp_base.py` — 2 imports (`data_models` + `text_evaluator`)
- `src/framework/models/qwen3_llamacpp_wrapper.py` — 1 import
- `scripts/run_experiment.py` — 3 imports
- `scripts/run_beam_search_experiment.py` — 2 imports
- `scripts/analysis/visualize_beam_search_comparison.py` — 0 imports (no src imports)

Transformation rules:
```
from src.evaluation.experiment_framework.  →  from src.framework.
from src.evaluation.text_complexity.       →  from src.text_complexity.
```

### 3b. Remove TinyStoriesWrapper from `src/framework/models/__init__.py`

```python
# REMOVE this line:
from .tinystories_wrapper import TinyStoriesWrapper
```

### 3c. Hardcoded results directory strings

**`src/framework/experiments/factorial_experiment.py` line 40**
```python
# OLD
results_dir: str = "src/evaluation/experiment_framework/results"
# NEW
results_dir: str = "results"
```

**`src/framework/core/experiment_runner.py`** — default results_dir is currently computed as:
```python
framework_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(framework_dir, "results")
```
After move, `__file__` is `src/framework/core/experiment_runner.py`. Going up 2 levels gives `src/framework/` (not `results/`). Fix:
```python
# NEW: go up 3 levels to Codigo/, then into results/
codigo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(codigo_dir, "results")
```
(The `visualize_results.py` dynamic import `Path(self.results_dir) / "visualize_results.py"` continues to work unchanged.)

**`scripts/run_beam_search_experiment.py` line 33**
```python
# OLD
experiment = FactorialExperiment(results_dir="src/evaluation/experiment_framework/results")
# NEW
experiment = FactorialExperiment(results_dir="results")
```

**`scripts/analysis/visualize_beam_search_comparison.py` lines 92, 206**
```python
# OLD
"src/evaluation/experiment_framework/results"
"src/evaluation/experiment_framework/results/Qwen3"
# NEW
"results"
"results/Qwen3"
```

### 3d. Path anchor updates in analysis scripts (moved to `scripts/analysis/`)

All four scripts use `Path(__file__).parent.parent` to reach `Codigo/` from `scripts/`. Now in `scripts/analysis/` they must use `Path(__file__).parent.parent.parent`:

**`scripts/analysis/visualize_multi_weight.py` lines 117–118**
**`scripts/analysis/visualize_multi_weight_combined.py` lines 111–112**
**`scripts/analysis/visualize_weights_comparison.py` lines 122–123**
**`scripts/analysis/recover_weight_factors.py` lines 62–65**
```python
# OLD
Path(__file__).parent.parent / "src/evaluation/experiment_framework/results/multi"
# NEW
Path(__file__).parent.parent.parent / "results/multi"
```

### 3e. GGUF model paths in wrappers

Currently these navigate to `Tesis/` and then add `"Codigo"` back. After fixing `project_root` to point to `Codigo/`, remove the `"Codigo"` segment:

**All 4 wrappers (qwen3, qwen2, phi3, tinyllama)**:
```python
# OLD (project_root was Tesis/, needed "Codigo" in path)
model_path = os.path.join(project_root, "Codigo", "models", "gguf", "<filename>")
# NEW (project_root is now Codigo/)
model_path = os.path.join(project_root, "models", "gguf", "<filename>")
```

### Success Criteria:
#### Automated:
- [ ] `grep -r "src.evaluation" src/framework/` returns nothing
- [ ] `grep -r "src.evaluation" scripts/` returns nothing
- [ ] `grep -r "experiment_framework/results" src/` returns nothing
- [ ] `grep -r "experiment_framework/results" scripts/` returns nothing
- [ ] `grep -r '"Codigo"' src/framework/models/` returns nothing

---

## Phase 4: Update `__init__.py` Package Exports

**`src/framework/__init__.py`** (was `experiment_framework/__init__.py`):
```python
# Update all absolute import paths from src.evaluation.experiment_framework.X
# to src.framework.X
from .core.experiment_runner import ExperimentRunner, ...
from .core.data_models import ExperimentResult, ...
from .experiments.factorial_experiment import FactorialExperiment
from .experiments.experiment_configs import STANDARD_PROMPTS, create_factorial_configs
# (these are relative imports — no change needed)
```
The relative imports (`.core.X`, `.experiments.X`) don't need updating since they're relative. Just verify the file is correct.

**`src/framework/models/__init__.py`**: Remove `TinyStoriesWrapper` import (Phase 3b above).

### Success Criteria:
#### Automated:
- [ ] `python -c "from src.framework import ExperimentRunner"` succeeds
- [ ] `python -c "from src.framework.models import Qwen3LlamaCppWrapper"` succeeds

---

## Phase 5: Smoke Test

Run the actual experiment to confirm end-to-end functionality.

### Success Criteria:
#### Automated:
- [ ] `python scripts/run_experiment.py --experiment Qwen3 --prompts 1` completes, writes a CSV to `results/Qwen3/`
- [ ] `python scripts/run_beam_search_experiment.py` resolves paths without error (can abort early)
- [ ] `python scripts/analysis/visualize_multi_weight.py` loads the default CSV without FileNotFoundError

#### Manual:
- [ ] New CSV file appears in `results/Qwen3/` (not in old `src/evaluation/...` path)
- [ ] No Python traceback on any of the above commands

---

## References

- Dependency analysis: full import graph and sys.path table documented during planning session (2026-02-20)
- `Codigo/CLAUDE.md`: architecture overview (paths will need updating after restructure)
- `PROJECT_STRUCTURE.md`: top-level map (will need updating after restructure)
