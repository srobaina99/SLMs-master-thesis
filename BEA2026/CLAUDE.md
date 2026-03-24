# BEA 2026 Shared Task: Vocabulary Difficulty Prediction

## Competition Overview

**Task:** Predict the difficulty (GLMM score, continuous ~[-6, +5]) of English words for learners with L1 = Spanish, German, or Mandarin.
**Metric:** RMSE (primary), Pearson correlation (secondary).
**Tracks:** Closed (provided data only) and Open (external data/LLMs allowed).
**Focus languages:** ES (Spanish) and DE (German) only. We skip CN (Mandarin) because our intuitions about cognates and lexical similarity don't transfer to a non-Latin script.

### Key Dates (AoE)

| Date | Milestone |
|------|-----------|
| Mar 20, 2026 | Test data release |
| Mar 27, 2026 | Submission deadline |
| Apr 3, 2026 | Results announced |
| Apr 24, 2026 | System description paper due |
| Jul 2-3, 2026 | Workshop at ACL 2026 (San Diego) |

### Baseline to Beat (XLM-RoBERTa-base, fine-tuned)

| Track | ES RMSE | DE RMSE | CN RMSE |
|-------|---------|---------|---------|
| Closed | 1.357 | 1.328 | 1.175 |
| Open | 1.206 | 1.149 | 1.021 |

## Data

Located in `vocab-difficulty/` (cloned from https://github.com/britishcouncil/bea2026st).

- **Train:** 6,091 items per L1 (18,273 total)
- **Dev:** 677 items per L1 (2,031 total)
- **Input fields:** `en_target_word`, `en_target_pos`, `en_target_clue`, `L1_source_word`, `L1_context`
- **Target:** `GLMM_score` (lower = harder)

## Build & Run

```bash
# Official baseline pipeline
cd vocab-difficulty
conda env create -f environment.yml
conda activate baseline_env
python run_pipeline.py --evaluate

# Our feature experiments (run from BEA2026/ dir)
source ../Tesis/Codigo/venv/bin/activate
python feature_experiments.py
```

## Project Structure

```
BEA2026/
├── CLAUDE.md                    # This file — onboarding and structure only
├── feature_experiments.py       # Main experiment runner (cached, incremental)
├── word_length_baseline.py      # Initial word-length-only baseline
├── finetune/                    # mDeBERTa fine-tuning pipeline (Colab-ready)
│   ├── train.py                 # Training + prediction + ensemble script
│   └── spec.md                  # Design spec for the fine-tuning approach
├── results/
│   └── results_cache.csv        # Auto-generated experiment metrics (do not edit manually)
├── predictions/                 # Best model prediction CSVs by track/split/L1
│   └── {track}/dev/{es,de}/
└── vocab-difficulty/            # Official shared task repo (British Council)
    ├── data/{train,dev}/{es,de,cn}/
    ├── models/                  # Fine-tuned baselines (download from HF)
    ├── predictions/             # Baseline prediction CSVs
    ├── results/                 # Baseline evaluation summaries
    ├── finetune.py / predict.py / evaluate.py / run_pipeline.py
    └── environment.yml
```

## Experiment Workflow

Experiments are defined in `feature_experiments.py` in the `EXPERIMENTS` dict.
Results are cached in `results/results_cache.csv` — only new experiments run.

- **Add experiment:** add an entry to `EXPERIMENTS` dict and run `python feature_experiments.py`
- **Re-run experiment:** delete its rows from `results/results_cache.csv`
- **Log results:** update `results_log.md` with the new experiment, metrics, and notes

## Submission Format

Place CSV files in: `vocab-difficulty/predictions/{track}/{split}/{L1}/{model_name}_preds.csv`

Required columns: `item_id`, `prediction`

## Reference Files

- `results_log.md` — experiment history, current best, and key findings (source of truth)
- `strategy.md` — approach ideas and things to avoid
- `examples.md` — curated word examples by difficulty (ES)
- `BEA2026_COMPETITION.md` — detailed competition analysis

## Resources

- GitHub: https://github.com/britishcouncil/bea2026st
- Website: https://www.britishcouncil.org/data-science-and-insights/bea2026st
- Google Group: https://groups.google.com/g/bea-2026-shared-task/
- Baseline models on HF: https://huggingface.co/lucyskidmore/models
- Reference paper: Skidmore et al. (2025) -- https://doi.org/10.18653/v1/2025.bea-1.12
