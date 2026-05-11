# Project Structure: SLM Master Thesis

**Date**: 2026-03-04 (updated)
**Git Commit**: 2fe39cb
**Branch**: feature/refactor

## Overview

This is a master thesis project investigating **inference-time complexity control for Small Language Models (SLMs) in educational applications**. The goal is to make SLMs (0.5B-3.8B parameters) generate A1-level English text for young learners, targeting deployment through Uruguay's Ceibal initiative (~550,000 students).

The project contains: experimental code, a paper submission (LREC 2026), bibliography notes, and thesis notes.

---

## Top-Level Directory Map

```
Tesis/
├── Codigo/                          # CODE: Experiment framework & analysis
├── paper/                           # PAPER: LaTeX submission & supporting docs
├── Bibliografía/                    # LITERATURE: Bibliography notes & PDFs
├── reference_papers/                # REFERENCE: LaTeX source of cited papers
├── thoughts/                        # NOTES: Improvement ideas, handoffs, plans
├── CLAUDE.md                        # AI assistant instructions for this project
├── .obsidian/                       # CONFIG: Obsidian vault settings
│
├── Primeros pasos tesis.md          # Chronological thesis progress log (weeks 1-7+)
├── Notas tiny stories.md            # Notes on TinyStories paper
├── SLM Agents.md                    # Notes on SLMs in agentic settings
├── PROJECT_STRUCTURE.md             # This file
├── starters.txt                     # Cambridge A1 Starters word list (raw)
├── Ruiz-Noel-Tesis-final.pdf        # Reference thesis PDF
├── 149680-yle-movers-word-list.pdf  # Cambridge Movers word list PDF
└── starters-word-list-picture-book.pdf  # Cambridge Starters picture book
```

---

## 1. `Codigo/` - Experiment Framework

This is the core codebase: a Python framework that evaluates 4 SLMs on their ability to generate A1-level English text using a 2x2 factorial design (probability weighting x contextual prompting).

> **Note**: The source layout was restructured in the `feature/refactor` branch. The old `src/evaluation/experiment_framework/` and `src/evaluation/text_complexity/` paths are now `src/framework/` and `src/text_complexity/` respectively. Old code lives in `legacy/`.

```
Codigo/
├── scripts/                    # CLI entry points
│   ├── run_experiment.py           # Main experiment runner (all models, single, multi-weight)
│   ├── run_beam_search_experiment.py  # Beam search experiment (Qwen3 only)
│   ├── run_all_models_safe.sh      # Shell script to run all models sequentially
│   ├── analysis/                   # Visualization & utility scripts
│   │   ├── visualize_multi_weight.py           # Plot: per-model weight comparison
│   │   ├── visualize_multi_weight_combined.py  # Plot: all models by weight factor
│   │   ├── visualize_weights_comparison.py     # Plot: side-by-side weight factors
│   │   ├── visualize_beam_search_comparison.py # Plot: beam vs baseline
│   │   └── recover_weight_factors.py           # Utility: backfill weight_factor column
│   └── legacy_tests/              # Integration tests per model wrapper
│       ├── test_qwen2_llamacpp_integration.py
│       ├── test_qwen3_llamacpp_integration.py
│       ├── test_smollm_llamacpp_integration.py
│       ├── test_tinyllama_llamacpp_integration.py
│       └── benchmark_qwen3_llamacpp.py
│
├── src/
│   ├── framework/                  # Main experiment framework (refactored)
│   │   ├── core/
│   │   │   ├── data_models.py         # ExperimentConfig, ExperimentResult, ExperimentDataManager
│   │   │   └── experiment_runner.py   # ExperimentRunner (simplified public API)
│   │   ├── experiments/
│   │   │   ├── experiment_configs.py      # Config factories, prompts, model registry
│   │   │   ├── factorial_experiment.py    # FactorialExperiment (main experiment loop)
│   │   │   └── test_multi_weight.py       # Multi-weight experiment test
│   │   ├── models/
│   │   │   ├── base_model.py             # BaseModelWrapper (abstract)
│   │   │   ├── llamacpp_base.py          # LlamaCppBaseWrapper (shared llama.cpp logic)
│   │   │   ├── qwen3_llamacpp_wrapper.py     # Qwen3 0.6B (ChatML)
│   │   │   ├── qwen2_llamacpp_wrapper.py     # Qwen2 0.5B (ChatML)
│   │   │   ├── phi3_llamacpp_wrapper.py      # Phi3 3.8B (custom template, GPU)
│   │   │   ├── tinyllama_llamacpp_wrapper.py # TinyLlama 1.1B
│   │   │   └── beam_search_generator.py      # Beam search with A1-ratio selection
│   │
│   │
│   ├── text_complexity/            # Text evaluation (refactored from src/evaluation/)
│   │   ├── text_evaluator.py         # Readability metrics (textstat + NLTK)
│   │   ├── response_formatter.py     # Regex-based response cleaning
│   │   └── obsidian_logger.py        # Logging utility for Obsidian vault
│   │
│   └── evaluation/experiment_framework/results/  # Legacy results placeholder (mostly empty)
│
├── tests/                      # Pytest test suite
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_imports.py             # Import validation
│   ├── test_data_models.py         # Data model tests
│   ├── test_experiment_configs.py  # Config tests
│   ├── test_text_evaluator.py      # Text evaluator tests
│   ├── test_response_formatter.py  # Response formatter tests
│   └── test_pipeline_integration.py  # Full pipeline integration tests
│
├── results/                    # Experiment output (main results directory)
│   ├── visualize_results.py        # Boxplot visualization
│   ├── human_evaluation.csv        # Full human evaluation dataset
│   ├── 60_human_eval.csv           # Top 60 human evaluations
│   ├── config_rankings.csv         # Configuration rankings
│   ├── data_sources_map.csv        # Data source mapping
│   ├── aggregate/plots/            # Combined boxplots across all models
│   ├── multi/                      # Multi-weight experiment results
│   │   ├── full_data/              # Raw CSV
│   │   └── plots/                  # Per-model and combined weight plots
│   ├── quick/                      # Quick smoke-test results
│   ├── human_tagged/               # Human-tagged evaluation data
│   │   ├── human_eval_by_config.csv
│   │   └── Evaluación humana - sorted top 60.csv
│   ├── Phi3/
│   │   ├── full_data/
│   │   └── plots/
│   ├── Qwen2/
│   │   ├── full_data/
│   │   └── plots/
│   ├── Qwen3/                      # Includes beam search CSVs and plots
│   │   ├── full_data/
│   │   └── plots/
│   └── TinyLlama/
│       ├── full_data/
│       └── plots/
│
├── legacy/                     # Archived pre-refactor code
│   ├── run_local_llm.py            # Early local LLM utility
│   ├── tinystories_wrapper.py      # TinyStories 33M wrapper (Transformers)
│   ├── weighted_stories.py         # Early weighted generation utility
│   ├── probability_processor.py    # ProbabilityWeightingLogitsProcessor
│   └── small_models/               # Early standalone model scripts
│       ├── tinyllama/
│       └── tinystories/
│
├── data/vocabularies/
│   ├── starters_vocab.txt             # Raw Cambridge A1 vocab with POS tags
│   └── filtered_starters_vocab.txt    # 493 cleaned A1 words (used by all models)
│
├── models/gguf/                    # GGUF model binaries (not in git, ~4.6 GB)
│   ├── Qwen3-0.6B-Q4_0.gguf
│   ├── qwen2.5-0.5b-instruct-q4_0.gguf
│   ├── Phi-3-mini-4k-instruct-q4.gguf
│   └── TinyLlama-1.1b-chat-v1.0.Q4_0.gguf
│
├── notebooks/legacy/               # Old exploratory notebooks
│   ├── biased_stories.ipynb
│   └── chating_w_models.ipynb
│
├── docs/                           # Technical documentation (see below)
│
├── .cursor/rules/                  # Cursor AI configuration
├── venv/                           # Python 3.11 virtual environment (not in git)
│
├── README.md                       # Main project documentation
├── ExperimentSpecification.md      # Formal experiment design spec
├── pytest.ini                      # Pytest configuration
└── requirements.txt                # Python dependencies
```

### How experiments work

1. `scripts/run_experiment.py` parses CLI args and delegates to `ExperimentRunner`
2. `ExperimentRunner` wraps `FactorialExperiment` which loops: **model -> prompt -> intervention config**
3. Each model wrapper (extending `LlamaCppBaseWrapper`) loads a GGUF model, formats prompts with model-specific templates, optionally applies logit bias (weighting) and/or simplification context (prompting)
4. Generated responses are cleaned by `ResponseFormatter` then scored by `TextEvaluator` (FK Grade, Gunning Fog, SMOG, Spache)
5. Results are saved as CSVs (European decimal format for specification, standard for full data) and JSON summaries

### Key docs inside `Codigo/docs/`

| File | What it documents |
|------|-------------------|
| `text_metrics.md` | Why these 4 metrics were chosen (from 18 candidates) |
| `LLAMACPP_MIGRATION_GUIDE.md` | How to add new llama.cpp model wrappers |
| `GPU_OPTIMIZATION_GUIDE.md` | GPU vs CPU benchmarks per model on M2 Mac |
| `SLM_GUIDE.md` | Model specifications and integration checklist |
| `SLM_BENCHMARK_COMPARISON.md` | Quality benchmarks and tier rankings |
| `BEAM_SEARCH_EXPERIMENT_SUMMARY.md` | Beam search design and results |
| `HYPERPARAMETER_EXPERIMENTS.md` | Planned hyperparameter exploration |
| `WEIGHTING_MECHANISM.md` | How probability weighting works end-to-end |
| `LEGACY_CLEANUP_SUMMARY.md` | Migration from Transformers to llama.cpp |
| `weekly_progress/` | Weekly progress reports |
| `meeting_transcriptions/` | Supervisor meeting notes |

---

## 2. `paper/` - LREC 2026 Paper Submission

```
paper/
├── simple-SLM-FINAL-SUBMISSION.tex  # FINAL SUBMISSION (real author names, median statistics)
├── references.bib                   # Bibliography references
├── plots/                           # Figures referenced by the paper
│   └── aggregate_all_models_1023_2345.png  # Main boxplot figure (Fig. 1 in paper)
├── results/
│   ├── combined_summary_MEDIAN.json  # Median-based aggregate statistics (source of truth)
│   ├── human_auto_correlation_response_level.csv   # Human-auto metric correlation
│   └── human_auto_rank_agreement_config_level.csv  # Human-auto rank agreement
├── SOTA/
│   ├── sota.md                    # Broad literature review (5 categories)
│   └── llm_output_control_review.md  # Focused review on output control (5 papers)
├── LREC2026 Author's kit/        # Conference template files
│
├── TODO.md                       # Paper task list
├── WORKSHOP_INFO.MD              # Workshop submission information
├── LREC_reviews.md               # Reviewer feedback from LREC submission
├── verify_paper_values.py        # Verifier for paper vs data consistency
├── recalculate_summary_with_median.py  # Script that produced median statistics
└── Proyecto Postulaciones.pdf    # ANII project proposal PDF
```

### Paper data pipeline

```
Experiment CSVs (results/)
  → recalculate_summary_with_median.py
    → results/combined_summary_MEDIAN.json
      → simple-SLM-FINAL-SUBMISSION.tex (tables populated manually from JSON)
        → verify_paper_values.py (checks consistency)
```

---

## 3. `Bibliografía/` - Literature Notes

```
Bibliografía/
├── Bibliografía índice.md             # Central bibliography index (2 sections)
├── ChatGPT impacts student engagement review.md  # Meta-analysis notes (17 studies)
├── Harvard AI tutor for physics.md    # Harvard Gazette article notes
├── Learn_LM report.md                # LearnLM report excerpt
├── Learning from human tutoring.md   # Wiley article link
├── Notas Tutor_CoPilot.md            # Tutor_CoPilot limitations notes
├── English as foreign language/       # EFL-specific papers
│   ├── AI_Applications_in_EFL.pdf
│   └── AI_writing_tutor_EFL.pdf
├── 1740344561233.pdf                  # Research paper PDF
├── ssrn-4337484.pdf                   # SSRN paper PDF
└── Tutor_CoPilot.pdf                  # Tutor CoPilot paper PDF
```

Organized into two themes: **"Tutores y LLMs"** (AI tutoring literature) and **"SLMs"** (small language model papers).

---

## 4. `reference_papers/` - Cited Paper Sources

```
reference_papers/
├── Nie-acl2023.tex              # ACL 2023 paper LaTeX source
└── readctrl-emnlp24.tex         # ReadCtrl EMNLP 2024 paper LaTeX source
```

---

## 5. `thoughts/` - Notes, Plans & Handoffs

```
thoughts/
├── improving_paper.md               # Notes on potential improvements to the paper
├── paper-polish-plan.md             # Paper polishing plan
├── paper_assesment_for_workshop     # Workshop assessment notes
└── shared/
    ├── handoffs/general/            # Session handoff documents
    │   ├── 2026-02-23_..._include-readctrl-and-malik-et-al.md
    │   ├── 2026-02-23_..._paper-shortening.md
    │   └── 2026-03-05_..._hyperparameter-experiments-mapping.md
    ├── research/                    # Codebase research documents
    │   ├── 2026-03-04-hyperparameter-experiments-mapping.md
    │   ├── 2026-03-10-subtoken-space-prefix-handling.md
    │   └── 2026-03-12-project-structure-audit.md
    └── plans/
        └── 2026-02-20-codigo-restructure.md  # Source layout restructure plan
```

---

## 6. Root-Level Notes

| File | Content |
|------|---------|
| `Primeros pasos tesis.md` | Chronological progress log (weeks 1-7+), the thesis "journal" |
| `Notas tiny stories.md` | TinyStories paper summary (2 lines) |
| `SLM Agents.md` | Detailed notes on SLMs in agentic architectures, deployed examples, engineering strategies |

---

## Chronological Evolution of the Project

Based on the files and git history, the project evolved through these phases:

1. **Literature survey & ideation** - Reading TinyStories, SLM surveys, AI tutoring papers; brainstorming thesis directions
2. **Presentation to research group** - SLM survey presentation for Fing-Ceibal group
3. **Initial experiments** - TinyStories 33M, early Transformers-based wrappers, 2-model scope (Qwen2 + Qwen3)
4. **llama.cpp migration** - Major performance improvement (14x load, 4x generation, 65% memory reduction)
5. **Expanded experiment scope** - 4 models (Phi3, Qwen2, Qwen3, TinyLlama), 15 prompts, 240 observations
6. **Beam search experiments** - A1-ratio beam selection on Qwen3
7. **Paper writing** - LREC 2026 submission, mean-to-median migration, reviewer feedback received
8. **Source refactor** - `feature/refactor` branch: `src/evaluation/` flattened to `src/framework/` + `src/text_complexity/`, old code archived to `legacy/`, pytest suite added
9. **Human evaluation** - Human-tagged evaluation data, correlation analysis with automatic metrics
