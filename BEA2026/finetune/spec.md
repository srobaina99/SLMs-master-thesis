# Feature-Enriched mDeBERTa Ensemble — Design Spec

**Date:** 2026-03-22
**Track:** Closed
**Target:** Beat XLM-RoBERTa-base baseline (ES RMSE 1.357, DE RMSE 1.328)

## Overview

Fine-tune `microsoft/mdeberta-v3-base` on feature-enriched text inputs for vocabulary difficulty prediction. Run 3 seeds per L1 (ES, DE), average predictions for final submission.

## Input Format

Prepend computed features as text tokens before the standard baseline concatenation:

```
wlen=4 | nedit=0.75 | pos=noun | clue=0.25 | esim=0.812 | lapso </s> El eclipse solar... </s> s___ </s> span
```

### Features

| Feature | Source | Computation |
|---------|--------|-------------|
| `wlen` | `en_target_word` | `len(word)` |
| `nedit` | `en_target_word`, `L1_source_word` | `levenshtein(en, l1) / max(len(en), len(l1), 1)` |
| `pos` | `en_target_pos` | POS tag as text (e.g., `noun`) |
| `clue` | `en_target_clue`, `en_target_word` | `non_underscore_chars / len(word)` |
| `esim` | `en_target_word`, `L1_source_word` | Cosine similarity of LaBSE embeddings (cross-lingual semantic similarity) |

## Training Configuration

| Setting | Value |
|---------|-------|
| Model | `microsoft/mdeberta-v3-base` |
| Target scaling | Normalize GLMM to zero-mean, unit-variance on train; inverse-transform predictions |
| Learning rate | 2e-5 |
| LR scheduler | Cosine with 10% warmup |
| Epochs | 10 |
| Early stopping | Patience 3, metric = eval RMSE |
| Batch size | 32 |
| Weight decay | 0.01 |
| Max sequence length | 256 (sufficient for these inputs) |
| Seeds | 10, 42, 123 |
| L1s | ES, DE, CN |

## Ensemble

Simple averaging of 3-seed predictions per L1. No stacking or meta-learner.

## Colab Execution

- Runs on free-tier T4 GPU (~10 min per run, ~1 hour total for 6 runs)
- Checkpoints saved to Google Drive after each run
- Script detects completed runs and skips them on resume
- Data uploaded to Drive or Colab session storage

## Output

- Per-seed prediction CSVs (for analysis)
- Averaged ensemble prediction CSVs in submission format: `predictions/closed/{dev,test}/{es,de,cn}/mdeberta_ensemble_preds.csv`
- Evaluation metrics printed to console

## Success Criteria

- Avg RMSE < 1.34 on dev (beats closed baseline)
- Each individual seed run should be competitive (~1.2-1.35 range)
