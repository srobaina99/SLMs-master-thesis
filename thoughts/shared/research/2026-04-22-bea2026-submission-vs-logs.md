---
date: 2026-04-22T13:35:36Z
researcher: Santiago Robaina
git_commit: 57301823366127f11cd14a17b6a995224b6fd98a
branch: feature/refactor
repository: SLMs-master-thesis
topic: "Are results_log.md and strategy.md up to date with the models actually submitted to BEA2026?"
tags: [research, bea2026, submission, results-log, strategy, audit]
status: complete
last_updated: 2026-04-22
last_updated_by: Santiago Robaina
---

# Research: Are `results_log.md` and `strategy.md` up to date with the BEA2026 submission?

**Date**: 2026-04-22T13:35:36Z
**Researcher**: Santiago Robaina
**Git Commit**: 57301823366127f11cd14a17b6a995224b6fd98a
**Branch**: feature/refactor
**Repository**: SLMs-master-thesis

## Research Question

The actual BEA2026 submission lives in `BEA2026/submission/`. Check whether `BEA2026/results_log.md` and `BEA2026/strategy.md` accurately describe the models that were actually submitted, and flag any drift.

## Summary

**`results_log.md` is partially out of date.** The two submitted models (`full_xgb_v2_embed` and `mdeberta_embed_ensemble`) are both documented there as experiments #19 and #20, and the DEV numbers for each match the submission artifacts. But the log is missing three important pieces of truth-as-of-submission:

1. **Test-set scores are nowhere in the log.** The official leaderboard numbers (in `BEA2026/results_summary_test.md`) are never cross-referenced into `results_log.md`.
2. **Experiment #20 still says "DE and CN runs are pending"** — they never happened, and `finetune/train_final.py` is hard-coded to `L1S = ["es"]`. Only ES was ever fine-tuned; DE was never attempted and CN was deliberately skipped.
3. **The log does not identify which experiments are the submission.** It calls #19 the "current best feature-only model" and #20 "current best" but does not annotate either as the chosen submission artifact.

**`strategy.md` is a pre-experiment planning document** and, per `CLAUDE.md` ("Experimentation is done"), is frozen historical context rather than a living description of the system. Roughly 60% of its proposals were actually executed; the rest were explored in a different way or dropped. It is accurate to what was *planned*, not to what was *submitted*.

**Bonus finding: `CLAUDE.md` itself is wrong.** It claims "We submitted two approaches xgboost (3 langs) and ensamble (ES and DE)." The actual submission has ensemble for **ES only** — no `predictions_ensemble.csv` exists for DE.

## Detailed Findings

### What was actually submitted

Directory structure of `BEA2026/submission/closed/`:

```
closed/
├── cn/predictions_xgboost.csv     (749 rows)
├── de/predictions_xgboost.csv     (749 rows)
└── es/
    ├── predictions_xgboost.csv    (749 rows)
    └── predictions_ensemble.csv   (749 rows)
```

Only one `predictions_ensemble.csv` exists — Spanish. No open-track submission is present.

### Mapping submission files → source experiments

Byte-for-byte diff confirms the mapping:

| Submission file                                    | Source experiment (in `results_log.md`) | Generating script              |
|----------------------------------------------------|-----------------------------------------|--------------------------------|
| `closed/es/predictions_xgboost.csv`                | #19 `full_xgb_v2_embed`                 | `feature_experiments.py`       |
| `closed/de/predictions_xgboost.csv`                | #19 `full_xgb_v2_embed`                 | `feature_experiments.py`       |
| `closed/cn/predictions_xgboost.csv`                | #19 `full_xgb_v2_embed`                 | `feature_experiments.py`       |
| `closed/es/predictions_ensemble.csv`               | #20 `mdeberta_embed_ensemble`           | `finetune/train_final.py`      |

Test predictions under `vocab-difficulty/predictions/closed/test/{es,de,cn}/full_xgb_v2_embed_preds.csv` are identical to the corresponding `submission/closed/*/predictions_xgboost.csv` files (verified with `diff`, no output). The mDeBERTa test predictions at `predictions/closed 2/test/es/mdeberta_embed_ensemble_preds.csv` are identical to `submission/closed/es/predictions_ensemble.csv`.

### `results_log.md` gaps

`BEA2026/results_log.md:1-3` — Top-of-file summary: "Current best: `mdeberta_embed_ensemble` (ES RMSE 1.103...)". That 1.103 is the **dev** RMSE. The official test RMSE for the same submission file is **1.094** (`results_summary_test.md:22`). No test number is anywhere in the log.

`BEA2026/results_log.md:175-190` — Entry #20 says:

> **L1s:** ES (DE, CN pending)
> ...
> Notes: ... DE and CN runs are pending — expect even larger gains for CN where the embedding feature had the most impact in XGBoost experiments.

This is stale. `finetune/train_final.py:42` hard-codes `L1S = ["es"]` and there is no DE or CN mDeBERTa artifact anywhere in the repo. `CLAUDE.md` now explicitly states "CN was skipped because our cognate/orthographic intuitions don't transfer to non-Latin scripts." The log should be updated to reflect the final decision, not the in-progress hope.

`BEA2026/results_log.md:117-122` — Entry #19 is labeled "**current best feature-only model**" but is not flagged as "the XGBoost submission for all three L1s." Someone reading only `results_log.md` would not be able to tell which experiment went into the zip.

### Official test-set results (missing from the log)

From `BEA2026/results_summary_test.md` — our team is `RETUYT-InCo`:

| Track / L1 | Submission file          | Test RMSE | Test Pearson | vs Baseline (closed) |
|------------|--------------------------|-----------|--------------|----------------------|
| Closed ES  | `predictions_ensemble.csv` | **1.094** | 0.843        | baseline 1.257 → -13.0% |
| Closed ES  | `predictions_xgboost.csv`  | 1.323     | 0.713        | baseline 1.257 → +5.3% (worse) |
| Closed DE  | `predictions_xgboost.csv`  | 1.260     | 0.713        | baseline 1.258 → +0.2% (roughly tied) |
| Closed CN  | `predictions_xgboost.csv`  | 1.106     | 0.754        | baseline 1.140 → -3.0% |

Notable: the ES **ensemble** test RMSE (1.094) is slightly better than the dev RMSE (1.103) reported in the log — plausibly because `train_final.py:2-5` trains on **train + dev combined** before predicting test. For XGBoost, DE test (1.260) is noticeably worse than DE dev (1.334 reported in the log), and CN test (1.106) is better than CN dev (1.158). The gap between dev and test is material and worth surfacing in the paper.

Also notable: the closed-track baselines in `results_summary_test.md` (ES 1.257 / DE 1.258 / CN 1.140) differ slightly from the "baseline to beat" table in `results_log.md:7-9` (ES 1.357 / DE 1.328 / CN 1.175). The log table is the **dev** XLM-R baseline from the task README; the leaderboard number is the **test** baseline. `CLAUDE.md` uses the dev numbers. This should be reconciled in the paper so readers know which baseline each delta is against.

### `strategy.md` status

`BEA2026/strategy.md` enumerates 10 ideas split into High/Medium impact plus "Things to Avoid." Checked against what was done:

| # | Proposal                                      | Shipped? | Evidence |
|---|-----------------------------------------------|----------|----------|
| 1 | Feature-augmented regression (cognate, length, Zipf, clue, POS) | Yes | `full_xgb_v2_embed` and mDeBERTa input format |
| 2 | Swap to mDeBERTa-v3-base                      | Yes | `finetune/train_final.py:41` |
| 3 | Ensemble averaging (3-5 seeds)                | Yes | 3 seeds (10, 42, 123) in `train_final.py:43` |
| 4 | Input reordering / drop clue                  | Partial | Current order puts `en_word` last — not reordered; clue kept |
| 5 | Cross-lingual joint training + L1 adapters    | **No**  | Each L1 trained separately |
| 6 | LR / scheduler tuning (cosine + warmup)       | Yes | `lr=2e-5`, cosine, 10% warmup (`train_final.py:47-53`) |
| 7 | Target scaling                                | Yes | Noted in `results_log.md` #17 onward |
| 8 | External difficulty features (Zipf, MRC, AoA) | Partial | `zipf_freq` via `wordfreq` — no MRC/AoA |
| 9 | LLM distillation                              | **No**  | Not in any script |
| 10 | Back-translation data augmentation           | **No**  | Not in any script |

Given `CLAUDE.md` explicitly says experimentation is over and `strategy.md` is a pre-experiment doc, the mismatches are expected. The question for the paper is whether to preserve `strategy.md` as historical context or to prune it down to only the proposals that shipped. Recommendation in the "Open Questions" section below.

### `CLAUDE.md` internal contradiction

`BEA2026/CLAUDE.md:15` states:

> We submitted two approaches xgboost (3 langs) and ensamble (ES and DE) — CN was skipped because our cognate/orthographic intuitions don't transfer to non-Latin scripts.

This is inconsistent with the filesystem: `submission/closed/de/` contains only `predictions_xgboost.csv`. The ensemble was ES-only. The CN skip reasoning is also slightly muddled — CN was skipped from the **ensemble** track, but CN *is* part of the XGBoost submission (and the CN XGBoost submission actually beats the closed-track baseline). The paper will need to phrase this carefully.

## Code References

- `BEA2026/submission/closed/es/predictions_ensemble.csv` — ES ensemble submission (RMSE 1.094 test)
- `BEA2026/submission/closed/{es,de,cn}/predictions_xgboost.csv` — XGB submissions for all three L1s
- `BEA2026/feature_experiments.py:308-335` — `full_xgb_v2_embed` config (XGB submission model)
- `BEA2026/feature_experiments.py:495` — Dev predictions saved to `predictions/closed/dev/`; no test-writing path in this script
- `BEA2026/finetune/train_final.py:2-5` — Trains on train+dev combined, predicts on test
- `BEA2026/finetune/train_final.py:42` — `L1S = ["es"]` (hard-coded ES only)
- `BEA2026/finetune/train_final.py:43,46-63` — 3-seed ensemble, cosine LR, 2e-5, 10 epochs, fp16
- `BEA2026/vocab-difficulty/predictions/closed/test/{es,de,cn}/full_xgb_v2_embed_preds.csv` — Source of the submitted XGB predictions
- `BEA2026/predictions/closed 2/test/es/mdeberta_embed_ensemble_preds.csv` — Source of the submitted ES ensemble predictions
- `BEA2026/results_summary_test.md` — Official test leaderboard (includes RETUYT-InCo's numbers)
- `BEA2026/results_log.md:175-190` — Out-of-date "pending DE/CN" note on the submitted model
- `BEA2026/CLAUDE.md:15` — Incorrect claim about DE ensemble submission

## Architecture Insights

- **Two scripts produce submission artifacts** (`feature_experiments.py` for XGB, `finetune/train_final.py` for mDeBERTa), and they live in different trees. There is no single "make submission" script, which is why drift between the submission and the narrative files is easy.
- **Dev predictions and test predictions are written to different roots.** `feature_experiments.py` writes dev-only to `BEA2026/predictions/closed/dev/`. The test XGB predictions that became the submission live under `BEA2026/vocab-difficulty/predictions/closed/test/` — i.e. inside the task's own repo. Anyone reproducing numbers needs to know this.
- **Training regime differs between dev evaluation and final submission.** `train_final.py` trains on **train + dev** then predicts test, so the 1.094 test number is not comparable to the 1.103 dev number in the log — the model saw more data for the submission. Worth stating explicitly in the paper.

## Historical Context (from thoughts/)

No prior research document in `thoughts/shared/research/` addresses submission-vs-logs consistency. The only adjacent doc is `thoughts/shared/research/2026-04-20-lean-docker-image-for-multi-weight-experiment.md`, which is unrelated.

## Open Questions

1. **Should `results_log.md` be updated with test-set numbers?** Options: (a) add a "Submission / test results" section at the top with the four leaderboard numbers, (b) leave the log as a pure dev-set experiment journal and put test numbers only in the paper. I lean (a) since the paper will need to cite these and having them in one source of truth is safer.
2. **Fix the stale "pending DE/CN" note on experiment #20.** Low-risk edit: change `L1s: ES (DE, CN pending)` to `L1s: ES only (DE ensemble not attempted; CN deliberately skipped — orthographic features do not transfer to non-Latin scripts)`.
3. **Fix `CLAUDE.md:15`** to reflect that only ES ensemble was submitted.
4. **Reconcile the two XLM-R baselines** (dev vs test) referenced across `results_log.md`, `CLAUDE.md`, and `results_summary_test.md` so the paper's "vs baseline" deltas are unambiguous.
5. **Strategy.md**: keep as historical pre-experiment context, or prune to only-shipped proposals? Given `CLAUDE.md` says experimentation is frozen, the simpler move is to leave it alone and not cite it in the paper.

## Related Research

- `thoughts/shared/research/2026-04-20-lean-docker-image-for-multi-weight-experiment.md` — adjacent thesis infrastructure (not relevant to this question)
