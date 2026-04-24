# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current phase: system description paper

Experimentation is **done**. Submissions are in. The active deliverable is the **BEA 2026 system description paper (due Apr 24, 2026)**, for the workshop at ACL 2026 (San Diego, Jul 2-3).

All work now is writing, not modeling. Do not propose new experiments, refactors, or code changes unless the paper explicitly needs them (e.g., a missing number for a table).

## Task recap (for paper context)

Regression: predict GLMM psychometric difficulty scores (~[-6, +5], lower = harder) for English target words given an L1 cue (L1 source word + L1 context sentence + partial English spelling clue + POS). L1s: Spanish, German, Mandarin. **We submitted two approaches: xgboost (3 langs) and ensemble (ES only)** -CN was skipped because our cognate/orthographic intuitions don't transfer to non-Latin scripts.

**Metric:** RMSE (primary), Pearson r (secondary).

**Baseline to beat (XLM-RoBERTa-base, 278M params, fine-tuned):**

| Track  | ES RMSE | DE RMSE | CN RMSE |
| ------ | ------- | ------- | ------- |
| Closed | 1.357   | 1.328   | 1.175   |
| Open   | 1.206   | 1.149   | 1.021   |

**Our best (Closed):** `mdeberta_embed_ensemble` — ES RMSE **1.103** (-18.7% vs XLM-R closed baseline; -8.5% vs XLM-R open baseline). mDeBERTa-v3-base is ~86M params, so we also win on efficiency

## System architecture (what the paper describes)

Two parallel modeling tracks that cross-pollinate:

1. **Feature-based XGBoost** (`feature_experiments.py`). Handcrafted string + semantic features: word length, Levenshtein, Jaro-Winkler, char n-gram overlap, Zipf frequency (`wordfreq`), POS one-hot, L1-specific suffix-transform rules (ES: `-tion→-ción`; DE: `-tion→-ierung`), and LaBSE cross-lingual embedding cosine (`embed_cosine`). Best feature-only config `full_xgb_v2_embed` hits avg RMSE 1.273 — already beats the XLM-R closed baseline without any neural fine-tuning.
2. **mDeBERTa fine-tuning** (`finetune/train.py`). `microsoft/mdeberta-v3-base` fine-tuned for regression with computed features **prepended as text tokens** (`wlen=N | nedit=N | pos=X | clue=N | esim=N | <source> [SEP] <context> [SEP] <clue> [SEP] <en_word>`). Target scaling to zero-mean/unit-variance, cosine LR schedule with 10% warmup, 3-seed ensemble. See `finetune/spec.md` for the full config.

The LaBSE `embed_cosine` feature was the single largest gain in track 1 (-0.086 avg RMSE going `full_xgb_v2` → `full_xgb_v2_embed`) and was promoted into track 2 as the `esim` input token — this is a central narrative thread for the paper.

## Source-of-truth files (read these before writing)

- **`results_log.md`** — every experiment run, with RMSE/Pearson per L1 and what we learned. This is where numbers for the paper come from.
- **`BEA2026_COMPETITION.md`** — competition framing, data schema, score distributions, baseline details.
- **`examples.md`** — curated hard/easy ES word examples, useful for qualitative analysis in the paper.
- folder **`submission/`** — final submissions to the competiton
- **`results_summary_test.md`** — final results and leaderboard of the competition (Ours is RETUYT-inco)
- **`paper/main.tex`** — the active BEA 2026 draft. ACL style files and bibs (`acl.sty`, `acl_natbib.bst`, `anthology.bib`, `custom.bib`) live alongside it in `paper/`; compiled output is `paper/main.pdf`.

## Paper style and writing conventions

Take style and structure from **`bea2025_chatbots/`**, our camera-ready BEA 2025 paper (same venue, same author). Relevant files:

- `bea2025_chatbots/main.tex` — final camera-ready version. Use this as the reference for tone, section structure, table/figure formatting, and citation style.
- `bea2025_chatbots/acl.sty`, `acl_natbib.bst` — ACL style files (the BEA template uses these; copy them verbatim for the new paper).
- `bea2025_chatbots/anthology.bib`, `custom.bib` — bibliography sources to draw from when a previously-cited paper recurs.
- `bea2025_chatbots/galpon/main_pre_reviews.tex` and `reviews_BEA2025.tex` — the pre-review draft and reviewer feedback. Worth skimming: reviewer complaints from last year are things **not** to repeat this year.

Match that paper's register: concise, empirical, no marketing language. Tables for numbers, prose for interpretation. Cite the baseline paper (~Skidmore et al. 2025, `10.18653/v1/2025.bea-1.12`~ new 2026 one) as the reference for the shared task data and baseline.

## Minimal operational notes

If a paper-writing task genuinely requires re-running something (e.g., a missing ablation number):

```bash
source ../Tesis/Codigo/venv/bin/activate
python feature_experiments.py   # cached; only new EXPERIMENTS entries run
```

Fine-tuning (`finetune/train.py`) was run on **Google Colab T4**, not locally. If a number needs to be regenerated, notify it.

## End goal

4 page paper reviewed and excelent camera ready
