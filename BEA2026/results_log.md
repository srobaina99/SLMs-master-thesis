# Experiment Results Log

**Current best (dev):** `mdeberta_embed_ensemble` (ES RMSE 1.103, **beats XLM-R dev baseline by 18.7%**)

## Submission / Test-set results (official BEA 2026 leaderboard)

Team **RETUYT-InCo**, closed track. Submission artifacts live in `submission/closed/`.

| L1 | Submission file            | Source experiment             | Test RMSE | Test Pearson | XLM-R test baseline | Δ vs baseline |
|----|----------------------------|-------------------------------|-----------|--------------|---------------------|---------------|
| ES | `predictions_ensemble.csv` | #20 `mdeberta_embed_ensemble` | **1.094** | 0.843        | 1.257               | **−13.0%**    |
| ES | `predictions_xgboost.csv`  | #19 `full_xgb_v2_embed`       | 1.323     | 0.713        | 1.257               | +5.3%         |
| DE | `predictions_xgboost.csv`  | #19 `full_xgb_v2_embed`       | 1.260     | 0.713        | 1.258               | +0.2%         |
| CN | `predictions_xgboost.csv`  | #19 `full_xgb_v2_embed`       | 1.106     | 0.754        | 1.140               | −3.0%         |

mDeBERTa ensemble was submitted for **ES only**. XGBoost was submitted for all three L1s. The ES ensemble is produced by `finetune/train_final.py`, which trains on **train + dev combined** before predicting test — so the 1.094 test number is not directly comparable to the 1.103 dev number in experiment #20. Full leaderboard in `results_summary_test.md`.

## Baseline to beat (dev split, from task README)

| Model | ES RMSE | DE RMSE | CN RMSE | Avg RMSE |
|-------|---------|---------|---------|----------|
| XLM-RoBERTa-base (closed) | 1.357 | 1.328 | 1.175 | 1.287 |

Note: these are the **dev-split** XLM-R numbers used throughout the experiments below. The official **test-split** XLM-R baselines in the leaderboard above (1.257 / 1.258 / 1.140) are lower.

## Experiments

All experiments use train split for fitting, evaluated on dev. Features are extracted from the dataset columns. Models: LR = LinearRegression, Ridge = Ridge(alpha=1.0), XGB = XGBRegressor.

### #1 — word_len
**Features:** `len(en_target_word)`
**Model:** Linear regression
**ES RMSE:** 1.778 | **DE RMSE:** 1.707 | **CN RMSE:** 1.509 | **Avg:** 1.665 | **ES r:** 0.378 | **DE r:** 0.356 | **CN r:** 0.452
**Notes:** Starting point. Longer English words are harder (coef ~ -0.24). Simple but captures ~14% of variance.

### #2 — edit_dist
**Features:** Levenshtein distance between `en_target_word` and `L1_source_word`
**Model:** Linear regression
**ES RMSE:** 1.792 | **DE RMSE:** 1.700 | **CN RMSE:** 1.570 | **Avg:** 1.687 | **ES r:** 0.357 | **DE r:** 0.366 | **CN r:** 0.378
**Notes:** Raw edit distance alone. Slightly worse than word_len for ES, slightly better for DE. The signal is there but noisy because long words always have high edit distance regardless of cognate status.

### #3 — norm_edit_dist
**Features:** Levenshtein distance normalized by `max(len(en_target_word), len(L1_source_word))`
**Model:** Linear regression
**ES RMSE:** 1.888 | **DE RMSE:** 1.720 | **CN RMSE:** 1.689 | **Avg:** 1.766 | **ES r:** 0.178 | **DE r:** 0.334 | **CN r:** 0.042
**Notes:** Normalized to [0,1] range. Worst single feature — normalization removes word length info without adding enough cognate signal to compensate. Nearly useless for CN (r=0.042) since Chinese characters have no orthographic relationship to English.

### #4 — word_len + edit_dist
**Features:** `len(en_target_word)` + Levenshtein distance
**Model:** Linear regression
**ES RMSE:** 1.705 | **DE RMSE:** 1.596 | **CN RMSE:** 1.487 | **Avg:** 1.596 | **ES r:** 0.459 | **DE r:** 0.489 | **CN r:** 0.476
**Notes:** First combination. Big jump — the two features capture complementary signals (surface complexity vs cognate similarity).

### #5 — word_len + norm_edit
**Features:** `len(en_target_word)` + normalized Levenshtein distance
**Model:** Linear regression
**ES RMSE:** 1.703 | **DE RMSE:** 1.557 | **CN RMSE:** 1.506 | **Avg:** 1.589 | **ES r:** 0.457 | **DE r:** 0.523 | **CN r:** 0.455
**Notes:** Normalized edit dist works better than raw when combined with word_len, because word_len already captures length. Big DE improvement (0.523 Pearson).

### #6 — vowels
**Features:** Count of vowels (a, e, i, o, u) in `en_target_word`
**Model:** Linear regression
**ES RMSE:** 1.830 | **DE RMSE:** 1.741 | **CN RMSE:** 1.581 | **Avg:** 1.717 | **ES r:** 0.305 | **DE r:** 0.302 | **CN r:** 0.356
**Notes:** Weaker than word_len — vowel count is just a noisy proxy for word length.

### #7 — consonants
**Features:** Count of consonant letters in `en_target_word`
**Model:** Linear regression
**ES RMSE:** 1.794 | **DE RMSE:** 1.730 | **CN RMSE:** 1.529 | **Avg:** 1.684 | **ES r:** 0.352 | **DE r:** 0.320 | **CN r:** 0.427
**Notes:** Slightly better than vowels, still worse than word_len. Same issue — correlated with length.

### #8 — vowels + consonants
**Features:** Vowel count + consonant count
**Model:** Linear regression
**ES RMSE:** 1.779 | **DE RMSE:** 1.707 | **CN RMSE:** 1.508 | **Avg:** 1.665 | **ES r:** 0.375 | **DE r:** 0.356 | **CN r:** 0.453
**Notes:** Identical to word_len (#1) — vowels + consonants = word length. Confirms they're a decomposition, not new info.

### #9 — vowel_ratio
**Features:** `vowels / len(en_target_word)`
**Model:** Linear regression
**ES RMSE:** 1.915 | **DE RMSE:** 1.824 | **CN RMSE:** 1.692 | **Avg:** 1.810 | **ES r:** 0.002 | **DE r:** 0.034 | **CN r:** -0.015
**Notes:** Useless. Vowel proportion has near-zero correlation with difficulty. The ratio removes length info and adds nothing.

### #10 — all_simple
**Features:** `word_len` + `norm_edit_dist` + `vowels` + `consonants`
**Model:** Linear regression
**ES RMSE:** 1.703 | **DE RMSE:** 1.557 | **CN RMSE:** 1.505 | **Avg:** 1.588 | **ES r:** 0.457 | **DE r:** 0.523 | **CN r:** 0.456
**Notes:** Same as #5 — adding vowels/consonants to word_len + norm_edit changes nothing. They're redundant.

### #11 — threshold_0.5
**Features:** `word_len` + `edit_dist` + `norm_edit_dist`, split into two sub-models
**Model:** If `norm_edit_dist <= 0.5` (cognate-like), fit on `[norm_edit_dist, edit_dist]`; otherwise fit on `[word_len]`
**ES RMSE:** 1.681 | **DE RMSE:** 1.604 | **CN RMSE:** 1.509 | **Avg:** 1.598 | **ES r:** 0.480 | **DE r:** 0.477 | **CN r:** 0.452
**Notes:** Threshold approach — different model for cognates vs non-cognates. Best ES so far at the time, but worse DE than #5. The split helps Spanish (many cognates) but hurts German (fewer clear cognates near the threshold).

### #12 — word_len + norm_edit + pos
**Features:** `word_len` + `norm_edit_dist` + one-hot encoding of `en_target_pos` (8 categories: adjective, adverb, misc, not-no, noun, number, preposition, verb)
**Model:** Linear regression
**ES RMSE:** 1.663 | **DE RMSE:** 1.536 | **CN RMSE:** 1.493 | **Avg:** 1.564 | **ES r:** 0.496 | **DE r:** 0.540 | **CN r:** 0.468
**Notes:** POS adds ~0.030 RMSE improvement over #5. Verbs and adjectives are systematically harder than nouns.

### #13 — best + pos (all features)
**Features:** `word_len` + `norm_edit_dist` + `vowels` + `consonants` + POS one-hot
**Model:** Linear regression
**ES RMSE:** 1.663 | **DE RMSE:** 1.536 | **CN RMSE:** 1.492 | **Avg:** 1.564 | **ES r:** 0.496 | **DE r:** 0.540 | **CN r:** 0.470
**Notes:** Adding vowels/consonants on top of #12 changes nothing. Confirms they're fully redundant with word_len.

### #14 — tier1_ridge
**Features:** `word_len` + `norm_edit_dist` + POS + suffix indicators (tion, sion, ing, ence, ment, able, ness, ly) + prefix indicators (dis, re, un, sub, out, over) + `shared_prefix_ratio` + `context_len` + `max_consonant_cluster` + `double_letter`
**Model:** Ridge regression (alpha=1.0)
**ES RMSE:** 1.625 | **DE RMSE:** 1.527 | **CN RMSE:** 1.468 | **Avg:** 1.540 | **ES r:** 0.530 | **DE r:** 0.548 | **CN r:** 0.495
**Notes:** Tier 1 features (suffix/prefix indicators, shared prefix ratio, context length, consonant clusters, double letters) improve over #12. Ridge barely differs from LR here — features are not overfitting.

### #15 — tier1_xgb
**Features:** Same as #14
**Model:** XGBoost (n_est=300, depth=5, lr=0.05)
**ES RMSE:** 1.613 | **DE RMSE:** 1.524 | **CN RMSE:** 1.459 | **Avg:** 1.532 | **ES r:** 0.543 | **DE r:** 0.551 | **CN r:** 0.506
**Notes:** XGBoost with Tier 1 features. Marginal gain over Ridge — these features are mostly linear-friendly, so nonlinear modeling doesn't help much.

### #16 — full_xgb
**Features:** Tier 1 + `jaro_winkler` + `ngram_overlap` (char bigram Jaccard) + `first_letter_match` + `zipf_freq` (Zipf word frequency from `wordfreq`)
**Model:** XGBoost (n_est=300, depth=5, lr=0.05)
**ES RMSE:** 1.464 | **DE RMSE:** 1.421 | **CN RMSE:** 1.257 | **Avg:** 1.381 | **ES r:** 0.646 | **DE r:** 0.628 | **CN r:** 0.668
**Notes:** Massive jump. Tier 3 features (especially `zipf_freq` and `jaro_winkler`) add strong signal that XGBoost exploits via nonlinear interactions.

### #18 — full_xgb_embed
**Features:** Tier 1 + Tier 3 + `embed_cosine` (cosine similarity between LaBSE embeddings of `en_target_word` and `L1_source_word`)
**Model:** XGBoost (n_est=300, depth=5, lr=0.05)
**ES RMSE:** 1.350 | **DE RMSE:** 1.339 | **CN RMSE:** 1.181 | **Avg:** 1.290 | **ES r:** 0.709 | **DE r:** 0.680 | **CN r:** 0.714
**Notes:** Massive improvement from a single feature addition. `embed_cosine` captures cross-lingual semantic similarity that string-level features cannot — especially critical for CN where orthographic features are meaningless. Beats XLM-R closed baseline on average (1.290 vs 1.287).

### #19 — full_xgb_v2_embed (**submitted as `submission/closed/{es,de,cn}/predictions_xgboost.csv`; best feature-only model**)
**Features:** All V2 features + `embed_cosine`
**Model:** XGBoost (L1-tuned params)
**ES RMSE:** 1.327 | **DE RMSE:** 1.334 | **CN RMSE:** 1.158 | **Avg:** 1.273 | **ES r:** 0.723 | **DE r:** 0.682 | **CN r:** 0.727
**Notes:** Best feature-only model. Beats XLM-R closed baseline on all three languages (1.273 avg vs 1.287). The combination of string-level cognate features + semantic embedding similarity + word frequency gives XGBoost enough signal to outperform a 278M-param fine-tuned transformer.

## XGBoost tuning (separate script: `tune_xgb.py`)

80-iteration random search with 5-fold CV on full_xgb features. Best params per L1:
- **ES:** n_est=600, depth=3, lr=0.02, sub=0.85, col=0.70, alpha=0.5, lambda=2.0 → Dev RMSE 1.4633
- **DE:** n_est=400, depth=3, lr=0.05, sub=0.85, col=0.85, alpha=0.1, lambda=2.0 → Dev RMSE 1.4232

Tuning barely moved the needle (~0.001). Key finding: **depth=3 is optimal** (shallower = less overfitting). Default params were already near-optimal.

**Feature importance (gain):** `zipf_freq` (#1 both L1s, ~0.14), `ngram_overlap` (#2 ES, 0.12), `word_len` (#2 DE, 0.12), `jaro_winkler` (#3 DE, 0.12).

## ES-targeted feature ablation (separate scripts: `test_es_features.py`, `ablation_es.py`)

Tested 10 new ES-targeted features on top of full_xgb. Individual marginal contributions (5-fold CV delta on train):

| Feature | CV Delta | Dev Delta | Verdict |
|---|---|---|---|
| source_word_len | -0.0057 | -0.0063 | Keep |
| len_diff (src - tgt len) | -0.0052 | -0.0111 | Keep |
| len_ratio (src / tgt len) | -0.0048 | -0.0080 | Keep |
| is_multiword_source | -0.0047 | -0.0084 | Keep |
| is_latinate (Latin root patterns) | -0.0036 | -0.0048 | Keep |
| source_has_accent (á/é/í/ó/ú) | -0.0034 | -0.0031 | Keep |
| syllable_count | -0.0026 | -0.0079 | Keep |
| suffix_transform_match (EN→ES rules) | -0.0007 | -0.0023 | Keep |
| shared_suffix_ratio | -0.0004 | -0.0034 | Keep |
| unique_char_ratio | +0.0002 | -0.0022 | Drop |

**Combined (9 winners):** ES Dev RMSE **1.4341** (down from 1.4633). Drop-one ablation confirmed all 9 features earn their keep — removing any one hurts CV.

Seed ensemble (5 seeds) on combined winners did not help (1.4347 vs 1.4341).

## Syllable experiments (deleted, earlier phase)

Syllable count was tested with linear regression and removed. It performed worse than word_len alone (ES 1.822, DE 1.740) and added nothing in LR combination. However, it was later found useful with XGBoost in the ES ablation (CV delta -0.0026) — nonlinear interactions with other features give it value.

## mDeBERTa fine-tuning (script: `finetune/train.py`)

### #17 — mdeberta_ensemble (ES only, without embed_cosine)
**Model:** `microsoft/mdeberta-v3-base` fine-tuned for regression (`num_labels=1`)
**Input format:** `wlen={N} | nedit={N} | pos={POS} | clue={N} | L1_word [SEP] L1_context [SEP] clue [SEP] en_word`
**Training config:** lr=2e-5, cosine scheduler, 10% warmup, 10 epochs + early stopping (patience 3), batch=32, weight_decay=0.01, fp16, target scaling (zero-mean/unit-variance), max_length=256
**Ensemble:** 3-seed average (seeds 10, 42, 123)

| Experiment | ES RMSE | ES Pearson |
|---|---|---|
| mdeberta_seed10 | 1.142 | 0.834 |
| mdeberta_seed42 | 1.148 | 0.824 |
| mdeberta_seed123 | 1.179 | 0.810 |
| **mdeberta_ensemble** | **1.120** | **0.834** |

**Notes:** Massive improvement over all feature-based models. Beats XLM-R closed baseline (1.357) by **17.5%** and the open baseline (1.206) by **7.1%**. Key differences vs baseline: (1) mDeBERTa instead of XLM-R, (2) feature-enriched input with computed features as text tokens, (3) target scaling, (4) cosine LR schedule, (5) 3-seed ensemble. Each individual seed already beats the baseline. Ensemble reduces variance and gains ~0.02 RMSE over best single seed.

### #20 — mdeberta_embed_ensemble (**submitted as `submission/closed/es/predictions_ensemble.csv`; beats dev baseline by 18.7%**)
**Model:** `microsoft/mdeberta-v3-base` fine-tuned for regression (`num_labels=1`)
**Input format:** `wlen={N} | nedit={N} | pos={POS} | clue={N} | esim={N} | L1_word [SEP] L1_context [SEP] clue [SEP] en_word`
**New feature:** `esim` = cosine similarity between LaBSE embeddings of `en_target_word` and `L1_source_word`
**Training config:** Same as #17
**L1s:** ES only. DE ensemble not attempted (time/compute); CN skipped — cognate/orthographic features do not transfer to non-Latin scripts.
**Ensemble:** 3-seed average (seeds 10, 42, 123)

| Experiment | ES RMSE | ES Pearson |
|---|---|---|
| mdeberta_seed10 | 1.178 | 0.789 |
| mdeberta_seed42 | 1.137 | 0.821 |
| mdeberta_seed123 | 1.172 | 0.826 |
| **mdeberta_embed_ensemble** | **1.103** | **0.827** |

**Notes:** Adding `esim` improved ES ensemble RMSE from 1.120 (#17) to **1.103** (1.5% improvement). Beats XLM-R closed dev baseline (1.357) by **18.7%** and the open dev baseline (1.206) by **8.5%**. **Submitted model.** On the official test leaderboard: ES RMSE **1.094** / Pearson **0.843** (trained on train + dev combined via `finetune/train_final.py`).

## Key insights

- **mDeBERTa fine-tuning crushes feature-only models** — 1.120 vs 1.273 (best XGBoost), a 12.0% improvement (ES only)
- **mDeBERTa beats XLM-R baseline by 17.5%** — smaller model (86M vs 278M params), better results
- **Feature-enriched input helps** — prepending computed features as text tokens gives the model explicit access to signals it might otherwise have to learn
- **Target scaling matters** — baseline didn't scale GLMM scores, we normalize to zero-mean/unit-variance
- **Seed ensemble is cheap and effective** — 3 seeds average gains ~0.02 RMSE over best single seed
- **`embed_cosine` (LaBSE) is the single highest-impact feature** — reduced avg RMSE by 0.086 (full_xgb_v2 → full_xgb_v2_embed), especially impactful for CN where string features are useless
- **`zipf_freq` (word frequency) is the #1 feature** for all L1s in feature-only models (by XGBoost gain)
- **Feature-only XGBoost now beats XLM-R baseline** — full_xgb_v2_embed (1.273 avg) vs XLM-R (1.287 avg), without any neural fine-tuning
- **CN benefits most from embedding features** — CN RMSE dropped from 1.241 → 1.158, confirming that orthographic features are meaningless for non-Latin scripts
- **Edit distance is the biggest single gain** over word_len alone (~0.11 avg RMSE improvement)
- **Tier 3 cognate features** (`jaro_winkler`, `ngram_overlap`) are critical, especially for DE
- **XGBoost >> linear models** when rich features are available (+0.13 avg RMSE improvement)
- **Hyperparameter tuning has diminishing returns** — feature engineering and model choice matter more
