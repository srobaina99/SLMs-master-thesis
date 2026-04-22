# BEA 2026 System Description Paper — Design Spec

**Date:** 2026-04-22
**Author:** Santiago Robaina (RETUYT-INCO)
**Deadline:** 2026-04-24 (system description paper due)
**Venue:** BEA 2026 @ ACL 2026 (San Diego, Jul 2–3)
**Shared task:** Vocabulary Difficulty Prediction for English Learners (British Council, Felice & Skidmore)
**Page limit:** 4 pages body + unlimited references + appendix

## Tentative title

*RETUYT-INCO at BEA 2026 Shared Task: Feature-Enriched mDeBERTa for Cross-Lingual Word Difficulty Prediction*

Alternate options to consider:
- *RETUYT-INCO at BEA 2026 Shared Task: Two Lightweight Tracks for Psychometric Word Difficulty Prediction*
- *RETUYT-INCO at BEA 2026 Shared Task: When Handcrafted Features Meet a Small Multilingual Encoder*

## Author list (to confirm)

Default (matching the 2025 team): Santiago Góngora, Ignacio Sastre, Santiago Robaina, Ignacio Remersaro, Luis Chiruzzo, Aiala Rosá. Affiliation: Instituto de Computación, Facultad de Ingeniería, Universidad de la República, Montevideo, Uruguay.

Corresponding author(s) to be confirmed.

## 1. Framing and narrative

### Narrative angle (hybrid)

Two threads held together:

1. **Lightweight / efficiency continuity** (motivation): the team works under a self-imposed constraint of sub-1B models and affordable compute, reflecting the research environment in the Global South. This motivation is restated from scratch in 2026 — no self-citation of prior RETUYT-INCO papers.
2. **Feature→neural cross-pollination** (technical contribution): a LaBSE-derived cross-lingual similarity feature (`embed_cosine`) is discovered in the XGBoost track as the single highest-impact feature, then promoted into the mDeBERTa input as the `esim` text token. The two tracks inform each other sequentially.

### Track weighting (progression, not parallel)

- §3 presents the **feature + XGBoost** track first as a *self-contained* result: minimal-resources system that beats the XLM-R closed baseline on average. Not a ranking contender; fit-to-intent.
- §4 presents **mDeBERTa fine-tuning** as the next step that *absorbs* the best feature from §3 (`embed_cosine` → `esim`). This is the primary system and delivers the best test result (ES RMSE 1.094).

### Research question

Paraphrased: *Can a small multilingual encoder, enriched with handcrafted cross-lingual similarity features, match or beat a larger fine-tuned multilingual baseline on psychometric vocabulary difficulty prediction?*

### Contribution preview (to state in §1)

- Feature-only XGBoost (no neural fine-tuning) beats the XLM-R-base closed baseline on average across ES/DE/CN (1.273 vs 1.287 avg RMSE on dev).
- mDeBERTa-v3-base (~86M params) fine-tuned with feature-enriched input beats the XLM-R-base closed baseline (278M params) by a wide margin on ES: 1.103 dev / 1.094 test RMSE vs 1.357 dev / 1.257 test baseline.
- The LaBSE cross-lingual cosine feature is the single highest-impact signal in both tracks.

## 2. Section-by-section content plan

### §1 Introduction (~0.5 page)

- Motivate vocabulary difficulty prediction as a core problem for L2 learning applications (one sentence, no deep lit review).
- State the task briefly: GLMM psychometric difficulty regression, three L1s (ES/DE/CN), closed and open tracks.
- Introduce the self-imposed lightweight constraint (reframed from scratch — no self-citation of 2025/2024/2023 RETUYT papers). Motivate by Global South compute realities and privacy-constrained downstream applications.
- Preview the two-track architecture and the cross-pollination mechanism (`embed_cosine` → `esim`).
- State the three contributions listed above.

### §2 Dataset & Task (~0.4 page)

- Input schema: `en_target_word`, `en_target_pos`, `en_target_clue`, `L1_source_word`, `L1_context`.
- Target: GLMM psychometric difficulty score, continuous, ~[−6, +5], **lower = harder**.
- Split sizes: 6,091 train / 677 dev per L1.
- Metric: RMSE (primary), Pearson r (secondary).
- Tracks: closed vs open. **We focused on the closed track** — one sentence acknowledgment that the open track was not pursued.
- Cite the 2026 task overview paper (Skidmore et al. 2026 or successor — look up in the ACL Anthology once published; placeholder BibTeX entry until then).

### §3 Feature engineering and XGBoost (~1.0 page)

- Subsection structure:
  - **Feature families** (bulleted or short prose):
    - String/length: `word_len`, `source_word_len`, `len_diff`, `len_ratio`, edit distance (raw and normalised), character n-gram overlap, Jaro-Winkler, `shared_prefix_ratio`, `shared_suffix_ratio`, first-letter match.
    - Morphology: English suffix indicators (-tion, -sion, -ing, -ence, -ment, -able, -ness, -ly), prefix indicators (dis-, re-, un-, sub-, out-, over-), max consonant cluster, double letter, syllable count.
    - POS and context: POS one-hot (8 categories), context length.
    - Frequency: Zipf frequency via `wordfreq`.
    - L1-specific (ES v2): suffix transforms (en→es: -tion→-ción, -ment→-mento, etc.), `source_has_accent`, `is_multiword_source`, `is_latinate`, `syllable_count`, `source_word_len`, `len_diff`, `len_ratio`, `shared_suffix_ratio`. The 9 features listed in the ES ablation block of `results_log.md`.
    - L1-specific (DE): DE-specific suffix transforms (en→de: -tion→-ierung mentioned in `CLAUDE.md`). **Open item:** confirm from `feature_experiments.py` whether DE received its own rule set comparable to ES v2, or whether DE used the ES-designed features minus the ES-specific suffix rules. If DE parity with ES v2 is not documented, the paper must state honestly that the v2 features were ES-targeted and were used as-is for DE.
    - CN: only the L1-agnostic features apply (orthographic features are meaningless for non-Latin scripts). `embed_cosine` is the dominant signal for CN.
    - **Cross-lingual semantic similarity:** LaBSE cosine between `en_target_word` and `L1_source_word` — called `embed_cosine`.
  - **XGBoost configuration:** T2 rows 4–6 (`+ Tier-1` through `+ embed_cosine`) use the default config `n_estimators=300, depth=5, lr=0.05`. Row 7 (`+ L1-specific (v2)`) uses L1-tuned params from `tune_xgb.py` (ES: `n_est=600, depth=3, lr=0.02, subsample=0.85, colsample=0.70, alpha=0.5, lambda=2.0`; DE: `n_est=400, depth=3, lr=0.05, subsample=0.85, colsample=0.85, alpha=0.1, lambda=2.0`). One sentence in the paper: 80-iter random search with 5-fold CV moved RMSE by ~0.001 so feature engineering mattered more than hyperparameters. Footnote T2 if the mixed-config rows would confuse a reader.
  - **Feature progression table (T2)** (below, §5 Tables). Text around it emphasises: the largest single-feature drop is `embed_cosine` (−0.091 avg dev RMSE), and feature-only XGBoost already beats the XLM-R-base closed dev baseline on average (1.273 vs 1.287).
  - **Leave-one-out ablation** (1 sentence): a drop-one CV on 9 ES-targeted features confirmed each contributes positively (deltas 0.0004–0.0057); details omitted for space.

### §4 mDeBERTa fine-tuning (~1.0 page)

- Model: `microsoft/mdeberta-v3-base`, ~86M effective params, fine-tuned for regression (`num_labels=1`).
- **Input format** (verbatim): `wlen={N} | nedit={N} | pos={POS} | clue={N} | esim={N} | L1_word [SEP] L1_context [SEP] clue [SEP] en_word`. Call out that `esim` is the **cross-pollinated feature from §3** — explicitly cross-reference the XGBoost track.
- **Target scaling:** GLMM scores normalised to zero-mean/unit-variance at training time; predictions denormalised before RMSE computation.
- **Training config:** lr=2e-5, cosine schedule with 10% warmup, 10 epochs + early stopping (patience 3), batch=32, weight_decay=0.01, fp16, max_length=256. Training on Google Colab T4.
- **3-seed ensemble:** seeds {10, 42, 123}, prediction = mean of seed outputs. Report per-seed and ensemble numbers (T1).
- **Why ES only (one honest paragraph):** time and compute constraints for DE; for CN we additionally chose not to run because the handcrafted cognate/orthographic features and the `esim` token are designed for Latin-script L1s and are not expected to transfer to Mandarin (no orthographic overlap with English). Our CN submission is therefore XGBoost-only.
- **`train_final.py` note:** the submitted ES ensemble was retrained on `train + dev` combined before predicting the test set, which is why the test RMSE (1.094) is not directly comparable to the dev RMSE (1.103). State this clearly.

### §5 Results and analysis (~0.8 page)

- Lead with **T1 (dev results)** and **T3 (test leaderboard slice)**.
- Brief narrative (3–4 short paragraphs):
  1. Feature-only XGBoost matches/beats the XLM-R dev baseline on all three L1s — the `embed_cosine` feature makes the difference, especially for CN where orthographic features are meaningless.
  2. mDeBERTa ensemble substantially outperforms the feature-only model on ES (1.103 vs 1.327 dev RMSE).
  3. On the official test leaderboard, ES mDeBERTa (1.094) places RETUYT-InCo near the middle of the ES closed track, beating the XLM-R test baseline (1.257) by 13.0%. XGBoost on CN (1.106) also beats the test baseline (1.140); DE is within noise of the baseline (1.260 vs 1.258); XGBoost on ES is below baseline (1.323 vs 1.257) — honest discussion of dev/test gap.
  4. Optional: 1–2 inline qualitative word examples from `examples.md` (hard vs easy words) if space permits, otherwise omit.

### §6 Conclusions and limitations (~0.3 page)

- Restate the answer to the research question: yes, a feature-enriched ~86M encoder beats the 278M XLM-R-base closed baseline substantially on ES; feature engineering alone (XGBoost) already suffices to beat the baseline on average.
- Limitations:
  - ES-only for the neural ensemble (time/compute; no transfer of Latin-script features to CN).
  - Breadth over depth: no per-L1 mDeBERTa hyperparameter search.
  - Dev/test distribution mismatch observed for XGBoost ES.
  - Did not explore open-track resources (external data, larger LLMs).

## 3. Tables

### T1 — Dev results (main comparison table)

Columns: Model | Params | ES RMSE | DE RMSE | CN RMSE | Avg RMSE

Rows (in this order):
- XLM-RoBERTa-base (closed baseline, reference)
- `full_xgb_v2_embed` (best XGBoost, our submitted XGBoost system)
- mDeBERTa seed 10 (ES only, cells for DE/CN = "—")
- mDeBERTa seed 42 (ES only)
- mDeBERTa seed 123 (ES only)
- **mDeBERTa ensemble (ES only)** — bold

Source: `results_log.md` exp #19 (XGBoost), #20 (mDeBERTa seeds and ensemble with `esim`).

**Note:** Use experiment #20 numbers (the submitted `mdeberta_embed_ensemble` with `esim`), not #17. Double-check per-seed Pearson values if reporting Pearson — currently only ES RMSE is listed in the log.

### T2 — Feature progression (XGBoost section)

Columns: Feature set | Model | ES RMSE | DE RMSE | CN RMSE | Avg | Δ vs prev

Rows:
1. `word_len` | LR | 1.778 | 1.707 | 1.509 | 1.665 | —
2. + edit distance | LR | 1.705 | 1.596 | 1.487 | 1.596 | −0.069
3. + POS | LR | 1.663 | 1.536 | 1.493 | 1.564 | −0.032
4. + Tier-1 morphology | XGBoost | 1.613 | 1.524 | 1.459 | 1.532 | −0.032
5. + Tier-3 features | XGBoost | 1.464 | 1.421 | 1.257 | 1.381 | −0.151
6. **+ `embed_cosine`** (LaBSE) | XGBoost | **1.350** | **1.339** | **1.181** | **1.290** | **−0.091**
7. + L1-specific (v2) | XGBoost | 1.327 | 1.334 | 1.158 | 1.273 | −0.017
8. XLM-R baseline (reference) | XLM-R-278M | 1.357 | 1.328 | 1.175 | 1.287 | —

Bold the `embed_cosine` row. Model column makes the LR→XGBoost transition explicit.

Source: `results_log.md` experiments #1, #4, #12, #15, #16, #18, #19.

### T3 — Official test leaderboard slice

Columns: Track | L1 | System | RMSE | Pearson | Rank | Total teams | Δ vs baseline

Rows:
- Closed | ES | mDeBERTa ensemble | 1.094 | 0.843 | (rank) | (total) | −13.0%
- Closed | ES | XGBoost | 1.323 | 0.713 | (rank) | (total) | +5.3%
- Closed | DE | XGBoost | 1.260 | 0.713 | (rank) | (total) | +0.2%
- Closed | CN | XGBoost | 1.106 | 0.754 | (rank) | (total) | −3.0%

Rank and total-teams counts to be read off `results_summary_test.md`. An early count: ES closed ensemble sits at roughly 20/~58 submissions; XGBoost on ES at ~52/58; DE XGBoost at ~47/~54; CN XGBoost at ~45/~54. Confirm exact rank-per-team (collapsing to best submission per team) when writing.

Source: `results_log.md` submission table and `results_summary_test.md`.

## 4. Figures

None. Page budget does not allow it, and three tables carry the numerical story.

## 5. Writing conventions

- **Style:** concise, empirical, no marketing language. Tables for numbers, prose for interpretation. Match tone and register of `bea2025_chatbots/main.tex` as a **silent** template.
- **No self-citation** of prior RETUYT-INCO BEA papers (2023, 2024, 2025). The paper stands alone.
- **Cite the 2026 baseline/overview paper** as the reference for data and baseline. BibTeX entry goes in `custom.bib`; placeholder until ACL Anthology publishes the official entry.
- LaTeX template: copy `acl.sty`, `acl_natbib.bst`, `anthology.bib`, `custom.bib` verbatim from `bea2025_chatbots/`.
- Use `\citep` for parenthetical, `\citet` for textual (as in 2025's `main.tex`).
- Numbers in tables: 3 decimal places for RMSE, 3 for Pearson. Use the `−` en-dash for negative values in prose, plain hyphen in math.
- Use commas and parentheses for mid-sentence interruption rather than em-dashes. En-dashes (` – `) are acceptable for ranges and for the interrupter role if truly needed — match the sparing usage in `bea2025_chatbots/main.tex`.

## 6. Data sources (reference index)

Every number written in the paper comes from one of these files — do not recompute:

| Number group | Source |
|---|---|
| Dev RMSE per L1 per experiment | `results_log.md`, experiments #1 through #20 |
| Submitted XGBoost (test) | `results_log.md` submission table + `results_summary_test.md` |
| Submitted mDeBERTa ensemble (test) | `results_log.md` exp #20 + `results_summary_test.md` |
| XLM-R closed dev baseline | `BEA2026_COMPETITION.md` §3 / `results_log.md` baseline block |
| XLM-R closed test baseline | `results_summary_test.md` baseline rows |
| Feature progression rows | `results_log.md` experiments #1, #4, #12, #15, #16, #18, #19 |
| Feature ablation (leave-one-out) | `results_log.md` ES ablation block |
| Ensemble seed per-seed numbers | `results_log.md` exp #20 |
| Qualitative word examples | `examples.md` |
| Task dataset stats | `BEA2026_COMPETITION.md` §2 |

If a number required by the plan is absent from these files, the plan step must flag it explicitly rather than inventing.

## 7. Out of scope

- No new experiments (paper writes from existing results only). If a number is missing, stop and flag.
- No open track discussion beyond one sentence in §2 ("we focused on closed").
- No DE/CN mDeBERTa ensembles (not run; §4 gives the honest reason).
- No figures.
- No comparison to specific other teams' methods unless one illuminates our result.
- No self-citation of prior RETUYT-INCO papers.
- No ethics/broader-impact statement beyond the standard ACL limitations section.

## 8. Deliverables

- `paper/main.tex` — the paper source
- `paper/acl.sty`, `paper/acl_natbib.bst` — ACL style (copied from `bea2025_chatbots/`)
- `paper/anthology.bib`, `paper/custom.bib` — bibliography (reuse + add 2026 task entries)
- `paper/img/` — empty unless a figure is later added
- Compiled PDF under 4 pages body (references and appendix unlimited)
- This spec committed to git

## 9. Open items for user confirmation

- Title (3 options above — pick one or propose).
- Author list and corresponding author(s).
- Whether to include 1–2 inline qualitative word examples in §5 (default: include if they fit in 1 sentence each).
- Acknowledgments: reuse 2025's ANII grant line (*FMV_1_2023_1_176581*), or new grant line to add.
- Whether the 2026 task overview paper is already in the ACL Anthology — if not, placeholder `custom.bib` entry until it is.
