# BEA 2026 Shared Task: Vocabulary Difficulty Prediction for English Learners

**Status:** PRIORITY
**Deadlines are imminent** -- test data releases March 20, submissions due March 27.

---

## 1. Competition Context

**BEA 2026** (Building Educational Applications) is the 21st annual workshop co-located with **ACL 2026** in San Diego, CA (July 2-3, 2026). It hosts two shared tasks this year:

1. **Vocabulary Difficulty Prediction for English Learners** (British Council) -- **this is the target task**
2. Rubric-based Short Answer Scoring for German (DIPF/IPN) -- secondary option, German-language

### Task 1: Vocabulary Difficulty Prediction (selected)

**Goal:** Build regression models to predict the difficulty of English words for learners with different L1 backgrounds (Spanish, German, Mandarin). The target variable is a GLMM psychometric difficulty score (continuous, roughly -6 to +5; lower = harder).

**Organizers:** Mariano Felice and Lucy Skidmore (British Council)

**Why this task fits the thesis:** It is a regression task on structured text features that can be approached with small language models -- fine-tuning transformer encoders on concatenated text fields. The baseline uses XLM-RoBERTa-base (~278M params), leaving room for SLM approaches with smaller models.

### Tracks

| Track | Constraint | Models |
|-------|-----------|--------|
| **Closed** | Only provided data + standard NLP resources | One model per L1 (es, de, cn) |
| **Open** | External data + LLMs allowed | Single multilingual model or per-L1 |

### Timeline (AoE)

| Date | Milestone |
|------|-----------|
| Jan 26, 2026 | Training data + baselines released |
| **Mar 20, 2026** | **Test data release** |
| **Mar 27, 2026** | **Submission deadline** |
| Apr 3, 2026 | Results announced |
| Apr 24, 2026 | System description paper due |
| May 1, 2026 | Reviews returned |
| May 12, 2026 | Camera-ready deadline |
| Jul 2-3, 2026 | Workshop at ACL (San Diego) |

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error) -- primary ranking metric
- **Pearson correlation** -- secondary metric

### Submission Format

CSV files placed in: `predictions/{track}/{dataset_split}/{L1}/{model_name}_preds.csv`

Required columns: `item_id`, `prediction`

### Resources

- GitHub repo: https://github.com/britishcouncil/bea2026st
- Task website: https://www.britishcouncil.org/data-science-and-insights/bea2026st
- Google Group: https://groups.google.com/g/bea-2026-shared-task/
- Contact: bea2026st@britishcouncil.org
- Baseline models on HuggingFace: https://huggingface.co/lucyskidmore/models

---

## 2. Data Summary

**Location:** `data/bea2026/vocab-difficulty/` (cloned from the official repo)

### Dataset: Extended KVL (Knowledge-based Vocabulary Lists)

| Split | Rows per L1 | Total |
|-------|-------------|-------|
| Train | 6,091 | 18,273 |
| Dev | 677 | 2,031 |
| Test | TBD (Mar 20) | TBD |

### Columns

| Column | Description | Example (ES) |
|--------|-------------|-------------|
| `item_id` | Numeric ID (1-6,768), parallel across L1s | 3 |
| `L1` | Learner's native language | es |
| `en_target_word` | English word to predict difficulty for | supermarket |
| `en_target_pos` | Part of speech | noun |
| `en_target_clue` | Partial spelling hint | s__________ |
| `L1_source_word` | Translation in L1 | supermercado |
| `L1_context` | Contextual sentence in L1 | Vamos al supermercado y compramos... |
| `GLMM_score` | **Target** -- psychometric difficulty (lower=harder) | 2.733 |

### Score Distribution

| L1 | Split | Min | Max | Mean | Std |
|----|-------|-----|-----|------|-----|
| ES | train | -5.29 | +5.14 | -0.03 | 1.86 |
| ES | dev | -4.96 | +4.54 | +0.03 | 1.92 |
| DE | train | -6.50 | +4.32 | -0.05 | 1.77 |
| DE | dev | -5.54 | +4.06 | -0.09 | 1.83 |
| CN | train | -5.93 | +4.83 | -0.03 | 1.67 |
| CN | dev | -5.27 | +4.04 | +0.07 | 1.69 |

### POS Distribution (same across all L1s)

| POS | Train count | % |
|-----|------------|---|
| noun | 3,220 | 52.9% |
| adjective | 1,386 | 22.7% |
| verb | 1,030 | 16.9% |
| adverb | 375 | 6.2% |
| preposition | 38 | 0.6% |
| misc/other | 42 | 0.7% |

### Key Observations

1. **Scores are approximately normal** around 0, with range roughly [-6, +5].
2. **Harder words tend to be longer:** avg word length is 8.0 chars for hard words (score < -1) vs 6.1 for easy words (score > 1).
3. **Cross-language correlations are moderate:** Pearson(ES,DE)=0.684, Pearson(ES,CN)=0.634, Pearson(DE,CN)=0.662. This means difficulty rankings are related but differ meaningfully across L1s -- justifying per-L1 models and the cross-lingual open track.
4. **5,658 unique target words** out of 6,091 items (some words appear multiple times, likely with different POS or contexts).
5. **Examples:** Easiest words include "dance", "zoo", "orange", "sun"; hardest include "systematic", "testament", "analogy", "projection".

---

## 3. Baseline Performance (to beat)

| Track | L1 | RMSE | Pearson |
|-------|----|------|---------|
| Closed | ES | 1.357 | 0.748 |
| Closed | DE | 1.328 | 0.753 |
| Closed | CN | 1.175 | 0.736 |
| Open | ES | 1.206 | 0.787 |
| Open | DE | 1.149 | 0.800 |
| Open | CN | 1.021 | 0.804 |

Baseline model: **XLM-RoBERTa-base** fine-tuned for regression on concatenated fields (`L1_source_word; L1_context; en_target_clue; en_target_word`). Batch size 32, lr 3e-5, 5 epochs.

---

## 4. Proposed Approaches

### Approach A: Small Multilingual Encoders (Closed Track)

**Idea:** Replace XLM-RoBERTa-base with smaller multilingual models.

**Candidates:**
- `microsoft/Multilingual-MiniLM-L12-H384` (~117M params, ~2.3x smaller than XLM-R-base)
- `nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large` (~107M, distilled)
- `microsoft/mdeberta-v3-base` (~86M effective params with disentangled attention)

**Pros:** Directly thesis-relevant (SLM angle), faster training, competitive with larger models on many tasks after fine-tuning.
**Cons:** May lose some multilingual representation quality.

### Approach B: Feature-Augmented Regression (Closed Track)

**Idea:** Extract handcrafted features (word length, frequency, cognate similarity, POS embeddings, edit distance between clue and word) and combine them with transformer embeddings via a two-head model or simple concatenation before the regression head.

**Pros:** Linguistic features like cognate overlap (L1_source_word vs en_target_word) are strong difficulty predictors. Cheap to compute.
**Cons:** Feature engineering effort; may not generalize to test set if features overfit.

### Approach C: Ensemble of Small Models (Open Track)

**Idea:** Train multiple small models (different architectures, seeds, input orderings) and ensemble predictions via simple averaging or stacking.

**Pros:** Ensembles consistently improve RMSE in regression tasks. Still SLM-aligned if each component is small.
**Cons:** More compute at inference; paper needs to justify SLM angle.

### Approach D: Cross-lingual Transfer with Adapter Layers (Closed Track)

**Idea:** Use a shared small backbone with L1-specific adapter modules (e.g., LoRA or bottleneck adapters). Train jointly on all L1s but route through L1-specific adapters.

**Pros:** Parameter-efficient, thesis-relevant (adapter methods for SLMs), can leverage shared vocabulary structure while respecting L1 differences.
**Cons:** Adapter tuning requires careful hyperparameter search.

### Approach E: LLM-as-Judge + Calibration (Open Track)

**Idea:** Prompt a large LLM (GPT-4, Claude, Llama-3-70B) to estimate word difficulty given the context, then calibrate outputs to GLMM scores with a simple linear regression on the train set.

**Pros:** Zero-shot or few-shot, minimal training. Good for open track comparison.
**Cons:** Not thesis-aligned (uses large models); expensive; likely less precise than fine-tuned models for continuous regression.

### Recommended Strategy

1. **Primary submission (Closed):** Approach A + B -- fine-tune a small multilingual encoder with feature augmentation. This is the strongest thesis-aligned path.
2. **Secondary submission (Open):** Approach C -- ensemble of the Closed models + an adapter-based variant.
3. **Ablation for the paper:** Compare SLM vs full XLM-R-base to quantify the accuracy/efficiency tradeoff -- core thesis contribution.

---

## 5. Immediate Next Steps

- [ ] **Mar 19-20:** Set up training pipeline. Adapt the provided `finetune.py` for custom models. Test with MiniLM on the train/dev split.
- [ ] **Mar 20:** Download test data as soon as it is released.
- [ ] **Mar 20-26:** Run experiments: (1) small encoder baselines, (2) feature-augmented variants, (3) ensembles.
- [ ] **Mar 26-27:** Generate final predictions and submit.
- [ ] **Apr 1-24:** Write system description paper (5 pages + references).

---

## 6. Task 2 Reference: Rubric-based Short Answer Scoring (German)

Included for completeness. This task could be a secondary contribution but has a different timeline and requires German NLP expertise.

- **Dataset:** ALICE-LP-1.0 -- 7,899 train answers, 2,008 + 3,168 test answers
- **Task:** Classify short answers as correct/partially correct/incorrect using scoring rubrics
- **Metric:** Quadratic Weighted Kappa (QWK)
- **Deadline:** Results due March 28
- **Website:** https://edutec.science/bea-2026-shared-task/

This task is also SLM-relevant (classification with small German models like German-BERT variants), but the Vocabulary Difficulty task is a better fit given the existing multilingual focus and the tighter integration with the thesis.
