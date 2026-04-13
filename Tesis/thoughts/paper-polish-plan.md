# Paper Polish Plan
_Four targeted fixes for the LREC submission_

---

## 1. Add generation hyperparameters to Section 3.2.4

**Where:** Section 3.2.4 "Computational Environment" (paper lines 254–262), inside the existing bullet list.

**What to add:** A new bullet item with the exact values used in the experiment pipeline (`data_models.py` and `experiment_configs.py` are the canonical source):

```
temperature = 0.7
top_k       = 50
top_p       = 0.95
max_tokens  = 200  (~150 words)
repetition_penalty: none applied
```

Replace the current last bullet (`Acceleration: Metal...`) by inserting the new item before it so the section reads:
- Hardware
- Software (llama-cpp-python)
- Readability Computation (textstat)
- **Generation parameters** ← new
- Acceleration (MPS)

**Draft text:**
> **Generation parameters**: All models used identical decoding settings: temperature 0.7, top-$k$ 50, top-$p$ 0.95, maximum output length 200 tokens. No repetition penalty was applied.

---

## 2. Add logit weighting pseudocode to Section 3.2.1

**Where:** Section 3.2.1 "Probability Weighting Intervention" (paper lines 143–148), after the existing "Token-Level Weighting" paragraph.

**What to add:** A small algorithm block showing the per-step decoding logic. The actual implementation uses additive log-space manipulation:

```
Algorithm 1: Vocabulary Probability Weighting
─────────────────────────────────────────────
Input:  logits ∈ ℝ^|V|   (raw token scores at each decoding step)
        V_A1              (set of token IDs from A1 vocabulary)
        α = 1.5           (boost factor)

For each decoding step:
  for token_id in V_A1:
      logits[token_id] += log(α)   # additive boost in log-space
  next_token = sample(softmax(logits))
```

Note: adding `log(α)` in log-space is equivalent to multiplying the probability by α before renormalization, which is the standard soft-constraint logits processor pattern in llama-cpp-python and HuggingFace.

**Also clarify in the surrounding text:** the boost is applied at *every* decoding step (not just the first token of a word), and multi-token words receive the boost on each of their constituent sub-tokens independently.

---

## 3. Fix typos

All line numbers refer to `paper/simple-SLM-FINAL-SUBMISSION.tex`.

| Line | Current (wrong) | Correction |
|------|----------------|------------|
| 252  | `ransomized`   | `randomized` |
| 254 (section header) | `enviroment` | `environment` |
| 399 (table caption) | `Weignted` | `Weighted` |
| 75   | `adjustment the complexity` | `adjusting the complexity` |
| 93   | `inference-time controlling` | `inference-time control` |
| 176  | `primary metrics with formulas that evaluate` | `primary metrics that evaluate complexity` (incomplete clause) |
| 265  | `prompt base interventions` | `prompt-based interventions` |
| 71   | Second sentence opens with `This work` right after a sentence ending in `...this work.` | Change to `It focuses on...` or restructure |

---

## 4. Standardize model name to "Qwen2.5"

The model used is **Qwen2.5-0.5B** but the paper calls it "Qwen2" throughout. Find and replace every occurrence of `Qwen2` (when referring to the model, not to the model family in general) with `Qwen2.5`.

Key locations:
- Abstract: "Qwen2, Qwen3, Phi3, TinyLlama" → "Qwen2.5, Qwen3, Phi3, TinyLlama"
- Section 3.2 model list: "Qwen2.5 (0.5B)"
- Table 2 row headers: "Qwen2" → "Qwen2.5"
- All in-text references in Results and Discussion sections
- Figure captions (if the figure legend says "Qwen2")

The model identifier in the CSV (`model_id` column) is `Qwen2` — this is fine to leave as-is in the code/data; only the paper text needs updating.

---

## Order of execution

1. [x] Typos — mechanical find-and-replace, zero risk, do first
2. [x] Model name — global search-and-replace on the `.tex` file
3. [x] Generation hyperparameters — one new bullet in Section 3.2.4
4. [x] Pseudocode block — requires writing the algorithm environment in LaTeX
