# Weighting Mechanism: How Probability Weighting Works

This document explains the full end-to-end process of the **probability weighting intervention** — from the A1 vocabulary file through tokenization and logit bias construction to its effect during text generation.

## Overview

The weighting intervention increases the probability of A1-level English vocabulary tokens during text generation. It works by adding a fixed numeric bias to the raw logit scores of specific tokens before the softmax function converts them into probabilities. This happens at every decoding step throughout the entire generation.

```
A1 Vocabulary File (493 words)
    → Tokenize each word into token IDs
    → Build logit_bias dict: {token_id: weight_factor, ...}
    → Pass to llama.cpp at generation time
    → At each step: logit[token_id] += weight_factor (before softmax)
```

---

## Step 1: Loading the A1 Vocabulary

**File:** `src/framework/models/base_model.py:102-115`

When any model wrapper is instantiated, `BaseModelWrapper.__init__` loads the target vocabulary from `data/vocabularies/filtered_starters_vocab.txt`:

```python
# base_model.py:102-104
self.vocab_path = os.path.join(
    project_root, "data", "vocabularies", "filtered_starters_vocab.txt"
)
self.target_vocabulary = self._load_target_vocabulary()
```

The loading method reads each line, strips whitespace, lowercases, and filters empty lines:

```python
# base_model.py:106-115
def _load_target_vocabulary(self) -> List[str]:
    with open(self.vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip().lower() for line in f if line.strip()]
    return vocab
```

**Result:** `self.target_vocabulary` is a `List[str]` of 493 A1 English words, e.g.:

```
["a", "about", "add", "afternoon", "again", "alex", "alien", "alphabet",
 "an", "and", "angry", "animal", "ann", "anna", "answer", "apartment",
 "flat", "apple", "arm", "armchair", "ask", "baby", ...]
```

The file also includes punctuation (`.`, `,`, `!`, `?`) and model-specific stop tokens (`<|im_end|>`, `<|endoftext|>`) as entries.

---

## Step 2: The ExperimentConfig Controls

**File:** `src/framework/core/data_models.py:14-47`

Two fields on `ExperimentConfig` control whether and how much weighting is applied:

```python
# data_models.py:23-28
config_weighting: bool = False   # Toggle: apply weighting or not
weight_factor: float = 1.0      # The bias value (default: no effect)
```

When the factorial experiment creates weighted configurations:

```python
# experiment_configs.py:94-103 (inside create_factorial_configs)
config = ExperimentConfig(
    config_weighting=config_weighting,  # True for weighted conditions
    weight_factor=1.5,                  # Fixed at 1.5 for standard experiments
    ...
)
```

For the multi-weight sweep experiment, the weight varies:

```python
# experiment_configs.py:234-243 (inside create_multi_weight_configs)
for weight in weight_factors:     # e.g., [1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0]
    configs.append(ExperimentConfig(
        config_weighting=True,
        weight_factor=weight,     # Different value per config
        config_prompting=False,   # Prompting disabled in multi-weight sweep
        ...
    ))
```

---

## Step 3: Building the logit_bias Dictionary

**File:** `src/framework/models/llamacpp_base.py:183-208`

When generation is requested with `config_weighting=True`, the model wrapper tokenizes every vocabulary word and maps each resulting token ID to the `weight_factor`:

```python
# llamacpp_base.py:183-208
def _create_logit_bias(self, vocab: List[str], weight_factor: float) -> Dict[int, float]:
    if not self.llm or not vocab:
        return {}

    logit_bias = {}
    for word in vocab:
        try:
            tokens = self.llm.tokenize((" " + word).encode('utf-8'), add_bos=False)
            for token_id in tokens:
                logit_bias[token_id] = weight_factor
        except Exception as e:
            continue

    return logit_bias
```

### How tokenization works

Each word is prepended with a space (`" " + word`) and encoded to UTF-8 bytes, then passed to the model's own tokenizer via `self.llm.tokenize(..., add_bos=False)`. The space prefix ensures token IDs match how words appear mid-sentence (after a space) in BPE-based tokenizers. `add_bos=False` prevents a beginning-of-sequence token from being included. This returns a list of integer token IDs. A word may produce one or more tokens depending on the model's vocabulary:

| Word | Possible tokens | Token IDs (example) |
|---|---|---|
| `"dog"` | 1 token | `[1234]` |
| `"armchair"` | 2 tokens | `[567, 890]` |
| `"watermelon"` | 3 tokens | `[111, 222, 333]` |

All resulting token IDs get the same flat `weight_factor` value. For a word that tokenizes into multiple sub-tokens, every sub-token receives the bias independently.

### Example output

With `weight_factor=1.5` and a simplified vocabulary of `["dog", "cat", "armchair"]`:

```python
logit_bias = {
    1234: 1.5,   # "dog" → token 1234
    5678: 1.5,   # "cat" → token 5678
    567:  1.5,   # "armchair" → first sub-token
    890:  1.5,   # "armchair" → second sub-token
}
```

If two different vocabulary words share a sub-token, the dictionary key is simply overwritten with the same value (all values are identical).

---

## Step 4: Generation with Logit Bias

**File:** `src/framework/models/llamacpp_base.py:210-289`

During response generation, the bias dictionary is conditionally built and passed to the llama.cpp inference call:

```python
# llamacpp_base.py:242-259
logit_bias = {}
if config.config_weighting and self.target_vocabulary:
    logit_bias = self._create_logit_bias(
        self.target_vocabulary,
        config.weight_factor
    )

output = self.llm(
    formatted_prompt,
    max_tokens=config.max_new_tokens,
    temperature=config.temperature,
    top_p=config.top_p,
    top_k=config.top_k,
    stop=self._get_stop_tokens(),
    echo=False,
    logit_bias=logit_bias if logit_bias else None
)
```

The ternary `logit_bias if logit_bias else None` ensures that an empty dict (when `config_weighting=False`) becomes `None`, meaning no bias is applied.

---

## Step 5: What Happens Inside llama.cpp

The `logit_bias` parameter in `llama-cpp-python` maps directly to an additive modification of the model's raw output logits. At **every** autoregressive decoding step:

1. The model computes a raw logit score for every token in its vocabulary
2. For each `token_id` in the `logit_bias` dictionary, `logit_bias[token_id]` is **added** to that token's logit
3. Temperature scaling, top-k, and top-p filtering are applied
4. Softmax converts the modified logits into a probability distribution
5. A token is sampled from this distribution

### Mathematical effect

Adding a constant `b` to a token's logit before softmax multiplies its probability by `e^b` relative to unbiased tokens:

```
P(token_with_bias) = e^(logit + b) / Z = e^b * e^logit / Z
```

| weight_factor | Additive logit bias | Probability multiplier |
|---|---|---|
| 1.0 | +1.0 | e^1.0 = 2.72x |
| 1.5 | +1.5 | e^1.5 = 4.48x |
| 2.0 | +2.0 | e^2.0 = 7.39x |
| 2.5 | +2.5 | e^2.5 = 12.18x |
| 3.0 | +3.0 | e^3.0 = 20.09x |
| 4.0 | +4.0 | e^4.0 = 54.60x |

With `weight_factor=1.5`, every A1 vocabulary token is approximately **4.5 times more likely** to be selected at each generation step compared to an otherwise equal non-A1 token.

### Key characteristics

- **Static:** The same bias dictionary is built once before generation and applies identically at every token position
- **Uniform:** All A1 vocabulary tokens receive the same bias value
- **Position-independent:** The bias applies at every step, not just at word boundaries
- **Cumulative with other sampling:** The bias interacts with temperature, top-k, and top-p — a high temperature reduces the relative impact of the bias, while a low temperature amplifies it

---

## Note: Spec Document vs. Actual Implementation

The experiment specification (`docs/HYPERPARAMETER_EXPERIMENTS.md:144`) describes the logit bias formula as:

```python
logit_bias = log(weight_factor)    # natural logarithm
```

Under this formula, `weight_factor=1.5` would produce `log(1.5) = +0.41` added to each logit, resulting in exactly a `1.5x` probability multiplier (since `e^log(1.5) = 1.5`).

The actual code at `llamacpp_base.py:203` applies `weight_factor` directly:

```python
logit_bias[token_id] = weight_factor    # no log() transform
```

| weight_factor | Spec formula: log(wf) → multiplier | Actual code: wf directly → multiplier |
|---|---|---|
| 1.5 | +0.41 → 1.5x | +1.5 → 4.48x |
| 2.0 | +0.69 → 2.0x | +2.0 → 7.39x |
| 3.0 | +1.10 → 3.0x | +3.0 → 20.09x |
| 4.0 | +1.39 → 4.0x | +4.0 → 54.60x |

The actual implementation applies a **much stronger** bias than the spec document describes.

---

## Full Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  filtered_starters_vocab.txt (493 words)                     │
│  "a", "about", "add", "afternoon", ..., ".", ",", "!", "?"  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │  BaseModelWrapper.__init__()
                       │  _load_target_vocabulary()
                       │  → line.strip().lower(), filter empty
                       ▼
              self.target_vocabulary: List[str]
                       │
                       │  At generation time:
                       │  config.config_weighting == True?
                       │
                  ┌────┴────┐
                  │  YES    │  NO → logit_bias = {} → None
                  └────┬────┘
                       │
                       │  _create_logit_bias(vocab, weight_factor)
                       │  For each word:
                       │    self.llm.tokenize((" " + word).encode('utf-8'), add_bos=False)
                       │    → [token_id_1, token_id_2, ...]
                       │    logit_bias[token_id] = weight_factor
                       ▼
              logit_bias: Dict[int, float]
              e.g. {1234: 1.5, 5678: 1.5, 890: 1.5, ...}
                       │
                       │  self.llm(formatted_prompt, ..., logit_bias=logit_bias)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  llama.cpp decoding loop (repeated for each generated token) │
│                                                              │
│  1. Model computes raw logits for all ~32k tokens            │
│  2. For token_id in logit_bias:                              │
│       logits[token_id] += 1.5  (weight_factor)               │
│  3. Apply temperature (0.7), top_k (50), top_p (0.95)       │
│  4. softmax(logits) → probability distribution               │
│  5. Sample next token                                        │
│  6. Repeat until stop token or max_tokens                    │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
              Generated response text
```

---

## Running Weight Experiments

### Standard factorial experiment (weight_factor fixed at 1.5)

```bash
python scripts/run_experiment.py --experiment all --prompts 5
# Creates 4 models × 4 configs (control, weighted, prompted, both)
# Weighted configs use weight_factor=1.5
```

### Multi-weight sweep (variable weight_factor)

```bash
python scripts/run_experiment.py --experiment multi_weight \
    --weights 1.0,1.3,1.5,2.0,2.5,3.0,4.0 --prompts 5
# Creates 4 models × 7 weights = 28 configs
# config_prompting=False for all multi-weight configs
```

### Output

Results are saved to `results/multi/`:
- `multi_weight_experiment_specification_<MMDD_HHMM>.csv` — European decimal format
- `full_data/multi_weight_experiment_full_<MMDD_HHMM>.csv` — Standard CSV
- `multi_weight_experiment_summary_<MMDD_HHMM>.json` — Aggregated stats

---

## Key Source Files

| File | Relevant Lines | Role |
|---|---|---|
| `data/vocabularies/filtered_starters_vocab.txt` | All | A1 vocabulary source (493 words) |
| `src/framework/models/base_model.py` | 102-115 | Vocabulary loading |
| `src/framework/core/data_models.py` | 23-28 | `config_weighting` and `weight_factor` fields |
| `src/framework/models/llamacpp_base.py` | 183-208 | `_create_logit_bias()` — dict construction |
| `src/framework/models/llamacpp_base.py` | 242-259 | Gate check and `self.llm()` call |
| `src/framework/experiments/experiment_configs.py` | 58-120 | Factorial config factory (`weight_factor=1.5`) |
| `src/framework/experiments/experiment_configs.py` | 214-254 | Multi-weight config factory (variable) |
| `src/framework/experiments/factorial_experiment.py` | 421-559 | Multi-weight experiment loop |
| `scripts/run_experiment.py` | 86-107 | CLI entry point for multi-weight |
