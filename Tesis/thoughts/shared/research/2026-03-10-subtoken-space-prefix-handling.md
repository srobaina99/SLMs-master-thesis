---
date: 2026-03-10T12:00:00-03:00
researcher: Claude
git_commit: 2fe39cb
branch: feature/refactor
repository: SLMs-master-thesis
topic: "Do subtoken weights use words with or without space prefix?"
tags: [research, codebase, tokenization, logit-bias, weighting]
status: complete
last_updated: 2026-03-10
last_updated_by: Claude
---

# Research: Subtoken Space Prefix Handling in Weighting Intervention

**Date**: 2026-03-10
**Git Commit**: 2fe39cb
**Branch**: feature/refactor

## Research Question

When the experiment's weighting intervention tokenizes words into subtokens to build the logit_bias dict, are words considered as bare words (`"sleep"`) or with a leading space (`" sleep"`)?

## Summary

**The current production code tokenizes words WITHOUT a space prefix** — i.e., `"sleep"` not `" sleep"`.

This matters because BPE/SentencePiece tokenizers produce different token IDs for word-initial vs mid-sentence positions. The debug script (`scripts/debug_logit_bias_effect.py`) explicitly labels this as a bug. The legacy HuggingFace implementation tokenized words **both ways** (with and without space) and took the union.

## Detailed Findings

### 1. Vocabulary Loading (bare words, no spaces)

**File**: `src/framework/models/base_model.py:106-115`

Words are loaded from `data/vocabularies/filtered_starters_vocab.txt` with:
```python
vocab = [line.strip().lower() for line in f if line.strip()]
```
Result: bare lowercase words like `"sleep"`, `"dog"`, `"about"` — no leading spaces.

### 2. Current Production Code: NO space prefix

**File**: `src/framework/models/llamacpp_base.py:198-203`

```python
for word in vocab:
    try:
        tokens = self.llm.tokenize(word.encode('utf-8'))
        for token_id in tokens:
            logit_bias[token_id] = weight_factor
    except Exception as e:
        continue
```

Words are passed directly to `llm.tokenize()` without prepending a space and without `add_bos=False`. This means the tokenizer sees `"sleep"` (word-initial form), not `" sleep"` (mid-sentence form).

### 3. Debug Script: Labels This as a Bug

**File**: `scripts/debug_logit_bias_effect.py:138-177`

The debug script has a `build_bias_dict()` function with an explicit `space_prefix` parameter:

```python
def build_bias_dict(vocab, llm, bias_val, space_prefix=True):
    """
    space_prefix=True  → tokenize " word" (mid-sentence context, correct)
    space_prefix=False → tokenize "word"  (current experiment code bug)
    """
```

It runs both variants and labels `space_prefix=False` as "CURRENT CODE (BUG)" and `space_prefix=True` as "FIXED CODE".

### 4. Legacy Implementation: Both Prefixes (union)

**File**: `legacy/probability_processor.py:20-35`

The old HuggingFace-based implementation tokenized each word **twice** and took the union:

```python
word_ids = tokenizer.encode(word, add_special_tokens=False)
word_with_space_ids = tokenizer.encode(" " + word, add_special_tokens=False)
all_token_ids.update(word_ids)
all_token_ids.update(word_with_space_ids)
```

### 5. Why This Matters

In BPE tokenizers (including those used by Qwen2, Qwen3, TinyLlama, Phi3), the same surface text produces different token IDs depending on position:
- `"sleep"` → word-initial tokens (rarely appear mid-sentence during generation)
- `" sleep"` → mid-sentence tokens (the ones actually predicted during text generation)

Since the model generates text token-by-token mid-sentence, the logit bias built from bare `"sleep"` may target token IDs that rarely appear during generation, reducing the intervention's effectiveness.

## Code References

- `src/framework/models/base_model.py:106-115` — Vocabulary loading (bare words)
- `src/framework/models/llamacpp_base.py:183-208` — Production `_create_logit_bias()` (no space prefix)
- `scripts/debug_logit_bias_effect.py:138-177` — Debug script comparing both approaches
- `legacy/probability_processor.py:18-35` — Legacy implementation (both prefixes)

## Architecture Documentation

The data flow is:
```
filtered_starters_vocab.txt
  → base_model.py: line.strip().lower()  →  bare words
  → llamacpp_base.py: llm.tokenize(word.encode('utf-8'))  →  word-initial token IDs
  → logit_bias dict  →  passed to llm() during generation
```

## Open Questions

- What is the quantitative impact of the missing space prefix on actual bias effectiveness? (The debug script was built to answer this but results are not stored in the repo.)
- Should the fix follow the legacy approach (union of both) or only use the space-prefixed form?
