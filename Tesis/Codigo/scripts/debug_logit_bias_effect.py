"""
Debug script: Show exact logit bias effect using llama.cpp.

Loads Qwen3 model, runs a prompt with different weight_factor values
(both as direct logit_bias and as log(weight_factor)), and shows
the probability distribution change for A1 vocabulary tokens.
"""

import sys
import os
import math
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from llama_cpp import Llama

# --- Config ---
MODEL_PATH = os.path.join(project_root, "models/gguf/Qwen3-0.6B-Q4_0.gguf")
VOCAB_PATH = os.path.join(project_root, "data/vocabularies/filtered_starters_vocab.txt")
# Pre-fill the thinking block so the model skips it and generates a content word next.
# "A dog is a" forces competition between A1 words ("small", "good") and non-A1
# words ("domestic", "loyal", "canine") — exactly where logit bias is observable.
TEST_PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful English teacher for beginner students. "
    "Answer with a paragraph only with plain text<|im_end|>\n"
    "<|im_start|>user\n"
    "What is a dog?<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n</think>\n"
    "A dog is a"
)
WEIGHT_FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0]
N_TOKENS_TO_SHOW = 5  # top-N tokens per run
SAMPLE_A1_WORDS = ["small", "animal", "dog", "big", "run", "eat", "good", "can", "like", "has"]


def load_vocab(path):
    with open(path) as f:
        return [line.strip().lower() for line in f if line.strip()]


def get_token_ids(llm, words):
    """Get token IDs for a list of words, returning {word: [token_ids]}."""
    result = {}
    for word in words:
        try:
            ids = llm.tokenize(word.encode("utf-8"), add_bos=False)
            result[word] = ids
        except Exception as e:
            result[word] = []
    return result


def run_with_logit_bias(llm, prompt, logit_bias_dict, label, baseline_top_logprobs=None, n_tokens=5):
    """Run one forward pass with greedy decoding and show token chosen + biased probs."""
    start = time.time()
    output = llm(
        prompt,
        max_tokens=1,
        temperature=0.0,     # greedy: always pick argmax → deterministic
        top_p=1.0,
        top_k=0,
        logit_bias=logit_bias_dict if logit_bias_dict else None,
        echo=False,
        logprobs=n_tokens,
    )
    elapsed = time.time() - start

    choice = output["choices"][0]
    logprobs_data = choice.get("logprobs", {})
    top_logprobs = logprobs_data.get("top_logprobs", [{}])[0] if logprobs_data else {}
    token_chosen = choice.get("text", "")

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Token chosen (greedy): {repr(token_chosen)}")
    print(f"  Time: {elapsed:.2f}s")

    # Show actual biased distribution by adjusting the baseline logprobs
    if baseline_top_logprobs:
        print(f"  Top-{n_tokens} tokens — baseline vs biased probability:")
        print(f"  {'Token':20s}  {'Base logprob':>12}  {'Base prob':>10}  {'Bias added':>10}  {'Biased prob':>12}")
        print(f"  {'─'*20}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}")
        for tok, base_lp in sorted(baseline_top_logprobs.items(), key=lambda x: -x[1])[:n_tokens]:
            base_prob = math.exp(base_lp)
            # Look up this token's ID to find its bias
            bias = 0.0
            if logit_bias_dict:
                try:
                    ids = llm.tokenize(tok.encode("utf-8"), add_bos=False)
                    # Use first token ID if multi-token
                    if ids and ids[0] in logit_bias_dict:
                        bias = logit_bias_dict[ids[0]]
                except Exception:
                    pass
            biased_lp = base_lp + bias
            biased_prob_unnorm = math.exp(biased_lp)
            marker = " ◀ A1-biased" if bias > 0 else ""
            print(f"  {repr(tok):20s}  {base_lp:>12.3f}  {base_prob:>10.4f}  {bias:>+10.3f}  {biased_prob_unnorm:>12.4f}{marker}")
    elif top_logprobs:
        print(f"  Top-{n_tokens} tokens:")
        for tok, lp in sorted(top_logprobs.items(), key=lambda x: -x[1])[:n_tokens]:
            prob = math.exp(lp)
            print(f"    {repr(tok):20s}  logprob={lp:8.3f}  prob={prob:.4f}")

    return token_chosen, top_logprobs


def main():
    print("=" * 60)
    print("  LOGIT BIAS EFFECT DEMO — Qwen3 0.6B, llama.cpp")
    print("=" * 60)

    print(f"\nLoading model from: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=0,
        verbose=False,
        logits_all=True,
    )
    print("Model loaded.")

    # Load A1 vocab and get token IDs for sample words
    vocab = load_vocab(VOCAB_PATH)
    print(f"\nA1 vocab size: {len(vocab)} words")

    word_token_ids = get_token_ids(llm, SAMPLE_A1_WORDS)
    print("\nSample A1 word → token IDs:")
    for word, ids in word_token_ids.items():
        print(f"  {word:10s} → {ids}")

    def build_bias_dict(vocab, llm, bias_val, space_prefix=True):
        """Build token-ID → bias dict.

        space_prefix=True  → tokenize " word" (mid-sentence context, correct)
        space_prefix=False → tokenize "word"  (current experiment code bug)
        """
        d = {}
        for word in vocab:
            try:
                text = (" " + word) if space_prefix else word
                for tid in llm.tokenize(text.encode("utf-8"), add_bos=False):
                    d[tid] = bias_val
            except Exception:
                pass
        return d

    # --- Baseline (no bias) — capture top_logprobs for comparison ---
    _, baseline_lps = run_with_logit_bias(llm, TEST_PROMPT, {}, "BASELINE (no logit_bias, greedy)")

    # --- Current code (BUG): no space prefix → wrong token IDs ---
    print("\n" + "=" * 60)
    print("  CURRENT CODE (BUG): tokenize(word) — no space prefix")
    print("  Biases wrong token IDs; mid-sentence tokens get no boost")
    print("=" * 60)

    for wf in [1.5, 4.0]:
        bias_dict = build_bias_dict(vocab, llm, wf, space_prefix=False)
        label = f"wf={wf:.1f}, bias={wf:.2f} (no space prefix) — biases {len(bias_dict)} token IDs"
        run_with_logit_bias(llm, TEST_PROMPT, bias_dict, label, baseline_top_logprobs=baseline_lps)

    # --- Fixed: space prefix + direct weight_factor ---
    print("\n" + "=" * 60)
    print("  FIXED CODE: tokenize(' '+word) — correct mid-sentence IDs")
    print("  logit_bias = weight_factor  →  prob_mult = exp(wf)  [EXPONENTIAL]")
    print("=" * 60)

    for wf in WEIGHT_FACTORS:
        bias_dict = build_bias_dict(vocab, llm, wf, space_prefix=True)
        label = f"weight_factor={wf:.1f}  →  bias={wf:.2f}  →  prob_mult=exp({wf:.1f})={math.exp(wf):.2f}x"
        run_with_logit_bias(llm, TEST_PROMPT, bias_dict, label, baseline_top_logprobs=baseline_lps)

    # --- Doc intention: space prefix + log(weight_factor) ---
    print("\n" + "=" * 60)
    print("  DOC INTENTION: tokenize(' '+word) + logit_bias = log(wf)")
    print("  Effect: prob_multiplier = wf  [LINEAR — wf IS the multiplier]")
    print("=" * 60)

    for wf in WEIGHT_FACTORS:
        bias_val = math.log(wf) if wf != 1.0 else 0.0
        bias_dict = build_bias_dict(vocab, llm, bias_val, space_prefix=True)
        label = f"weight_factor={wf:.1f}  →  bias=log({wf:.1f})={bias_val:.2f}  →  prob_mult={wf:.1f}x"
        run_with_logit_bias(llm, TEST_PROMPT, bias_dict, label, baseline_top_logprobs=baseline_lps)

    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"{'weight_factor':>14} | {'current bias':>12} | {'current mult':>13} | {'doc bias':>9} | {'doc mult':>9}")
    print("-" * 65)
    for wf in WEIGHT_FACTORS:
        cur_bias = wf
        cur_mult = math.exp(wf)
        doc_bias = math.log(wf) if wf != 1.0 else 0.0
        doc_mult = wf
        print(f"{wf:>14.1f} | {cur_bias:>12.3f} | {cur_mult:>12.2f}x | {doc_bias:>9.3f} | {doc_mult:>8.1f}x")
    print()


if __name__ == "__main__":
    main()
