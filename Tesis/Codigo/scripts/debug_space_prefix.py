"""
Shows the probability of "dog" before and after the experiment's weighting,
using the exact same _create_logit_bias() call as the production code.

Since llama.cpp returns pre-bias logprobs, biased probs are computed manually:
  biased_prob(token) ∝ exp(baseline_logit + bias_for_that_token)
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework.models.qwen2_llamacpp_wrapper import Qwen2LlamaCppWrapper

TARGET_WORD = "dog"
WEIGHT_FACTORS = [1.5, 2.0, 4.0]
N_LOGPROBS = 40  # wide sample for renormalization

# Pre-fill assistant turn so the model is mid-sentence — next token is the animal
# The _format_prompt of Qwen2 only takes user+system, so we build the full prompt manually
SYSTEM_PROMPT = "You are a helpful English teacher for beginner students. Answer with a paragraph only with plain text"
PREFILLED_PROMPT = (
    "<|im_start|>system\n"
    f"{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n"
    "A good pet to have at home is a<|im_end|>\n"
    "<|im_start|>assistant\n"
    "A good pet to have at home is a"   # model continues from here
)
PROMPT = "A good pet to have at home is a"  # for display only


def get_logprobs(llm_raw, prompt_text, logit_bias=None, n=N_LOGPROBS):
    """One-token forward pass, returns {token_str: logprob}."""
    out = llm_raw(
        prompt_text, max_tokens=1, temperature=0.0,
        top_k=0, top_p=1.0, echo=False, logprobs=n,
        logit_bias=logit_bias or None,
    )
    lps = out["choices"][0]["logprobs"]["top_logprobs"][0]
    chosen = out["choices"][0]["text"]
    return chosen, lps


def biased_prob(baseline_lps, token_str, bias_dict_by_token_str):
    """
    Compute renormalized probability of token_str after additive logit biases.
    bias_dict_by_token_str: {token_str: bias_value}
    """
    unnorm = {t: math.exp(lp + bias_dict_by_token_str.get(t, 0.0))
              for t, lp in baseline_lps.items()}
    total = sum(unnorm.values())
    return unnorm.get(token_str, 0.0) / total


def main():
    print("Loading Qwen2LlamaCppWrapper (same as experiments)...")
    wrapper = Qwen2LlamaCppWrapper()

    # Second instance with logits_all=True to get logprobs — same model, same path
    from llama_cpp import Llama
    print("Loading second instance with logits_all=True for logprobs...")
    llm = Llama(model_path=wrapper.model_path, n_ctx=2048, n_threads=4,
                n_gpu_layers=0, verbose=False, logits_all=True)

    # --- Token IDs (exact same call as _create_logit_bias) ---
    id_no_space = llm.tokenize(TARGET_WORD.encode("utf-8"))          # bug path
    id_with_space = llm.tokenize((" " + TARGET_WORD).encode("utf-8"))  # correct path

    print(f"\n  Token IDs via llm.tokenize('{TARGET_WORD}'):    {id_no_space}")
    print(f"  Token IDs via llm.tokenize(' {TARGET_WORD}'):   {id_with_space}")
    print(f"  Are they the same? {id_no_space == id_with_space}")

    # --- Build logit_bias exactly as the experiment does ---
    # (no space prefix, no add_bos — production code)
    bias_dicts = {}
    for wf in WEIGHT_FACTORS:
        bias_dicts[wf] = wrapper._create_logit_bias(wrapper.target_vocabulary, wf)

    # Check whether " dog" token ID is actually in the bias dict
    print(f"\n  ID of ' dog' (spaced): {id_with_space}")
    for wf in WEIGHT_FACTORS:
        in_dict = all(tid in bias_dicts[wf] for tid in id_with_space)
        print(f"  bias_dict wf={wf}: ' dog' token in dict? {in_dict}")

    # --- Baseline forward pass (pre-filled assistant turn) ---
    chosen_base, lps_base = get_logprobs(llm, PREFILLED_PROMPT)
    tok_space    = " " + TARGET_WORD
    tok_no_space = TARGET_WORD
    base_p = math.exp(lps_base.get(tok_space, float("-inf")))
    print(f"\n  Prompt: '{PROMPT}'")
    print(f"  Greedy chose (baseline): {repr(chosen_base)}")
    print(f"\n  {'Condition':<45} P(' dog')")
    print(f"  {'─'*45} {'─'*9}")
    print(f"  {'BASELINE':45s} {base_p:.4f}")

    # Build token-str bias dicts for manual renorm
    # Map each token ID in the bias_dict back to its string representation
    # by checking which top-k tokens match
    for wf in WEIGHT_FACTORS:
        bd_ids = bias_dicts[wf]  # {token_id: wf}

        # Build token_str → bias for renorm using top-k strings
        bd_str = {}
        for tok_str in lps_base:
            try:
                ids = llm.tokenize(tok_str.encode("utf-8"), add_bos=False)
                if ids and ids[0] in bd_ids:
                    bd_str[tok_str] = bd_ids[ids[0]]
            except Exception:
                pass

        p_biased = biased_prob(lps_base, tok_space, bd_str)
        boosted = "← boosted" if tok_space in bd_str else ""
        print(f"  {'experiment wf='+str(wf)+' (no space prefix, BUG)':<45} {p_biased:.4f}  {boosted}")

    # Fixed version: bias on " dog" token IDs
    print()
    for wf in WEIGHT_FACTORS:
        bd_str = {tok_space: wf}   # only " dog" gets the bias
        p_biased = biased_prob(lps_base, tok_space, bd_str)
        label = f"fixed wf={wf} (space prefix, ' dog')"
        print(f"  {label:<45} {p_biased:.4f}  <- boosted")

    # Full top-10 baseline for context
    print(f"\n  Top-10 baseline tokens:")
    for tok, lp in sorted(lps_base.items(), key=lambda x: -x[1])[:10]:
        print(f"    {repr(tok):22s}  prob={math.exp(lp):.4f}")


if __name__ == "__main__":
    main()
