# Improvements to Address LREC Reviewer Comments

This document maps each reviewer concern to concrete actions for the paper revision.

---

## Meta-Review: Core Issues

The meta-review identifies four top-level problems:
1. Evaluation relies exclusively on traditional readability metrics (questionable reliability on short texts)
2. No human evaluation
3. No fluency assessment or error analysis
4. Vocabulary probability weighting description is unclear

---

## Reviewer #1

### R1-A: Abstract is result-focused, not contribution-focused
**Action:** Rewrite the abstract to lead with the problem (complexity mismatch in A1 educational AI), the gap (no training-free inference-time control for SLMs), and the contribution framing, then briefly mention results.

### R1-B: Opening example may be semantically misleading ("toys and stories" vs. "books")
**Action:** Either (a) replace the example with one where the simplified output is more semantically faithful, or (b) acknowledge the semantic drift explicitly in the main text (not just Limitations) and frame it as an illustration of the fluency/faithfulness tension the paper investigates. The current example actually undermines the paper's claim by showing the intervention producing a confused output.

### R1-C: Metrics share overlapping sub-components (ASL is shared by GF and FK)
**Action:** Add a table or paragraph in the Evaluation/Methods section listing what each metric actually measures (word count, syllable count, sentence count, polysyllabic word count) and which sub-components overlap. This makes it explicit that the four metrics are not four independent signals but a partially correlated set.

### R1-D: No target values shown in tables — distance to target is unclear except in the figure
**Action:** Add a "Target (A1)" row or column to every results table showing the reference value for each metric (e.g., FK Grade ≤ 5.0, Spache ≤ 2.0, etc.). Alternatively, report delta-from-target as an additional column.

### R1-E: Readability metric reliability on short passages
**Action:** Add a paragraph in the Limitations section (or a dedicated sub-section in Evaluation) discussing metric reliability on short texts. Cite the SMOG 30-sentence requirement and similar caveats for FK and Fog. Mitigate by: (a) reporting aggregate metrics over all 15 prompts × N responses per configuration rather than per-response, and (b) noting that the experiment was designed to aggregate across 15 prompts precisely to increase reliability.

### R1-F: No analysis of what linguistic features the model actually changes
**Action:** Add a surface feature analysis table reporting, per configuration: mean sentence count, mean word count, mean word/sentence ratio, % polysyllabic words, % words outside A1 vocabulary. This makes the mechanism visible and shows *where* the gain comes from (shorter sentences vs. simpler words vs. both).

---

## Reviewer #2

### R2-A: Only 15 trials — criteria for prompt selection not explained
**Action:** Add a sub-section in Methods explaining prompt selection: the 15 prompts represent the range of A1 pedagogical question types (vocabulary definitions, simple how-to questions, yes/no concept checks, etc.). If possible, include the full prompt list in an appendix. Consider whether more prompts can be added; if not, justify the number relative to what is feasible for the models tested.

### R2-B: Metrics are outdated; human evaluation would help
**Action:**
- **Short-term:** Add at least a small human evaluation (even 2–3 annotators rating a subset of responses on a 1–5 scale for appropriateness for A1 learners). This directly addresses the meta-review's main concern.
- **Alternative/complement:** Reference more recent automatic metrics such as FKGL alternatives or corpus-based lexical complexity measures (e.g., CEFR word-list coverage using the Cambridge English Profile wordlist, or text complexity via mean log word frequency from a learner corpus). These are not "replacements" but additional signals.

### R2-C: Effect of strategies on fluency is not analyzed
**Action:** Add a fluency analysis section. Options:
- **Automatic:** Use perplexity under a reference language model (e.g., a larger Qwen or GPT-2) as a proxy for fluency. Report per configuration.
- **Human:** Include fluency as a dimension in the human evaluation mentioned in R2-B (rate 1–5: "Does this read naturally?").
- Report whether the vocabulary weighting degrades fluency (expected: yes, which would explain its limited gain in readability metrics).

---

## Reviewer #3

### R3-A: Vocabulary probability weighting implementation is not clearly described
**Action:** Expand Section 3.2 (or wherever the weighting is described) with:
- The exact logits manipulation formula: `logit[token] += log(boost_factor)` if `token ∈ A1_vocab`, or the equivalent.
- How tokenization interacts with the word list (multi-token words, subword units).
- Whether the boost is applied at every decoding step or only at the start of a new word.
- A pseudocode block or algorithm box would substantially improve replicability (Reviewer #2 also marked Replicability: No).

### R3-B: Missing prompting-only column in Table 4 (or equivalent)
**Action:** Add a column for the `Prompt=ON, Weighting=OFF` condition broken down per model. This is a straightforward addition that directly addresses the reviewer's request and makes the prompting-only vs. combined comparison explicit per model. Currently the factorial structure is described but the individual model breakdown for the prompting-only condition seems absent.

### R3-C: Syntactic control is mentioned in the prompt but not evaluated
**Action:** Either (a) add a syntactic analysis (e.g., mean dependency tree depth, mean clause count per sentence using a dependency parser) to show whether syntactic simplification is actually happening, or (b) explicitly acknowledge in Limitations that the prompt instructs syntactic simplification but the evaluation does not measure it, and this is left for future work.

### R3-D: Faithfulness is not evaluated
**Action:** Add faithfulness/semantic preservation as an evaluation dimension. Options:
- **Automatic:** Compute BERTScore or cosine similarity between the control and intervention outputs to measure semantic drift. Flag cases of high divergence.
- **Human:** Include a "Is the meaning preserved?" question in the human evaluation.
- At minimum, add a Limitations paragraph explicitly naming faithfulness as an unaddressed dimension.

### R3-E: "Control" tag not explained until Section 3.1
**Action:** In the Introduction, add a one-sentence forward-reference explaining what "Control" means in context (inference-time complexity control = applying constraints at generation time without retraining).

### R3-F: Why couldn't DExperts/FUDGE target linguistic complexity? (Section 2.3)
**Action:** Add a brief explanation: these methods require training a discriminator or expert/anti-expert model on labeled data for the target attribute. Linguistic complexity at a specific CEFR level lacks large labeled corpora suitable for this, and the auxiliary model training contradicts the training-free requirement of the deployment context.

### R3-G: Avoid "real-time" in Section 3.2.1
**Action:** Replace "real-time" with "online" or "at inference time" throughout.

### R3-H: Target audience inconsistency ("young" vs. "A1-level English learners")
**Action:** Standardize the target audience description throughout. The paper targets "beginner A1-level English learners" which may include adults. If the implementation targets children specifically (as the vocabulary list and the toy/stories example suggest), state this clearly and early. If it targets A1 adults as well, adjust the prompt and example accordingly or add a note about the childish register as a known artifact.

### R3-I: Overclaiming in Section 5.2
**Action:** Soften "Standalone vocabulary manipulation is insufficient for complexity control in most architectures." to something like "Standalone vocabulary weighting, at the 1.5× boost level tested, does not achieve the complexity reductions obtained by prompting alone, suggesting that for this parameter setting, prompting is the dominant intervention." This is more precise and defensible.

---

## Cross-Cutting: Replicability (Reviewers #2 and #3 both scored No)

**Action:** Add a dedicated "Reproducibility" sub-section in Methods or an appendix with:
- Full prompt templates (both baseline and contextual prompting)
- The complete A1 Starters vocabulary list or a pointer to it
- Exact generation hyperparameters (temperature, top-p, max tokens, etc.)
- Model versions / HuggingFace identifiers
- The logits processor implementation (pseudocode or link to code repository)

---

## Priority Order for Revision

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Human evaluation (small-scale, 2–3 annotators, subset of responses) | High | Critical — directly addresses meta-review |
| 2 | Fluency analysis (automatic perplexity proxy) | Medium | Addresses R2-C and meta-review |
| 3 | Clarify vocabulary weighting implementation (formula + pseudocode) | Low | Fixes Replicability score |
| 4 | Surface feature analysis table (sentence length, polysyllabic %, OOV %) | Medium | Addresses R1-F, makes mechanism transparent |
| 5 | Add target values to all results tables | Low | Addresses R1-D, quick win |
| 6 | Add prompting-only per-model column to Table 4 | Low | Directly requested by R3-B |
| 7 | Rewrite abstract | Low | Addresses R1-A |
| 8 | Discuss metric reliability on short texts | Low | Addresses R1-E |
| 9 | Explain prompt selection criteria | Low | Addresses R2-A |
| 10 | Faithfulness evaluation or explicit Limitations note | Medium | Addresses R3-D |
| 11 | Fix opening example or contextualize semantic drift | Low | Addresses R1-B |
| 12 | Clarify DExperts/FUDGE explanation | Low | Addresses R3-F |
| 13 | Standardize audience description ("young" vs. A1) | Low | Addresses R3-H |
| 14 | Add metric overlap table | Low | Addresses R1-C |
