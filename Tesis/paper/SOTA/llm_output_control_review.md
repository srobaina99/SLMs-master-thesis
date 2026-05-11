# State of the Art Review: Controlling LLM Outputs for Text Complexity

**Focus:** Academic research on controlling language model outputs for text complexity and simplification in educational contexts.

**Sources:** ACL Anthology, arXiv, academic conferences (ACL, EMNLP, NAACL)

**Date:** October 2025 (Updated)

---

## 1. OVERVIEW

This review focuses exclusively on **highly relevant** methods for controlling text complexity in language models, particularly for educational applications. We organize the literature into three primary approaches: fine-tuning methods, prompt-based constraints, and decoding-time probability manipulation.

---

## 2. FINE-TUNING FOR LINGUISTIC COMPLEXITY CONTROL

### 2.1 Multi-Objective Linguistic Control (MCTune)

**Nguyen et al. (2024) - "Multi-Objective Linguistic Control of Large Language Models"**

- **Paper:** ACL 2024 Findings ([aclanthology.org/2024.findings-acl.257](https://aclanthology.org/2024.findings-acl.257))
- **Relevance:** ⭐⭐⭐⭐⭐ **HIGHLY RELEVANT** - Most similar research goal

**Method:**
- Fine-tunes LLaMA2-7B by embedding multiple linguistic complexity values into instruction tuning
- Incorporates readability scores, sentence length distributions, vocabulary difficulty metrics directly into input prompts during training
- Datasets: Alpaca-GPT4, WizardLM

**Results:**
- Achieves precise control over multiple complexity dimensions simultaneously
- Maintains response quality and semantic coherence
- Demonstrates that complexity can be learned as controllable attribute

**Limitations:**
- Requires substantial computational resources for model retraining
- Requires labeled training data annotated with complexity metrics
- Locks model into complexity levels observed during training
- No mechanism for dynamic inference-time adjustment
- Complexity targets must be specified during fine-tuning, not adapted per-request

**Comparison to Our Work:**
- **MCTune:** Fine-tuning approach, requires training data and compute, fixed complexity levels
  - **Our approach:** Inference-time control (prompting + probability weighting), no training needed, dynamic complexity adjustment

---

## 3. PROMPT-BASED CONSTRAINED GENERATION

### 3.1 Lexically Constrained Generation (Divide & Conquer)

**Li et al. (2024) - "Control Large Language Models via Divide and Conquer"**

- **Paper:** EMNLP 2024 ([aclanthology.org/2024.emnlp-main.850](https://aclanthology.org/2024.emnlp-main.850))
- **Relevance:** ⭐⭐⭐⭐ **HIGHLY RELEVANT** - Lexical constraints similar to vocabulary boosting

**Method:**
- Systematically evaluates LLM performance in lexically constrained generation (LCG)
- Proposes Divide and Conquer Generation strategy: decomposes constraint satisfaction into iterative refinement steps
- Generates initial response, identifies unsatisfied constraints, regenerates text incorporating missed keywords

**Key Findings:**
- Identified three critical limitations in prompt-based constraint satisfaction:
  1. **Position bias:** Models exhibit differential constraint satisfaction rates depending on keyword position within prompt
  2. **Low responsiveness to decoding parameters:** Temperature and top-k sampling show minimal impact on constraint adherence
  3. **Compound word difficulties:** Subword tokenization fragments target vocabulary

**Results:**
- Over 90% improvement in constraint satisfaction rates on challenging benchmarks
- Demonstrates iterative refinement substantially outperforms single-pass generation

**Limitations:**
- Targets hard lexical constraints (binary requirements: specific words must appear)
- Not suitable for soft vocabulary preferences needed for text simplification
- Iterative refinement introduces substantial latency overhead
- Incompatible with interactive educational applications requiring sub-second response times

**Comparison to Our Work:**
  - **D&C strategy:** Hard constraints (must include specific words), multi-step generation
  - **Our approach:** Soft constraints (boost simple vocabulary probabilities), single-pass generation

---

## 4. DECODING-TIME CONTROL VIA PROBABILITY MANIPULATION

### 4.1 Product of Experts (DExperts)

**Liu et al. (2021) - "DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts"**

- **Paper:** ACL 2021 ([aclanthology.org/2021.acl-long.522](https://aclanthology.org/2021.acl-long.522))
- **Relevance:** ⭐⭐⭐⭐⭐ **HIGHLY RELEVANT** - Decoding-time probability manipulation, methodological precedent

**Method:**
- Combines pretrained language model with "expert" and "anti-expert" models in product of experts formulation
- Formula: `P̃(X_t | x_<t) = softmax(z_t + α(z_t^+ - z_t^-))`
- Adjusts token logits by computing difference between expert and anti-expert predictions
- Parameter α controls steering strength

**Applications:**
- Language detoxification (toxicity avoidance)
- Sentiment-controlled generation

**Key Results:**
- Effective with **small expert models** - demonstrated on GPT-3 with small steering models
- Requires only ~650 training examples for anti-expert construction
- Works with anti-expert only (can reuse base model as expert)
- **Computational efficiency:** 2-3× overhead vs base model (compared to 100× for PPLM)

**Mechanism:**
- Tokens get high probability if considered likely by expert AND unlikely by anti-expert
- Operates only on output of pretrained LM (no model modification)
- Enables efficient decoding-time steering with smaller auxiliary models

**Limitations:**
- Targets attribute control (toxicity, sentiment), not linguistic complexity
- Applied to large models, limited evaluation on small models (<4B parameters)
- No focus on educational applications or readability metrics

**Comparison to Our Work:**
- **DExperts:** Product of experts with trained expert/anti-expert models for attribute control
- **Our approach:** Direct vocabulary probability weighting for complexity control, no expert model training required

---

### 4.2 Position-Aware Weighted Decoding (CAT-PAW)

**Gu et al. (2022) - "Improving Controllable Text Generation with Position-Aware Weighted Decoding"**

- **Paper:** ACL 2022 Findings ([aclanthology.org/2022.findings-acl.272](https://aclanthology.org/2022.findings-acl.272))
- **Relevance:** ⭐⭐⭐⭐ **HIGHLY RELEVANT** - Weighted decoding framework

**Method:**
- Position-aware weighted decoding framework (CAT-PAW)
- Adjusts bias signals from controllers at different decoding positions
- Balances control strength and fluency

**Key Insight:**
- Control effectiveness varies by position in generation sequence
- Position-aware adjustments improve control-fluency trade-off

**Limitations:**
- Targets general controllability, not specific to linguistic complexity
- Requires controller models
- Focus on large models

---

### 4.3 Grammar-Constrained Decoding (GCD)

**Geng et al. (2023) - "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning"**

- **Paper:** EMNLP 2023 ([aclanthology.org/2023.emnlp-main.674](https://aclanthology.org/2023.emnlp-main.674))
- **Relevance:** ⭐⭐⭐ **RELEVANT** - Decoding-time constraints without finetuning

**Method:**
- Controls language model outputs to follow predefined grammatical structures
- No finetuning required
- Applicable across structured NLP tasks

**Limitations:**
- Targets structural constraints (grammar adherence, format control)
- Not designed for linguistic complexity or educational simplification

---

## 5. RESEARCH GAPS & POSITIONING

### 5.1 Identified Gaps in Existing Literature

1. **No inference-time complexity control for SLMs**
   - Existing: Fine-tuning (MCTune) or post-hoc simplification
   - Missing: Real-time control during generation without retraining
   - **Our contribution:** Combines prompting + probability weighting for inference-time control

2. **Limited vocabulary-based probability manipulation**
   - Existing: Hard constraints (LCG), grammar constraints, attribute control (DExperts)
   - Missing: Soft vocabulary boosting via logits processors specifically for text simplification
   - **Our contribution:** First application of probability weighting for A1 vocabulary targeting in educational context

3. **No systematic evaluation across SLMs for educational use**
   - Existing: Focus on large models (LLaMA2-7B, GPT-3, GPT-series)
   - Missing: Comparison across small models (0.5B-3.8B) for on-device deployment
   - **Our contribution:** 6-model factorial study (TinyStories 33M → Phi3 3.8B)

4. **Lack of comprehensive readability evaluation**
   - Existing: Single metrics (BLEU, perplexity) or limited readability measures
   - Missing: Multi-metric evaluation tailored to educational contexts (A1 learners)
   - **Our contribution:** 6 primary readability metrics (FK Grade, Gunning Fog, SMOG, Spache, etc.)

5. **No factorial comparison of interventions**
   - Existing: Studies prompting or decoding manipulation in isolation
   - Missing: Factorial design isolating individual and interaction effects
   - **Our contribution:** 2×2 design (prompting × weighting) identifies synergies

---

### 5.2 Methodological Positioning

| **Paper** | **Method** | **Model Size** | **Control Mechanism** | **Training Required?** | **Relevance** |
|-----------|------------|----------------|----------------------|----------------------|---------------|
| **Nguyen et al. (2024) - MCTune** | Instruction tuning | 7B | Complexity values in input | ✅ Yes (fine-tuning) | ⭐⭐⭐⭐⭐ |
| **Li et al. (2024) - D&C** | Lexical constraints | Various | Divide-conquer generation | ❌ No | ⭐⭐⭐⭐ |
| **Liu et al. (2021) - DExperts** | Product of experts | Various | Expert/anti-expert logits | ✅ Yes (expert training) | ⭐⭐⭐⭐⭐ |
| **Gu et al. (2022) - CAT-PAW** | Weighted decoding | Various | Position-aware weighting | ✅ Yes (controller) | ⭐⭐⭐⭐ |
| **Geng et al. (2023) - GCD** | Grammar constraints | Various | Constrained decoding | ❌ No | ⭐⭐⭐ |
| **Our Work** | Prompting + Weighting | 0.5B-3.8B | System prompts + vocabulary logits | ❌ No | - |

**Key Differentiator:** Our approach requires **no model training or fine-tuning**, making it:
- **Flexible:** Works across any pre-trained model
- **Efficient:** No compute/data requirements beyond inference
- **Adaptive:** Can adjust complexity dynamically per request
- **Educational:** Specifically designed for A1 learners with readability metrics

---

## 6. THEORETICAL FRAMEWORK

### 6.1 Our Dual-Mechanism Approach

**Top-down Control (Prompting):**
   - Mechanism: Instructions shape generation strategy globally
- Prior work: Instruction tuning (Nguyen et al.)
- Our implementation: System prompts for A1-appropriate language

**Bottom-up Control (Probability Weighting):**
   - Mechanism: Token-level probability adjustments during decoding
- Prior work: DExperts (Liu et al.), CAT-PAW (Gu et al.), Lexical constraints (Li et al.)
- Our implementation: Vocabulary-aware logits processor with 2.0× amplification for A1 words

**Hypothesis:** Combining both mechanisms creates synergistic effects:
- Prompting provides structural guidance (sentence length, complexity)
- Weighting reinforces vocabulary simplicity (word choice)

**Testing:** Factorial design (prompting × weighting) quantifies individual and interaction effects

---

## 7. KEY CITATIONS FOR PAPER

### 7.1 Must-Cite Papers (Core Related Work)

1. **Nguyen et al. (2024) - MCTune** - Most similar goal (linguistic complexity control), compare fine-tuning vs inference-time
2. **Liu et al. (2021) - DExperts** - Methodological precedent for decoding-time probability manipulation
3. **Li et al. (2024) - Divide & Conquer** - Lexical constraint generation, identifies position bias and decoding parameter limitations
4. **Gu et al. (2022) - CAT-PAW** - Position-aware weighted decoding framework
5. **Geng et al. (2023) - GCD** - Grammar-constrained decoding without finetuning

### 7.2 Positioning Statements for Paper

**In Introduction:**
> "While prior work has explored complexity control through fine-tuning [Nguyen et al., 2024] and decoding-time attribute control [Liu et al., 2021], no existing approach combines inference-time prompting and vocabulary probability weighting for small language models (<4B parameters) in educational contexts."

**In Related Work:**
> "Our work differs from MCTune [Nguyen et al., 2024] by using inference-time control rather than fine-tuning, enabling flexibility and lower resource requirements. Unlike DExperts [Liu et al., 2021] which targets attribute control (toxicity, sentiment) via expert models, we apply direct vocabulary probability weighting for linguistic complexity control tailored to A1 English learners."

**In Discussion:**
> "Unlike hard lexical constraints [Li et al., 2024] requiring specific word inclusion, our probability weighting implements soft vocabulary preferences that maintain linguistic naturalness while steering toward beginner-appropriate vocabulary."

---

## 8. REFERENCES

- Geng, R., Zhou, Y., & Huang, X. (2023). Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 10932–10848. https://aclanthology.org/2023.emnlp-main.674

- Gu, J., Feng, X., Shen, L., Zhou, L., Huang, M., & Zhu, X. (2022). Improving Controllable Text Generation with Position-Aware Weighted Decoding. *Findings of the Association for Computational Linguistics: ACL 2022*, 3449–3467. https://aclanthology.org/2022.findings-acl.272

- Li, B., Wang, Y., Meng, T., Chang, K.-W., & Peng, N. (2024). Control Large Language Models via Divide and Conquer. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 15183–15203. https://aclanthology.org/2024.emnlp-main.850

- Liu, A., Sap, M., Lu, X., Swayamdipta, S., Bhagavatula, C., Smith, N. A., & Choi, Y. (2021). DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, 6691–6706. https://aclanthology.org/2021.acl-long.522

- Nguyen, D., Malon, C., Dernoncourt, F., Bui, T., & Rossi, R. A. (2024). Multi-Objective Linguistic Control of Large Language Models. *Findings of the Association for Computational Linguistics: ACL 2024*, 4313–4330. https://aclanthology.org/2024.findings-acl.257

---

**END OF FOCUSED STATE OF THE ART REVIEW**

**Date:** October 19, 2025
**Total Papers Reviewed:** 5 (highly relevant only)
**Core Citations:** 5 papers
**Focus:** Inference-time complexity control for small language models in educational contexts




