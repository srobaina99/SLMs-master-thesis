# 3. Methods

## 3.1 Experimental Design

A factorial experiment was conducted to systematically evaluate the effectiveness of two inference-time interventions for controlling text complexity in Small Language Models. The experimental design was the following:

- **Models (4):** Phi-3-mini (3.8B parameters), Qwen2.5 (0.5B), Qwen3 (0.6B), and TinyLlama (1.1B)
- **System prompt**:

  ```
  You are a helpful English teacher for beginner students. 
  Answer with a paragraph only with plain text.
  ```
- **Intervention Conditions (4):**

  1. **Control:** No interventions applied
  2. **Weighting Only:** Probability boosting via vocabulary list
  3. **Prompting Only:** Contextual instructions for simplification
  4. **Both:** Combined weighting and prompting
- **Test Prompts (15):** Diverse English learning questions covering vocabulary definitions, self-introduction, basic concepts, comparisons, descriptions, and daily activities

This 4 × 4 × 15 factorial design spans 240 total observations, enabling systematic analysis of main effects (weighting, prompting), interaction effects (weighting × prompting), and model-specific differences.

### 3.1.1 Model Selection Rationale

The selected models span a parameter range from 0.5B to 3.8B, representing the current landscape of deployable Small Language Models suitable for on-device inference in resource-constrained educational contexts. All models were evaluated using quantized 4-bit GGUF format implementations, ensuring consistent evaluation conditions and realistic deployment performance characteristics. Model selection prioritized:

1. **Parameter Efficiency:** All models under 4B parameters for on-device deployment feasibility
2. **Architectural Diversity:** Different model families (Qwen, Llama, Phi architectures) to assess generalizability
3. **Instruction-Following Capability:** All models fine-tuned for instruction-following tasks
4. **Open Availability:** Publicly accessible models enabling reproducibility

### 3.1.3 Prompt Design

The 15 test prompts were designed to represent authentic A1-level English learning scenarios, covering:

- **Vocabulary Definitions** (5 prompts): "What does the word 'library' mean?", "Can you explain what 'breakfast' is?"
- **Conceptual Explanations** (4 prompts): "What is a dog?", "What does 'happy' mean?"
- **Comparative Questions** (2 prompts): "What is the difference between 'big' and 'large'?"
- **Descriptive Tasks** (4 prompts): "Can you describe what happens in the morning?", "What colors do you see in a rainbow?"

All prompts were designed to elicit paragraph-length responses (30-100 words) suitable for evaluating readability metrics.

---

## 3.2 Intervention Methodologies

### 3.2.1 Probability Weighting Intervention

The probability weighting intervention implements soft vocabulary constraints through real-time manipulation of token generation probabilities during the decoding process. The mechanism operates as follows:

**Vocabulary Selection:** A target vocabulary was compiled from the Cambridge English A1 "Starters word list", comprising 494 word entries including basic words (e.g., "happy", "dog", "school"). This vocabulary represents the expected lexical knowledge of A1-level English learners. Special tokens, like end tokens, were included in the list to avoid long answers due to this weighting.

**Token-Level Weighting:** Each word in the vocabulary is tokenized using the model's tokenizer, producing one or more subword tokens depending on the model's vocabulary. The weighting intervention operates at the token level: bias values are applied to the token IDs resulting from this tokenization process, not to whole words directly. For example, the word "afternoon" may tokenize to ["after", "noon"], and both resulting token IDs receive the weight bias.

**Logit Bias Mechanism:** During each decoding step, the model's output logits are modified before sampling. For each token ID derived from the target vocabulary, a multiplicative weight factor of 1.5× is applied to its corresponding logit value. This increases the probability that target vocabulary tokens will be selected during generation, creating a soft bias toward simpler, A1-appropriate words.

**Implementation:** The weighting is applied at inference time through the inference framework's logit bias functionality, requiring no model retraining or fine-tuning. The tokenization process is performed once during initialization, creating a mapping from vocabulary words to their constituent token IDs, which is then applied consistently throughout generation.

**Computational Overhead:** The weighting intervention introduces minimal latency overhead (~5-10% increase in generation time) as the logit modification is a vectorized operation applied once per generated token.

### 3.2.2 Weight Factor Exploration

To select the weight factor, a supplementary experiment tested three multiplicative factors: **1.5×, 2.0×, and 4.0×**.

**Experimental Design:** The weight factor exploration tested each factor value in the weighting-only intervention condition (no prompting). This enabled a direct comparison of how weight magnitude affects text complexity without confounding effects from contextual prompting. The four models were evaluated using 15 prompts from the standard prompt set, with a total of 180 observations (60 per weight factor). A timeout of 5 minutes was set to prevent excessive generation times.

**Results:** Analysis of the 172 valid responses (excluding timeouts and empty outputs) revealed distinct patterns across weight factors. The 1.5× factor produced the lowest mean FK Grade (8.76), highest stability (standard deviation: 3.23), and most concise outputs (84.8 words). The 2.0× factor achieved a mean FK Grade of 9.74 with 90.4 words per response, showing weaker complexity reduction than 1.5×. The 4.0× factor paradoxically increased complexity (FK Grade: 13.85) and verbosity (102.0 words) with high variability (standard deviation: 13.04), suggesting that excessive vocabulary constraints triggered compensatory mechanisms producing longer sentences.

**Selected Weight Factor:** Based on this analysis, the **1.5× weight factor** was selected for the experiment. This factor achieved the lowest average complexity and most consistent performance across models (standard deviation: 3.23).

### 3.2.3 Context Prompting Intervention

The prompting intervention provides explicit instructions to guide the model toward generating simpler, more accessible text. The intervention consists of prepending a context instruction to each user query:

```
# Context
Please respond using simple words that a young non-English speaking student can understand. 
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.

[USER PROMPT]
```

**Design Rationale:** The context prompt establishes four explicit simplification constraints:

1. **Audience Specification:** "young non-English speaking student" activates instruction-following for complexity adjustment
2. **Vocabulary Constraint:** "simple words" and "basic English learning materials" guide lexical selection
3. **Sentence Structure:** "Keep sentences short and clear" targets syntactic simplification
4. **Grammar Constraint:** "Avoid complex grammar structures" prevents advanced constructions

**Implementation:** The context prompt is prepended to user queries in prompting and combined conditions only. Control and weighting-only conditions receive the user prompt directly after the system prompt (Section 3.1.2) without additional simplification instructions.

**Computational Overhead:** The prompting intervention introduces negligible latency overhead, as the additional context tokens (~40 tokens) represent a small fraction of total input length.

### 3.2.4 Combined Intervention

In the combined condition, both interventions are applied simultaneously: the system prompt establishes high-level simplification goals, while probability weighting reinforces vocabulary-level constraints during generation. This design tests whether the interventions exhibit synergistic effects or redundancy.

---

## 3.3 Evaluation Metrics

Text complexity evaluation for A1-level English learners requires metrics that capture multiple dimensions of linguistic difficulty. The selected evaluation framework employs six complementary metrics spanning sentence structure complexity, polysyllabic vocabulary density, and vocabulary difficulty calibrated to primary-grade reading levels.

The metric selection prioritizes the following criteria: (1) **educational relevance** to A1 proficiency targets, with established grade-level interpretations aligned to beginner learner capabilities; (2) **dimensional coverage**, ensuring that sentence structure, syllable complexity, and vocabulary difficulty are each independently assessed. This multi-metric approach enables cross-validation of intervention effects—consistent improvements across multiple independent measures provide stronger evidence of genuine complexity reduction than single-metric evaluations.

### 3.3.1 Primary Readability Metrics

The four primary readability formulas provide comprehensive coverage of different complexity dimensions:

**1. Flesch-Kincaid Grade Level**

The Flesch-Kincaid Grade Level formula \cite{kincaid1975derivation} is defined as:

$$
\text{FK}_{\text{Grade}} = 0.39 \cdot \text{ASL} + 11.8 \cdot \text{ASW} - 15.59
$$

where $\text{ASL}$ denotes average sentence length (words per sentence) and $\text{ASW}$ denotes average syllables per word. This metric provides a U.S. grade-level estimate and is the most widely adopted readability measure in educational contexts. Target for A1 learners: $\text{FK}_{\text{Grade}} \leq 5.0$ (elementary level).

**2. Gunning Fog Index**

The Gunning Fog Index \cite{gunning1952technique} is computed as:

$$
\text{FOG} = 0.4 \cdot \left(\frac{W}{S} + 100 \cdot \frac{C}{W}\right)
$$

where $W$ is the total word count, $S$ is the number of sentences, and $C$ is the count of complex words (words with $\geq 3$ syllables, excluding proper nouns). This metric is highly sensitive to polysyllabic vocabulary, making it particularly suitable for evaluating vocabulary-based interventions. Target: $\text{FOG} \leq 6.0$ (sixth grade or below).

**3. SMOG Index**

The SMOG (Simple Measure of Gobbledygook) index \cite{mclaughlin1969smog} is defined as:

$$
\text{SMOG} = 3 + \sqrt{P}
$$

where $P$ is the count of polysyllabic words (words with $\geq 3$ syllables). This metric is specifically designed for short texts and exhibits lower sensitivity to sentence length variations compared to Flesch-Kincaid. Target: $\text{SMOG} \leq 7.0$ (junior high or below).

**4. Spache Readability Formula**

The Spache Readability Score \cite{spache1953new} is calculated as:

$$
\text{Spache} = 0.141 \cdot \text{ASL} + 0.086 \cdot \text{PDW} + 0.839
$$

where $\text{ASL}$ is average sentence length and $\text{PDW}$ is the percentage of difficult words (words not on the Spache word list of approximately 1,000 common words for grades 1-4). This metric is specifically calibrated for primary-grade texts, providing superior discrimination at the A1 proficiency level. Target: $\text{Spache} \leq 4.0$ (primary grades).

### 3.3.2 Secondary Descriptive Metrics

**5. Word Count:** Total words per response, denoted as $|W|$, indicating conciseness and cognitive load. Target range: $30 \leq |W| \leq 60$ for manageable beginner-level responses.

**6. Difficult Words Count:** Number of words classified as "difficult" according to the Dale-Chall criterion \cite{dale1948formula} (words not in the 2,940-word Dale-Chall easy word list AND having $\geq 3$ syllables), denoted as $D$. This provides a directly interpretable count of vocabulary accessibility issues.

### 3.3.3 Complementary Coverage

The six metrics provide complementary information across complexity dimensions:

- **Sentence Structure:** Flesch-Kincaid and SMOG capture sentence length effects through $\text{ASL}$, assessing syntactic complexity
- **Polysyllabic Complexity:** Gunning Fog and SMOG emphasize complex word identification through syllable counting, targeting morphological complexity
- **Vocabulary Difficulty:** Spache uses a word list specifically calibrated for primary grades (1-4), matching A1 proficiency expectations
- **Direct Interpretability:** Word count ($|W|$) and difficult words count ($D$) provide immediately actionable metrics for practitioners
- **Cross-Validation:** Consistent improvement across multiple independent metrics indicates genuine intervention effectiveness rather than metric-specific artifacts

---

## 3.4 Experimental Procedure

### 3.4.1 Generation Parameters

All models were evaluated with consistent generation parameters to ensure comparability:

- **Temperature:** 0.7 (balanced between determinism and diversity)
- **Top-k sampling:** 50
- **Top-p (nucleus) sampling:** 0.95
- **Maximum tokens:** 200 (~150 words maximum response length)
- **Quantization:** 4-bit GGUF format for all models

### 3.4.2 Execution Protocol

For each of the 240 experimental conditions (4 models × 4 intervention conditions × 15 prompts):

1. **Model Loading:** Load the quantized model into memory
2. **Prompt Construction:** Construct the full prompt with appropriate system instructions based on intervention condition
3. **Generation:** Generate response with specified parameters and intervention(s) applied
4. **Metric Computation:** Calculate all six readability metrics on the generated text
5. **Latency Recording:** Record total generation time (excluding model loading)

To minimize memory overhead, experiments were grouped by model: all 60 conditions for a single model (4 configs × 15 prompts) were executed sequentially before unloading the model and proceeding to the next.

### 3.4.3 Computational Environment

**All experiments were conducted on a single machine to ensure consistent performance measurements:**

- **Hardware: MacBook Pro M2 (2022), 8-core CPU, 8GB unified memory**
- **Operating System: macOS 15.7 (Sequoia)**
- **Software: llama-cpp-python (≥0.2.0) for GGUF model inference, Python 3.12**
- **Readability Computation: textstat (≥0.7.3) for standard readability metrics**
- **Acceleration: Metal Performance Shaders (MPS) backend enabled for Phi3 (full GPU offload with `n_gpu_layers=-1`); Qwen2, Qwen3, and TinyLlama ran on CPU (`n_gpu_layers=0`)**

---

## 3.5 Limitations and Validity Considerations

### 3.5.1 Sample Size

The current sample of 15 prompts per model provides preliminary evidence but falls short of the 50+ prompts recommended for robust conclusions in per-model analyses. While the total sample (240 observations) enables observation of general trends, per-model patterns should be interpreted cautiously.

### 3.5.2 Prompt Representativeness

The test prompts focus primarily on vocabulary and conceptual explanations. Generalization to other A1-level interaction types (conversation, error correction, cultural topics) requires additional validation.

### 3.5.3 Metric Validity

Readability formulas were originally developed for native speaker populations. While these metrics demonstrate documented correlation with comprehension in L2 contexts, direct validation with A1 English learners remains necessary to establish ecological validity.

### 3.5.4 Hyperparameter Selection

The weight factor (2.0×) was selected based on preliminary exploration rather than systematic optimization. The optimal value may vary by model architecture and target vocabulary size.

### 3.5.5 Semantic Correctness

The evaluation framework measures text complexity exclusively. Semantic correctness, factual accuracy, and response completeness receive no quantitative assessment, representing a significant methodological limitation. Simpler text may sacrifice important information or introduce factual errors.

---

## References

```bibtex
@article{dale1948formula,
  title={A formula for predicting readability: Instructions},
  author={Dale, Edgar and Chall, Jeanne S.},
  journal={Educational Research Bulletin},
  volume={27},
  number={2},
  pages={37--54},
  year={1948}
}

@book{gunning1952technique,
  title={The Technique of Clear Writing},
  author={Gunning, Robert},
  year={1952},
  publisher={McGraw-Hill}
}

@techreport{kincaid1975derivation,
  title={Derivation of new readability formulas (Automated Readability Index, Fog Count and Flesch Reading Ease Formula) for Navy enlisted personnel},
  author={Kincaid, J. Peter and Fishburne, Robert P. and Rogers, Richard L. and Chissom, Brad S.},
  year={1975},
  institution={Naval Technical Training Command},
  number={Research Branch Report 8-75},
  address={Millington, TN}
}

@article{mclaughlin1969smog,
  title={SMOG grading: A new readability formula},
  author={McLaughlin, G. Harry},
  journal={Journal of Reading},
  volume={12},
  number={8},
  pages={639--646},
  year={1969}
}

@article{spache1953new,
  title={A new readability formula for primary-grade reading materials},
  author={Spache, George},
  journal={The Elementary School Journal},
  volume={53},
  number={7},
  pages={410--413},
  year={1953}
}
```
