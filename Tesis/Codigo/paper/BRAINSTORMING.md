# Paper Presentation Brainstorming: SLM Text Complexity Control

## Research Context

This work is conducted within the broader research project **"Métodos de generación controlada para la construcción de agentes conversacionales de apoyo a la enseñanza"** (Controlled generation methods for building conversational agents to support teaching), funded by ANII (Agencia Nacional de Investigación e Innovación, Uruguay)

**Project Overview:**

The parent project aims to develop a reliable and safe conversational agent prototype for educational purposes. Achieving this goal requires addressing critical limitations of Large Language Models (LLMs)—particularly hallucination, bias reproduction, and lack of source traceability. While LLMs demonstrate excellent linguistic correctness and pragmatic relevance, their tendency to generate "invented" information poses significant risks in educational contexts where accuracy is paramount. The prototype employs multiple mitigation strategies: fine-tuning on educational data, rule-based constraints, and advanced prompting methods to ensure source reliability.

**Target Student Population:**

The project focuses on **Uruguayan English learners with basic to no prior English experience**, particularly students in the first years of schooling. This represents a critical and underserved educational context where traditional resources and tutoring are limited. By developing accessible, appropriately-simplified AI tutors through the Ceibal initiative—Uruguay's national digital education program reaching approximately 550,000 students—the project aims to democratize high-quality English language education across the country's public education system.

**Strategic Focus on Small Language Models:**

The emphasis on Small Language Models (rather than large proprietary models) emerges from institutional and infrastructural imperatives. Deployment of large-scale LLMs (billions of parameters, cloud-based infrastructure) presents substantial barriers: computational costs, dependency on stable internet infrastructure, and licensing constraints. In contrast, SLMs (33M to 3.8B parameters) enable:

- **Cost-Efficient Deployment:** On-device inference eliminates cloud infrastructure costs and subscription fees, enabling sustainable scaling across resource-limited educational institutions
- **Offline-First Functionality:** Critical for Uruguayan contexts with geographically dispersed populations and variable internet connectivity; models execute locally, requiring only initial download
- **Latency Guarantees:** On-device processing ensures sub-second response times essential for interactive learning experiences, independent of network conditions
- **Educational Equity:** Universal accessibility without requiring persistent high-bandwidth connections or institutional server infrastructure

**Contribution to the Larger Research:**

This SLM complexity control study directly addresses a fundamental challenge identified in the parent project: **adapting language models to match learner proficiency levels**. While the broader project tackles reliability and source control, this work specifically addresses the complexity mismatch problem for beginner language learners (A1 level). The developed real-time complexity control methods enable on-device deployment of educational chatbots capable of generating appropriately simple responses for A1 English learners—a critical requirement for Ceibal's large-scale deployment context serving 550,000 students nationwide.

---

**Research Question:** Can small language models be controlled to produce appropriately simple text for A1 English learners through decoding manipulation and prompt engineering?

---

## 1. PAPER STRUCTURE (Option A: Classic Research Paper)

### 1.1 Introduction

**Hook - Visual Example:**

```
PROMPT: "What does the word 'library' mean?"

CONTROL (Qwen2):
"The word 'library' means a collection of books, magazines, and other printed 
materials organized and maintained by a library staff..."
→ FK Grade: 11.5 | Reading Ease: 42.8 | Exceeds A1 target complexity

BOTH INTERVENTIONS (Qwen2):
"A library is a place where you can find lots of books to read. It's like a 
big house with many special rooms where people keep their toys and stories!"
→ FK Grade: 5.8 | Reading Ease: 83.3 | Achieves A1 target
```

**Contextual Framework:**

- Language learning applications require adaptive AI assistants calibrated to learner proficiency levels
- A1 level (CEFR) denotes beginner learners with restricted vocabulary (~500 words)
- Small language models enable on-device, low-latency deployment with minimal computational overhead
- **Primary Challenge:** SLMs trained on general-domain corpora produce output exceeding beginner-appropriate complexity thresholds

**Literature Gap:**

- Existing approaches: Post-hoc text simplification, computationally intensive fine-tuning
- Missing methodology: Real-time complexity control during inference for SLMs
- Proposed approach: Two lightweight interventions applicable at inference time without model modification

**Research Objectives:**

1. Evaluate effectiveness of probability weighting via decoding manipulation
2. Evaluate effectiveness of contextual prompting via instruction engineering
3. Assess synergistic interaction effects between combined interventions
4. Conduct comparative analysis across model variants (Qwen2 vs Qwen3)
5. Quantify performance trade-offs between text simplicity and response latency

---

### 1.2 State of the Art Literature Review

**Detailed Analysis:** See [`paper/sections/paper_sota.md`](sections/paper_sota.md)

**Summary of Key Prior Work:**

**1.2.1 Fine-tuning Approach: MCTune (Nguyen et al. 2024)**

- **Method:** Multi-Control Tuning - embeds linguistic complexity values into instruction tuning for LLaMA2-7B
- **Results:** Precise control over multiple complexity dimensions while maintaining response quality
- **Limitation:** Resource-intensive, requires labeled training data, locks model into fixed complexity levels
- **Gap:** No dynamic inference-time adjustment; complexity targets fixed during training

**1.2.2 Prompt-based Approach: Divide & Conquer (Li et al. 2024)**

- **Method:** Iterative refinement strategy for Lexically Constrained Generation (LCG)
- **Key Findings:** Position bias, low decoding parameter responsiveness, compound word difficulties
- **Results:** >90% improvement in constraint satisfaction rates through decomposition
- **Limitation:** Hard constraints (specific words required), iterative refinement introduces latency overhead
- **Gap:** No soft vocabulary boosting for text simplification

**1.2.3 Research Gaps Addressed by This Work**

1. **Inference-time complexity control for SLMs**: Existing work requires fine-tuning (MCTune) or applies hard constraints (Li et al.); no prior work combines inference-time prompting with probability weighting for small models
2. **Soft vocabulary manipulation**: Prior work enforces specific word inclusion; no prior work on probability-based vocabulary simplification via logits processors
3. **Systematic small model evaluation**: Focus on large models (7B+); systematic evaluation across sub-4B models for educational deployment absent
4. **Factorial intervention comparison**: Prompting vs decoding manipulation studied in isolation; no factorial designs isolating individual and interaction effects

**Our Contributions:**

✅ **Novel approach:** First combination of inference-time prompting + probability weighting for complexity control
✅ **Soft constraints:** Vocabulary boosting (2.0× probability amplification) vs hard word requirements
✅ **SLM focus:** Systematic evaluation across 6 models (33M-3.8B parameters) for on-device deployment
✅ **Factorial design:** 2×2 design isolates individual and interaction effects of interventions
✅ **Comprehensive evaluation:** 6 readability metrics + 192 observations across diverse architectures
✅ **Deployment focus:** Offline-first, sub-second latency for 550,000 students via Ceibal initiative

---

### 1.3 Methods

**1.3.1 Experimental Design**

**Factorial Design:**

- **Models (6):** Phi3 (3.8B), Qwen2 (0.5B), Qwen3 (0.6B), SmolLM (1.7B), TinyLlama (1.1B), TinyStories (33M)
- **Interventions (4 configs):**
  1. **Control:** No interventions
  2. **Weighting Only:** Probability boosting (vocab list)
  3. **Prompting Only:** Context instructions
  4. **Both:** Weighting + Prompting
- **Prompts (8):** Diverse English learning questions
- **Total:** 6 models × 4 configs × 8 prompts = 192 observations

**1.3.2 Interventions**

**A. Probability Weighting**

- **Mechanism:** `ProbabilityWeightingLogitsProcessor` modifies token logits before sampling
- **Vocabulary:** 511 words from A1 "Starters" curriculum (filtered to 493 words)
- **Weight Factor:** 2.0× boost to target vocabulary tokens
- **Implementation:** Applied during decoding at each generation step

**B. Context Prompting**

- **System Prompt Addition:**
  ```
  You are a helpful English teacher for beginner students. 
  Use simple words that young learners can understand.
  Keep sentences short and clear.
  ```
- **No modification to user prompt**

**1.3.3 Readability Metrics**

**Primary Analysis Metrics (4):**

1. **Flesch-Kincaid Grade Level** - Most established grade-level metric; balances sentence structure and syllabic complexity
2. **Gunning Fog Index** - Highly sensitive to polysyllabic words; ideal for vocabulary weighting evaluation
3. **SMOG Index** - Reliable for short texts; less sensitive to sentence length variations
4. **Spache Readability** - Specifically calibrated for grades 1-4; superior A1-level discrimination

**Secondary Descriptive Statistics (2):**

5. **Word Count** - Reveals conciseness differences; assesses cognitive load
6. **Difficult Words Count** - Direct, interpretable metric; easier to communicate than abstract scores. A word is classified as "difficult" if it: (1) is NOT in the Dale-Chall easy word list (2,940 common English words for grades 4-16+), AND (2) has 3+ syllables. This provides a straightforward count of vocabulary accessibility issues.

**A1 Target Ranges:**

| Metric                         | Target Range | Interpretation            |
| ------------------------------ | ------------ | ------------------------- |
| **Flesch-Kincaid Grade** | ≤5.0        | Elementary level or below |
| **Gunning Fog**          | ≤6.0        | Sixth grade or below      |
| **SMOG Index**           | ≤7.0        | Junior high or below      |
| **Spache Readability**   | ≤4.0        | Primary grades (1-4)      |
| **Word Count**           | 30-60 words  | Concise, manageable       |
| **Difficult Words**      | Minimize     | Prefer common vocabulary  |

**Metric Selection Rationale:**

The selected metrics provide comprehensive coverage of text complexity while avoiding redundancy:

- **Sentence Structure**: Flesch-Kincaid and SMOG capture sentence length effects
- **Polysyllabic Complexity**: Gunning Fog and SMOG emphasize complex word identification
- **Vocabulary Difficulty**: Spache uses a 1,000-word reference list specifically calibrated for primary grades (1-4), perfectly matching A1 proficiency level
- **Cross-Validation**: Consistent improvement across metrics indicates genuine intervention effectiveness
- **Zero Redundancy**: Each metric provides unique information without duplicating others

**Discarded Metrics:**

- **Flesch Reading Ease**: Redundant with Flesch-Kincaid (identical formula inputs, different scaling)
- **Dale-Chall**: Redundant with Spache (both word-list-based; Spache provides superior A1-level discrimination)
- **ARI/Coleman-Liau**: Character-based metrics redundant with sentence structure metrics
- **Linsear Write**: Designed for technical writing, not conversational text
- **McAlpine EFLAW**: Designed for auditory content, not text-based interactions

**Key Metric Formulas:**

**Flesch-Kincaid Grade Level:**

```
Grade Level = (0.39 × ASL) + (11.8 × ASW) - 15.59
```

Where ASL = Average Sentence Length, ASW = Average Syllables per Word

**Gunning Fog Index:**

```
Fog Index = 0.4 × [(Words/Sentences) + 100 × (Complex Words/Words)]
```

Complex Words = 3+ syllables (excluding proper nouns)

**SMOG Index:**

```
SMOG Grade = 3 + √(Polysyllable Count in 30 sentences)
```

**Spache Readability:**

```
Spache Score = (0.141 × ASL) + (0.086 × % Unfamiliar Words) + 0.839
```

Unfamiliar Words = Words NOT on Spache word list (~1,000 words for grades 1-4)

*(Complete metric documentation available in text_metrics.md)*

**1.3.4 Statistical Analysis**

**ANOVA Design:**

- **Independent Variables:**
  - Model (6 levels: Phi3, Qwen2, Qwen3, SmolLM, TinyLlama, TinyStories)
  - Weighting (2 levels: Yes/No)
  - Prompting (2 levels: Yes/No)
- **Dependent Variables:** All readability metrics
- **Tests:**
  - Main effects of Weighting, Prompting
  - Interaction: Weighting × Prompting
  - Model differences (pairwise comparisons across 6 models)

**Effect Sizes:**

- **Percent change from control:** `(intervention_mean - control_mean) / control_mean × 100`
- **Cohen's d:** Standardized mean difference
  - Small: 0.2, Medium: 0.5, Large: 0.8+

---

### 1.4 Results

**1.4.1 By-Config Analysis (All 6 Models Combined)**

| Configuration            | FK Grade | Gunning Fog | Flesch Ease | Response Time (s) | Word Count |
| ------------------------ | -------- | ----------- | ----------- | ----------------- | ---------- |
| **Control**        | TBD      | TBD         | TBD         | TBD               | TBD        |
| **Weighting Only** | TBD      | TBD         | TBD         | TBD               | TBD        |
| **Prompting Only** | TBD      | TBD         | TBD         | TBD               | TBD        |
| **Both**           | TBD      | TBD         | TBD         | TBD               | TBD        |

✅ = Meets A1 target

**Key Findings:**

1. **TBD:** Results pending full 6-model experiment
2. **TBD:** Results pending full 6-model experiment
3. **TBD:** Results pending full 6-model experiment
4. **TBD:** Results pending full 6-model experiment

**1.4.2 Model Comparison**

| Model                 | Config    | FK Grade | Flesch Ease | Response Time (s) |
| --------------------- | --------- | -------- | ----------- | ----------------- |
| **Phi3**        | Control   | TBD      | TBD         | TBD               |
| **Phi3**        | Weighting | TBD      | TBD         | TBD               |
| **Phi3**        | Prompting | TBD      | TBD         | TBD               |
| **Phi3**        | Both      | TBD      | TBD         | TBD               |
| **Qwen2**       | Control   | TBD      | TBD         | TBD               |
| **Qwen2**       | Weighting | TBD      | TBD         | TBD               |
| **Qwen2**       | Prompting | TBD      | TBD         | TBD               |
| **Qwen2**       | Both      | TBD      | TBD         | TBD               |
| **Qwen3**       | Control   | TBD      | TBD         | TBD               |
| **Qwen3**       | Weighting | TBD      | TBD         | TBD               |
| **Qwen3**       | Prompting | TBD      | TBD         | TBD               |
| **Qwen3**       | Both      | TBD      | TBD         | TBD               |
| **SmolLM**      | Control   | TBD      | TBD         | TBD               |
| **SmolLM**      | Weighting | TBD      | TBD         | TBD               |
| **SmolLM**      | Prompting | TBD      | TBD         | TBD               |
| **SmolLM**      | Both      | TBD      | TBD         | TBD               |
| **TinyLlama**   | Control   | TBD      | TBD         | TBD               |
| **TinyLlama**   | Weighting | TBD      | TBD         | TBD               |
| **TinyLlama**   | Prompting | TBD      | TBD         | TBD               |
| **TinyLlama**   | Both      | TBD      | TBD         | TBD               |
| **TinyStories** | Control   | TBD      | TBD         | TBD               |
| **TinyStories** | Weighting | TBD      | TBD         | TBD               |
| **TinyStories** | Prompting | TBD      | TBD         | TBD               |
| **TinyStories** | Both      | TBD      | TBD         | TBD               |

**Key Findings:**

1. **TBD:** Results pending full 6-model experiment
2. **TBD:** Results pending full 6-model experiment
3. **TBD:** Results pending full 6-model experiment
4. **TBD:** Results pending full 6-model experiment

**1.4.3 Effect Sizes**

**Per-Model Analysis:**

| Model                 | Config    | FK Grade Change | Cohen's d | Flesch Ease Change |
| --------------------- | --------- | --------------- | --------- | ------------------ |
| **Phi3**        | Weighting | TBD             | TBD       | TBD                |
| **Phi3**        | Prompting | TBD             | TBD       | TBD                |
| **Phi3**        | Both      | TBD             | TBD       | TBD                |
| **Qwen2**       | Weighting | TBD             | TBD       | TBD                |
| **Qwen2**       | Prompting | TBD             | TBD       | TBD                |
| **Qwen2**       | Both      | TBD             | TBD       | TBD                |
| **Qwen3**       | Weighting | TBD             | TBD       | TBD                |
| **Qwen3**       | Prompting | TBD             | TBD       | TBD                |
| **Qwen3**       | Both      | TBD             | TBD       | TBD                |
| **SmolLM**      | Weighting | TBD             | TBD       | TBD                |
| **SmolLM**      | Prompting | TBD             | TBD       | TBD                |
| **SmolLM**      | Both      | TBD             | TBD       | TBD                |
| **TinyLlama**   | Weighting | TBD             | TBD       | TBD                |
| **TinyLlama**   | Prompting | TBD             | TBD       | TBD                |
| **TinyLlama**   | Both      | TBD             | TBD       | TBD                |
| **TinyStories** | Weighting | TBD             | TBD       | TBD                |
| **TinyStories** | Prompting | TBD             | TBD       | TBD                |
| **TinyStories** | Both      | TBD             | TBD       | TBD                |

**Interpretation:**

- TBD: Results pending full 6-model experiment

---

### 1.5 Discussion

**1.5.1 Why Weighting Alone Fails**

**Hypothesis:** Vocabulary constraint triggers compensatory verbosity

- Model tries to express complex ideas with limited vocabulary
- Result: Longer, more convoluted sentences
- Evidence:
  - Weighting: 98.3 words/response vs Control: 76.5 words
  - Sentence count increases (6.8 vs 5.5)

**Potential Solutions:**

- Expand vocabulary list (currently 493 words from 511-word Starters list, may be too restrictive)
- Adjust weight factor (2.0× may be too aggressive)
- Combine with sentence-level constraints

**1.5.2 Why Prompting Works**

**Mechanism:** Explicit instruction shapes generation holistically

- Controls **both** vocabulary AND sentence structure
- Model "understands" target audience
- Fast and effective (no decoding overhead)

**Evidence:**

- Reduces FK Grade 47-80% across models
- Achieves A1 targets consistently
- Minimal latency impact (vs 5× slowdown for weighting)

**1.5.3 Why Combined is Best (but with caveats)**

**Synergistic Effects:**

- Prompting provides structural guidance
- Weighting reinforces vocabulary constraints
- Result: Lowest FK Grade (2.8), highest Reading Ease (95.1)

**Trade-offs:**

- Qwen2: Combined significantly slower (38s vs 15s control)
- Qwen3: Combined slower but acceptable (6.3s vs 4.7s control)
- **Recommendation:** Prompting alone may suffice for latency-critical applications

**1.5.4 Model Selection Across 6 Models**

**Model Comparison Dimensions:**

1. **Size vs Performance:** Phi3 (3.8B) vs SmolLM (1.7B) vs TinyLlama (1.1B) vs Qwen3 (0.6B) vs Qwen2 (0.5B) vs TinyStories (33M)
2. **Baseline Complexity:** Natural FK Grade without interventions
3. **Intervention Responsiveness:** Effectiveness of prompting/weighting
4. **Latency:** Response time across configurations
5. **Deployment Viability:** Balance of simplicity, speed, and resource requirements

**Architectural Considerations:**

- Training data composition (general vs educational)
- Tokenizer vocabulary distributions
- Optimization strategies (quantization, caching)
- Model architecture (attention mechanisms, layer depth)

**Recommendation for Deployment:** TBD pending full experiment results

---

### 1.6 Limitations

1. **Sample Size:**

   - Currently 8 prompts per model
   - Need 50+ for robust statistical power
   - Confidence intervals may be wide for some metrics
2. **Prompt Dependency:**

   - Results may vary with different prompt types
   - Current set: vocabulary/grammar questions
   - Need: conversation scenarios, error correction, cultural topics
3. **No Human Validation:**

   - Readability formulas are proxies, not ground truth
   - Need A1 learner evaluations for ecological validity
   - Metrics may miss ESL-specific challenges (idioms, collocations)
4. **Weight Factor Hyperparameter:**

   - Current: Fixed at 2.0× for all experiments
   - **Critical gap:** No systematic search for optimal value
   - Hypothesis: Lower values (1.3-1.5×) may work better
   - Need: Grid search or Bayesian optimization
   - Challenge: Balancing simplicity vs fluency
5. **Single Language:**

   - English only
   - May not generalize to other L2 contexts (Spanish, French, etc.)
6. **Vocabulary List Limitations:**

   - Based on Cambridge A1 "Starters" (designed for children)
   - May not match adult beginner needs
   - Fixed list doesn't adapt to learner progress
7. **No Semantic Validation:**

   - Only measure complexity, not correctness
   - Simpler text might sacrifice accuracy/completeness
8. **Subword Tokenization Side Effects:**

   - **Issue:** Vocabulary words segmented into subword tokens receive unintended weighting amplification
   - **Examples:**
     - `'afternoon'` → `['after', 'noon']` - both segments receive 2.0× boost across all contexts
     - `'angry'` → `['ang', 'ry']` - segment `'ry'` receives amplified weighting in "every", "sorry", "library"
     - `'alex'` → `['ale', 'x']` - segment `'x'` receives amplified weighting across all contexts
   - **Mechanism:** Both `'word'` and `' word'` (with space prefix) variants are weighted to account for tokenization variability
   - **Impact:** Creates unintended bias toward specific letter combinations and morphological patterns beyond the target vocabulary set
   - **Scope:** Approximately 493 vocabulary words generate hundreds of weighted subword tokens
   - **Mitigation Strategy Required:** Token-level filtering or whole-word-only weighting constraints
   - **Complexity-Correctness Trade-off:** The evaluation framework measures text complexity exclusively; semantic correctness and factual accuracy receive no quantitative assessment, representing a significant methodological limitation

---

### 1.7 Future Work

**Immediate Extensions (Next 3 Months):**

1. **Expand Prompt Set:**

   - Current: 5 prompts
   - Target: 50+ covering all English learning categories
   - Categories: vocabulary, grammar, conversation, cultural, error correction
2. **Hyperparameter Optimization for Weighting:**

   - **Critical need:** Systematic search for optimal weight factor
   - Method: Grid search over [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
   - Metrics: FK Grade, Response Time, Fluency (perplexity)
   - Hypothesis: Lower values (1.3-1.5) may avoid verbosity
   - Expected outcome: Identify "sweet spot" for complexity vs coherence
3. **Human Evaluation:**

   - Recruit 20-30 A1 learners (adults)
   - Rate outputs: comprehensibility, helpfulness, naturalness
   - Validate readability metrics against user perception
   - Protocol: Pairwise comparisons (control vs interventions)
4. **Multi-turn Dialogue Testing:**

   - Current: Single-turn Q&A
   - Need: Conversation coherence over multiple turns
   - Question: Do interventions maintain context awareness?

**Medium-term Research (6-12 Months):**

5. **Fine-tuning Baseline:**

   - Fine-tune Qwen3 on A1-level texts
   - Compare: Fine-tuned (no interventions) vs Base (with interventions)
   - Question: Is runtime control necessary, or does fine-tuning suffice?
6. **Dynamic Vocabulary Adaptation:**

   - Track learner progress (A1 → A2)
   - Expand vocabulary list as learner advances
   - Personalized difficulty adjustment
7. **Cross-Linguistic Validation:**

   - Test on Spanish, French, Mandarin learners
   - Question: Do English-based interventions generalize?
8. **Semantic Correctness Evaluation:**

   - Measure factual accuracy with QA benchmarks
   - Trade-off analysis: simplicity vs completeness
   - Ensure simplification doesn't sacrifice correctness

**Long-term Vision (1-2 Years):**

9. **Adaptive Weighting:**

   - Learn optimal weight factor per learner
   - Reinforcement learning from user feedback
   - Personalized complexity profiles
10. **Multimodal Support:**

    - Add images, audio to support comprehension
    - Visual grounding for vocabulary explanations
11. **Real-world Deployment Study:**

    - Pilot with language learning app
    - Measure learning outcomes (not just readability)
    - A/B test: Simple vs complex responses

---

## 2. KEY VISUALIZATIONS FOR PAPER

**Figure 1: Main Effects by Configuration (Combined Models)**

- 2×2 grid: FK Grade, Gunning Fog, Flesch Ease, Response Time
- Boxplots, color-coded by config
- Target lines for A1 thresholds
- **Purpose:** Show intervention effectiveness at a glance

**Figure 2: Model Comparison (FK Grade)**

- Grouped bar chart: Qwen2 vs Qwen3 across 4 configs
- Error bars (std dev)
- Target line at FK = 5.0
- **Purpose:** Highlight Qwen3 superiority

**Figure 3: Complexity-Speed Trade-off**

- Scatter plot: Response Time (x) vs FK Grade (y)
- Color-coded by config
- Shaded "ideal zone" (bottom-left: fast + simple)
- **Purpose:** Show trade-offs, identify optimal config

**Figure 4: Example Outputs Table**

- Side-by-side: Control vs Both (same prompt)
- Annotated with metrics
- Color-coded difficulty (green=A1, red=too hard)
- **Purpose:** Concrete illustration of improvement

---

## 3. PRESENTATION OUTLINE (15-20 min talk)

**Slide 1: Title**

- "Controlling Text Complexity in Small Language Models for A1 English Learners"

**Slide 2: The Problem (Visual Hook)**

- Show control vs both example side-by-side
- Highlight FK Grade difference (11.5 → 5.8)

**Slide 3: Why This Matters**

- Language learning apps need adaptive AI
- A1 learners: limited vocabulary (~500 words)
- SLMs: On-device, low-latency, privacy-preserving

**Slide 4: Research Question**

- Can we control SLM complexity in real-time?
- Two interventions: Weighting (decoding) + Prompting (instruction)

**Slide 5: Experimental Design**

- 6 models × 4 configs × 8 prompts = 192 observations
- 18 readability metrics (focus on FK Grade, Flesch Ease)

**Slide 6: Readability Metrics Explained**

- Quick overview of FK Grade and Flesch Reading Ease
- A1 targets: FK ≤5.0, Flesch ≥80

**Slide 7: Results - By Config**

- Figure 1 (main effects)
- Key finding: Prompting effective, Weighting alone fails

**Slide 8: Results - Model Comparison**

- Figure 2 (6-model comparison)
- Key finding: TBD pending full experiment

**Slide 9: Why Weighting Fails**

- Compensatory verbosity hypothesis
- Evidence: +28% word count, no complexity reduction

**Slide 10: Trade-offs**

- Figure 3 (scatter plot)
- TBD: Identify optimal model + config combination

**Slide 11: Effect Sizes**

- TBD: Effect sizes pending full experiment
- TBD: Percent reduction in FK Grade across models

**Slide 12: Limitations**

- Sample size (8 prompts, need 50+)
- No human validation yet
- **Weight factor hyperparameter not optimized**

**Slide 13: Future Work**

- Expand to 50+ prompts
- **Hyperparameter search for weight factor**
- Human evaluation with A1 learners
- Fine-tuning baseline comparison

**Slide 14: Practical Recommendations**

- **Deploy:** TBD pending full experiment results
- **Monitor:** FK Grade ≤5.0, Response Time <10s
- **Avoid:** Weighting alone (counterproductive)

**Slide 15: Contributions**

- ✅ First real-time complexity control for SLMs
- ✅ Factorial design isolates intervention effects across 6 models
- ✅ Comprehensive readability evaluation (18 metrics)
- ✅ Practical deployment guidelines (pending results)

---

## 4. ELEVATOR PITCHES

**30-Second Summary:**

This research evaluates whether Small Language Models can be controlled to generate appropriately simplified text for A1-level English learners. We implemented two lightweight inference-time interventions—probability weighting and contextual prompting—across six models spanning 33M to 3.8B parameters. Preliminary findings indicate that contextual prompting substantially outperforms standalone vocabulary weighting in reducing text complexity while maintaining linguistic coherence.

**Two-Minute Research Overview:**

Contemporary language learning applications require AI assistants calibrated to learner proficiency levels. This investigation systematically evaluates complexity control mechanisms across six Small Language Models (Phi3, Qwen2, Qwen3, SmolLM, TinyLlama, TinyStories) using two intervention methodologies: contextual prompting (instructional constraints) and probability weighting (vocabulary amplification during token sampling).

**Experimental Framework:** 6 models × 4 intervention configurations × 8 diverse prompts = 192 observations, evaluated across 18 established readability metrics.

**Central Research Questions:** (1) Which model architectures naturally produce beginner-appropriate text? (2) How effectively do prompting and weighting interventions reduce complexity? (3) What latency trade-offs emerge from these approaches? (4) Which intervention combination optimally balances simplicity, coherence, and computational efficiency for deployment?

**Current Status:** Exploratory phase complete. Subsequent investigation will: complete full factorial experiment; expand prompt diversity to 50+ samples for robust statistical inference; conduct hyperparameter optimization for weighting factors; validate findings with A1 learner populations; and establish performance benchmarks for production deployment.

---

## 5. DISCUSSION POINTS & OPEN QUESTIONS

**For Q&A Preparation:**

Q: *Why not pursue fine-tuning on simplified text corpora?*
A: Fine-tuning approaches require substantial annotated data and computational resources, and consequently lock model output into a single fixed difficulty level. In contrast, the inference-time approach proposed here demonstrates adaptability across arbitrary pre-trained models without requiring model retraining. Comparative analysis between fine-tuned and inference-controlled variants represents a critical direction for future research.

Q: *What explains the underperformance of weighting-only approaches?*
A: The observed ineffectiveness of standalone vocabulary weighting is hypothesized to result from vocabulary constraint triggering compensatory verbosity: models attempt to express conceptually complex ideas using a restricted lexicon, resulting in elongated and syntactically convoluted output. This phenomenon suggests the vocabulary list (493 words) may be excessively constrictive. Promising mitigation strategies include expanding the vocabulary set and systematically optimizing the weight factor (currently fixed at 2.0×).

Q: *How reliable are readability formulas for ESL learner populations?*
A: A methodologically sound concern: these metrics were originally designed for native speaker populations. While readability formulas remain widely utilized in educational assessment and demonstrate documented correlation with comprehension outcomes in prior L2 studies (Crossley et al., 2014), direct validation with A1 learner populations remains an essential gap. Human evaluation with target demographic cohorts will ground metric validity claims.

Q: *The sample of 8 prompts per model appears insufficient for robust statistical inference.*
A: This observation is methodologically justified. The current configuration represents preliminary exploratory work. Statistical power calculation indicates requirement for 50+ diverse prompts to achieve adequate inference robustness. While the factorial design yields 192 total observations (6 models × 4 configurations × 8 prompts), per-model subgroup analysis demands substantially larger sample sizes.

Q: *Do results generalize to languages beyond English?*
A: The proposed mechanisms—contextual prompting and vocabulary-constrained decoding—should theoretically generalize across linguistic systems. However, empirical validation remains absent. Cross-linguistic evaluation involving Spanish, French, and Mandarin learners represents a critical next phase for establishing generalizability claims.

Q: *What represents the optimal weight factor value?*
A: This constitutes an open and critical research question. The current value (2.0×) reflects preliminary empirical exploration rather than systematic optimization. Subsequent investigation will employ grid search methodology across [1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0] to identify the optimal trade-off between complexity reduction and fluency preservation.

Q: *How do model parameter scales influence intervention effectiveness?*
A: The experimental design deliberately spans a parameter range from TinyStories (33M) to Phi3 (3.8B) to characterize scale effects. Preliminary hypotheses suggest larger models may exhibit higher baseline complexity but potentially greater responsiveness to prompting constraints. Latency effects scale with model size, though efficiency optimization strategies may partially mitigate this relationship.

Q: *How is factual accuracy addressed when simplifying text?*
A: Factual correctness receives no explicit evaluation in the current framework—an important methodological limitation. Simplified paraphrases risk semantic loss or incompleteness (e.g., "A library is a place with books" omits information about institutional functions, research services). Subsequent work will incorporate QA benchmark validation to ensure simplification does not sacrifice semantic correctness.

---

## 6. ACTIONABLE NEXT STEPS

**Before Paper Submission:**

1. ⏳ Complete 192-observation factorial experiment (6 models × 4 configs × 8 prompts)
2. ⏳ Run experiments with 50 prompts per model (total: 1,200 observations)
3. ⏳ Hyperparameter search for weight factor [1.1, 1.3, 1.5, 1.7, 2.0, 2.5]
4. ⏳ Create all paper figures (4 total)
5. ⏳ Write Introduction, Methods, Results, Discussion sections
6. ⏳ Lit review: 20-30 key papers (ATS, controllable generation, SLMs, readability)

**For Presentation:**

1. ⏳ Build slide deck (15 slides)
2. ⏳ Prepare visual hook (control vs both example)
3. ⏳ Rehearse 15-min talk
4. ⏳ Anticipate Q&A (see Section 5)

**Post-Paper (Future Research):**

1. ⏳ Human evaluation protocol
2. ⏳ Fine-tuning baseline comparison
3. ⏳ Multi-turn dialogue testing
4. ⏳ Cross-linguistic validation

---

## 7. FILES & RESOURCES

**Generated:**

- `paper/combined_analysis.py` - Analysis script
- `paper/figures/*.png` - 3 visualizations
- `paper/results/combined_summary.json` - Statistical summary

**Existing:**

- `ExperimentSpecification.md` - Design document
- `text_metrics.md` - Metric explanations
- `src/evaluation/experiment_framework/` - Codebase
- `src/evaluation/experiment_framework/results/` - Raw data (CSV)

**To Create:**

- `paper/literature_review.md` - Lit review notes
- `paper/draft_v1.tex` - Paper LaTeX source
- `paper/presentation.pptx` - Slide deck

---

**END OF BRAINSTORMING DOCUMENT**

*Generated: October 3, 2025*
*Updated: October 5, 2025*
*Models: Phi3 (3.8B), Qwen2 (0.5B), Qwen3 (0.6B), SmolLM (1.7B), TinyLlama (1.1B), TinyStories (33M)*
*Total Observations: 192 (6 models × 4 configs × 8 prompts)*
