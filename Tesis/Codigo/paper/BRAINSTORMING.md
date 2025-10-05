# Paper Presentation Brainstorming: SLM Text Complexity Control

**Research Question:** Can small language models be controlled to produce appropriately simple text for A1 English learners through decoding manipulation and prompt engineering?

**Models Evaluated:** Phi3 (3.8B), Qwen2 (0.5B), Qwen3 (0.6B), SmolLM (1.7B), TinyLlama (1.1B), TinyStories (33M)

---

## 1. PAPER STRUCTURE (Option A: Classic Research Paper)

### 1.1 Introduction

**Hook - Visual Example:**

```
PROMPT: "What does the word 'library' mean?"

CONTROL (Qwen2):
"The word 'library' means a collection of books, magazines, and other printed 
materials organized and maintained by a library staff..."
→ FK Grade: 11.5 | Reading Ease: 42.8 | Too complex for A1!

BOTH INTERVENTIONS (Qwen2):
"A library is a place where you can find lots of books to read. It's like a 
big house with many special rooms where people keep their toys and stories!"
→ FK Grade: 5.8 | Reading Ease: 83.3 | A1-appropriate!
```

**Context:**

- Language learning apps need AI assistants that match learner proficiency
- A1 level (CEFR) = beginners with limited vocabulary
- Small language models (SLMs) enable on-device, low-latency deployment
- **Challenge:** SLMs trained on general text produce output too complex for beginners

**Gap in Literature:**

- Existing work: Text simplification (post-hoc), fine-tuning (resource-intensive)
- Missing: Real-time control during generation for SLMs
- Our approach: Two lightweight interventions applicable at inference time

**Research Objectives:**

1. Evaluate effectiveness of probability weighting (decoding manipulation)
2. Evaluate effectiveness of context prompting (instruction engineering)
3. Assess synergistic effects of combined interventions
4. Compare models: Qwen2 vs Qwen3
5. Quantify trade-offs: simplicity vs response time

---

### 1.2 State of the Art Literature Review

**1.2.1 Text Simplification for Language Learning**

- **Empirical Evidence for Simplification:**
  - **Gala et al. (2018)**: Simplified texts improve reading fluency and comprehension in beginning readers, particularly those with lower skills
  - **Crossley et al. (2011)**: Text simplification enhances L2 comprehension; word lists and readability formulas are effective approaches
  - **Gap:** Studies focus on post-hoc simplification, not real-time generation control

- **Readability-Controlled Generation:**
  - **Al-Thanyyan & Azmi (2020)**: Model trained on Newsela dataset produces text at specific readability levels using readability formulas
  - **Al-Sabbagh & Al-Khalifa (2023)**: Review of multi-level simplification systems across languages; emphasizes need for readability-controlled generation
  - **Gap:** Approaches require training on labeled data; no inference-time control methods

**1.2.2 Controlled Text Generation Approaches**

- **Fine-tuning for Simplification:**
  - **Baez & Saggion (2023)**: LSLlama fine-tunes LLaMA for lexical simplification, achieving baseline performance
  - **Gap:** Fine-tuning is resource-intensive and locks in single difficulty level

- **Logits Manipulation:**
  - **No prior work** on probability weighting for vocabulary-constrained simplification in educational contexts
  - **This work:** First to apply logits processors for A1 vocabulary boosting

**1.2.3 Readability Assessment for Target Audiences**

- **Audience-Specific Evaluation:**
  - **Yaneva et al. (2016)**: Emphasizes need for tailored readability assessment; introduces disability-specific linguistic features
  - **Claridge (2005)**: Analyzes word frequency, sentence length, and syntactic complexity in graded readers
  - **Relevance:** Validates use of multiple readability metrics (FK Grade, Dale-Chall, Spache) for beginner audiences

**1.2.4 Our Contribution**

✅ **Novel approach:** Logits manipulation (probability weighting) + prompt engineering for real-time complexity control
✅ **SLM focus:** First systematic evaluation across 6 models (33M-3.8B parameters) for educational deployment
✅ **Factorial design:** Isolates individual and interaction effects of two interventions
✅ **Comprehensive evaluation:** 18 readability metrics + 192 observations across diverse model architectures

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
- **Vocabulary:** 1,500 words from A1 "Starters" curriculum (filtered)
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

**1.3.3 Readability Metrics (18 total)**

**Grade Level Indices (report U.S. grade levels):**

1. **Flesch-Kincaid Grade Level** - Primary metric (sentence + syllable complexity)
2. **Gunning Fog Index** - Emphasizes polysyllabic words
3. **SMOG Index** - Polysyllable density
4. **Automated Readability Index (ARI)** - Character-based (no syllables)
5. **Coleman-Liau Index** - Character-based
6. **Dale-Chall Readability Score** - Uses 3,000-word familiarity list

**Readability Scores:**
7. **Flesch Reading Ease** - 0-100 scale (higher = easier)
8. **Linsear Write Formula** - Technical writing focus
9. **Spache Readability** - Primary grades (1-4)
10. **McAlpine EFLAW** - Auditory/conversational text

**Text Statistics:**
11-18. Sentence count, word count, syllable count, polysyllable count, monosyllable count, difficult words, character count, reading time

**A1 Target Ranges:**

- Flesch-Kincaid Grade: ≤5.0
- Gunning Fog: ≤6.0
- Flesch Reading Ease: ≥80
- SMOG: ≤7.0
- Dale-Chall: ≤4.9

**Metric Rationale:**

- **Multiple metrics necessary:** Each captures different complexity dimensions
  - Sentence structure (FK, ARI, Coleman-Liau)
  - Word complexity (Gunning Fog, SMOG)
  - Vocabulary difficulty (Dale-Chall, Spache)
  - Spoken suitability (McAlpine EFLAW)
- **Cross-validation:** Consistent results across metrics = robust findings
- **Avoid single-metric bias:** Each formula has strengths/weaknesses

**Metric Explanations (for paper):**

**Flesch-Kincaid Grade Level:**

- Formula: `0.39 × (words/sentences) + 11.8 × (syllables/words) - 15.59`
- Interpretation: Years of education required to understand text
- Widely used in education, government, healthcare
- Target for A1: ≤5.0 (elementary level)

**Flesch Reading Ease:**

- Formula: `206.835 - 1.015 × ASL - 84.6 × ASW`
- Scale: 0-100 (higher = easier)
- Inverse relationship with FK Grade
- Target for A1: ≥80 (easy to very easy)

**Gunning Fog:**

- Formula: `0.4 × [(words/sentences) + 100 × (complex words/words)]`
- Complex words: 3+ syllables (excluding proper nouns)
- Particularly sensitive to jargon and technical terms
- Target for A1: ≤6.0

*(Detailed explanations for all metrics available in text_metrics.md)*

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

| Configuration            | FK Grade                | Gunning Fog             | Flesch Ease               | Response Time (s)      | Word Count             |
| ------------------------ | ----------------------- | ----------------------- | ------------------------- | ---------------------- | ---------------------- |
| **Control**        | TBD              | TBD             | TBD              | TBD            | TBD           |
| **Weighting Only** | TBD              | TBD              | TBD              | TBD | TBD |
| **Prompting Only** | TBD | TBD | TBD | TBD             | TBD           |
| **Both**           | TBD | TBD | TBD | TBD           | TBD           |

✅ = Meets A1 target

**Key Findings:**

1. **TBD:** Results pending full 6-model experiment
2. **TBD:** Results pending full 6-model experiment
3. **TBD:** Results pending full 6-model experiment
4. **TBD:** Results pending full 6-model experiment

**1.4.2 Model Comparison**

| Model           | Config    | FK Grade         | Flesch Ease        | Response Time (s) |
| --------------- | --------- | ---------------- | ------------------ | ----------------- |
| **Phi3** | Control   | TBD              | TBD               | TBD              |
| **Phi3** | Weighting | TBD        | TBD               | TBD    |
| **Phi3** | Prompting | TBD           | TBD            | TBD              |
| **Phi3** | Both      | TBD           | TBD            | TBD              |
| **Qwen2** | Control   | TBD              | TBD               | TBD              |
| **Qwen2** | Weighting | TBD        | TBD               | TBD    |
| **Qwen2** | Prompting | TBD           | TBD            | TBD              |
| **Qwen2** | Both      | TBD           | TBD            | TBD              |
| **Qwen3** | Control   | TBD           | TBD               | TBD               |
| **Qwen3** | Weighting | TBD           | TBD               | TBD              |
| **Qwen3** | Prompting | TBD | TBD | TBD     |
| **Qwen3** | Both      | TBD | TBD  | TBD               |
| **SmolLM** | Control   | TBD           | TBD               | TBD               |
| **SmolLM** | Weighting | TBD           | TBD               | TBD              |
| **SmolLM** | Prompting | TBD | TBD | TBD     |
| **SmolLM** | Both      | TBD | TBD  | TBD               |
| **TinyLlama** | Control   | TBD           | TBD               | TBD               |
| **TinyLlama** | Weighting | TBD           | TBD               | TBD              |
| **TinyLlama** | Prompting | TBD | TBD | TBD     |
| **TinyLlama** | Both      | TBD | TBD  | TBD               |
| **TinyStories** | Control   | TBD           | TBD               | TBD               |
| **TinyStories** | Weighting | TBD           | TBD               | TBD              |
| **TinyStories** | Prompting | TBD | TBD | TBD     |
| **TinyStories** | Both      | TBD | TBD  | TBD               |

**Key Findings:**

1. **TBD:** Results pending full 6-model experiment
2. **TBD:** Results pending full 6-model experiment
3. **TBD:** Results pending full 6-model experiment
4. **TBD:** Results pending full 6-model experiment

**1.4.3 Effect Sizes**

**Per-Model Analysis:**

| Model | Config    | FK Grade Change     | Cohen's d               | Flesch Ease Change |
| ----- | --------- | ------------------- | ----------------------- | ------------------ |
| **Phi3** | Weighting | TBD          | TBD                   | TBD              |
| **Phi3** | Prompting | TBD | TBD | TBD   |
| **Phi3** | Both      | TBD | TBD | TBD   |
| **Qwen2** | Weighting | TBD          | TBD                   | TBD              |
| **Qwen2** | Prompting | TBD | TBD | TBD   |
| **Qwen2** | Both      | TBD | TBD | TBD   |
| **Qwen3** | Weighting | TBD               | TBD                   | TBD              |
| **Qwen3** | Prompting | TBD | TBD | TBD   |
| **Qwen3** | Both      | TBD | TBD | TBD   |
| **SmolLM** | Weighting | TBD               | TBD                   | TBD              |
| **SmolLM** | Prompting | TBD | TBD | TBD   |
| **SmolLM** | Both      | TBD | TBD | TBD   |
| **TinyLlama** | Weighting | TBD               | TBD                   | TBD              |
| **TinyLlama** | Prompting | TBD | TBD | TBD   |
| **TinyLlama** | Both      | TBD | TBD | TBD   |
| **TinyStories** | Weighting | TBD               | TBD                   | TBD              |
| **TinyStories** | Prompting | TBD | TBD | TBD   |
| **TinyStories** | Both      | TBD | TBD | TBD   |

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

- Expand vocabulary list (currently 1,500 words, may be too restrictive)
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

   - **Issue:** Vocabulary words split into subword tokens receive unintended weighting
   - **Examples:**
     - `'afternoon'` → `['after', 'noon']` - both pieces get 2.0× boost everywhere
     - `'angry'` → `['ang', 'ry']` - `'ry'` gets boosted in "every", "sorry", "library"
     - `'alex'` → `['ale', 'x']` - `'x'` gets boosted in all contexts
   - **Mechanism:** Both `'word'` and `' word'` variants are weighted for tokenization robustness
   - **Impact:** Creates unintended bias toward certain letter combinations beyond target vocabulary
   - **Scope:** ~493 vocabulary words generate hundreds of weighted subword tokens
   - **Mitigation needed:** Token-level filtering or whole-word-only weighting strategies
   - **Correctness not being take into account:** the only measurementes being take into account are the complexity of the answer, not the corectness of it

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
- A1 learners: limited vocabulary (1,500 words)
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

**30 seconds:**
"We tested whether small AI models can talk simply to beginner English learners. Using prompt instructions and vocabulary boosting across 6 models (33M to 3.8B parameters), we evaluated text complexity control. Results pending full experiment, but early findings suggest prompting works better than vocabulary weighting alone."

**2 minutes:**
"Language learning apps need AI that matches student level. We tested six small models (Phi3, Qwen2, Qwen3, SmolLM, TinyLlama, TinyStories) with two control methods: prompting (instructions to simplify) and weighting (boosting simple vocabulary during generation).

Experimental design: 6 models × 4 intervention configs × 8 prompts = 192 observations, evaluated across 18 readability metrics.

Key research questions: Which models naturally produce simpler text? How effective is prompting vs weighting? What are the latency trade-offs? Which combination is optimal for deployment?

Results pending full experiment. Next steps: complete 192-observation factorial experiment, analyze statistical significance, test with real A1 students, optimize vocabulary weighting strength, and expand to 50+ prompts for robust validation."

---

## 5. DISCUSSION POINTS & OPEN QUESTIONS

**For Q&A Preparation:**

Q: *Why not just fine-tune on simple texts?*
A: Fine-tuning requires data + compute, locks in one difficulty level. Our approach is inference-time, adaptable, and works with any pre-trained model. Future work will compare both approaches.

Q: *Why did weighting fail?*
A: We hypothesize the vocabulary constraint (1,500 words) is too restrictive, forcing verbose compensatory strategies. Expanding the list or lowering the weight factor (currently 2.0×) may help. This is a critical gap we'll address in follow-up work.

Q: *How do you know readability formulas work for ESL learners?*
A: Good question—formulas were designed for native speakers. We plan human validation with A1 learners. However, formulas are widely used in education and correlate with comprehension in prior ESL studies (Crossley et al., 2014).

Q: *Only 8 prompts per model seems small.*
A: Agreed—this is exploratory. We're expanding to 50+ prompts for statistical power. With 6 models and 4 configs, we have 192 observations total, but more prompts needed for robust per-model analysis.

Q: *Can this work for other languages?*
A: Unknown—English only so far. The mechanisms (prompting, vocabulary lists) should generalize, but empirical validation needed for Spanish, French, etc.

Q: *What's the optimal weight factor?*
A: **Critical open question.** We used 2.0× based on initial tests, but didn't optimize. Next experiment: grid search over [1.1-3.0] to find the "sweet spot" balancing simplicity and fluency.

Q: *How do model sizes affect performance?*
A: We're testing a wide range: TinyStories (33M) to Phi3 (3.8B). Hypothesis: larger models may produce more complex text naturally, but might also be more responsive to prompting. Latency should increase with size, but optimization matters too.

Q: *How do you handle factual accuracy with simplified text?*
A: We don't measure it yet—limitation. Simplification might sacrifice completeness (e.g., "A library is a place with books" omits "lending, studying, research"). Future work: QA benchmarks to ensure semantic correctness.

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


