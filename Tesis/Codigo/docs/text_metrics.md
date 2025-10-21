# Text Readability Metrics Documentation

This document provides comprehensive explanations of all readability and complexity metrics used in the experimental framework for evaluating Small Language Model (SLM) outputs.

### Metrics selected

**Grade level:**

- Flesch-Kincaid Grade Level: Reading difficulty based on sentence length and word complexity (syllables per word).
- Gunning Fog Index: Years of formal education needed to understand text, with emphasis on complex (polysyllabic) words.
- SMOG Index: Reading difficulty based on polysyllabic word density. Considered highly accurate for technical and medical texts

**Readability score:**

- Spache Readability: Reading difficulty for primary-grade materials (grades 1-4). Based on a list of familiar words for young readers.

---

## Table of Contents

1. [Primary Analysis Metrics](#primary-analysis-metrics)
   - [Grade Level Indices](#grade-level-indices)
   - [Readability Scores](#readability-scores)
2. [Secondary Descriptive Statistics](#secondary-descriptive-statistics)
3. [Target Ranges for A1 English Learners](#target-ranges-for-a1-english-learners)
4. [Metric Selection Rationale](#metric-selection-rationale)
5. [Discarded Metrics](#discarded-metrics)

---

## Primary Analysis Metrics

These metrics form the core of the statistical analysis. They were selected to provide comprehensive coverage of text complexity while avoiding redundancy.

---

### Grade Level Indices

Grade level indices estimate the years of formal education required to understand a text on first reading. All report U.S. grade levels (e.g., 5.0 = 5th grade). These metrics are essential for establishing educational difficulty benchmarks.

#### 1. Flesch-Kincaid Grade Level

**What it measures:** Reading difficulty based on sentence length and word complexity (syllables per word).

**Formula:**

```
Grade Level = (0.39 × ASL) + (11.8 × ASW) - 15.59
```

Where:

- **ASL** = Average Sentence Length (words per sentence)
- **ASW** = Average Syllables per Word

**Interpretation:**

- **0-5**: Very easy (Elementary school) - **TARGET for A1 learners**
- **6-8**: Easy (Middle school)
- **9-10**: Fairly easy (High school)
- **11-12**: Moderate (College level)
- **13-16**: Difficult (College graduate)
- **17+**: Very difficult (Professional/academic)

**Why included:** Most established grade-level metric with strong academic validation. Balances sentence structure and syllabic complexity. Lower values indicate the intervention successfully simplified model output.

---

#### 2. Gunning Fog Index

**What it measures:** Years of formal education needed to understand text, with emphasis on complex (polysyllabic) words.

**Formula:**

```
Fog Index = 0.4 × [(Words/Sentences) + 100 × (Complex Words/Words)]
```

Where:

- **Complex Words** = Words with 3+ syllables (excluding proper nouns, familiar jargon, compound words)

**Interpretation:**

- **6**: Sixth grade - **TARGET for A1 learners**
- **8**: Eighth grade
- **12**: High school senior
- **17+**: College graduate

**Why included:** Particularly sensitive to polysyllabic words, making it ideal for evaluating vocabulary weighting interventions. Scores below 6 indicate very simple vocabulary appropriate for beginners.

---

#### 3. SMOG Index (Simple Measure of Gobbledygook)

**What it measures:** Reading difficulty based on polysyllabic word density. Considered highly accurate for technical and medical texts.

**Formula:**

```
SMOG Grade = 3 + √(Polysyllable Count in 30 sentences)
```

**Interpretation:**

- **7-9**: Junior high school
- **10-12**: High school
- **13-16**: College
- **17+**: Graduate school

**Why included:** Highly reliable for short texts and less sensitive to sentence length variations than Flesch-Kincaid. Provides robust polysyllable-based assessment. SMOG Grade ≤7 is **TARGET for A1 learners**.

---

### Readability Scores

Readability scores use different scales (not grade levels) but all measure how easy text is to read.

#### 4. Spache Readability

**What it measures:** Reading difficulty for primary-grade materials (grades 1-4). Based on a list of familiar words for young readers.

**Formula:**

```
Spache Score = (0.141 × ASL) + (0.086 × % Unfamiliar Words) + 0.839
```

Where:

- **Unfamiliar Words** = Words NOT on Spache word list (designed for grades 1-4)

**Interpretation:**

- **1-2**: 1st-2nd grade - **IDEAL for A1 learners**
- **3-4**: 3rd-4th grade - **TARGET for A1 learners**
- **5+**: Above primary level

**Why included:** Specifically designed for primary-grade materials (grades 1-4), perfectly matching A1 proficiency level. Uses vocabulary list (~1,000 words) calibrated for early readers, providing superior discrimination at beginner levels compared to broader word-list metrics. Ideal for evaluating weighted vocabulary intervention effectiveness with A1 learners.

---

## Secondary Descriptive Statistics

These metrics provide supplementary insights into model behavior and response characteristics. While not used for primary statistical analysis, they help interpret results and understand practical implications.

### Word Count

**What it measures:** Total number of words in the response.

**Why included:** While prompt-dependent, word count reveals verbosity differences between configurations. Helps assess whether interventions produce more concise responses, which reduces cognitive load for A1 learners. Useful for understanding model behavior patterns.

**Interpretation:** Shorter responses (30-60 words) are typically better for A1 learners, balancing information delivery with cognitive load.

---

### Difficult Words Count

**What it measures:** Words classified as "difficult" by the `textstat` library. A word is considered difficult if it: (1) is NOT in the Dale-Chall easy word list (2,940 common English words for grades 4-16+), AND (2) has 3+ syllables (syllable_threshold=2, meaning MORE than 2 syllables). This dual-criteria approach identifies vocabulary that is both uncommon and phonologically complex.

**Why included:** Provides a direct, interpretable count that's easier to communicate than abstract scores. "Model A reduced difficult words by 40%" is more intuitive than "Dale-Chall score decreased from 6.2 to 4.8." Complements word-list-based metrics while offering a straightforward accessibility measure.

**Interpretation:** Lower is better for A1 learners. Target: minimize difficult words to improve vocabulary accessibility.

---

## Target Ranges for A1 English Learners

Based on Common European Framework of Reference (CEFR) A1 level, target ranges for successful interventions:

### Primary Metrics

| Metric                         | Target Range | Interpretation            |
| ------------------------------ | ------------ | ------------------------- |
| **Flesch-Kincaid Grade** | ≤5.0        | Elementary level or below |
| **Gunning Fog**          | ≤6.0        | Sixth grade or below      |
| **SMOG Index**           | ≤7.0        | Junior high or below      |
| **Spache Readability**   | ≤4.0        | Primary grades (1-4)      |

### Secondary Statistics

| Metric                    | Target Range | Interpretation           |
| ------------------------- | ------------ | ------------------------ |
| **Word Count**      | 30-60 words  | Concise, manageable      |
| **Difficult Words** | Minimize     | Prefer common vocabulary |

---

## Metric Selection Rationale

### Why These Four Primary Metrics?

The selected metrics provide comprehensive coverage of text complexity while avoiding redundancy:

1. **Sentence Structure**: Flesch-Kincaid and SMOG capture sentence length effects
2. **Polysyllabic Complexity**: Gunning Fog and SMOG emphasize complex word identification
3. **Vocabulary Difficulty**: Spache uses a reference word list (~1,000 words) specifically calibrated for primary grades
4. **Target Audience Match**: Spache is designed for grades 1-4, perfectly matching A1 proficiency level
5. **No Redundancy**: Each metric provides unique information without duplicating others

### Complementary Strengths

- **Syllable-based** (Flesch-Kincaid, Gunning Fog): Capture phonological complexity
- **Word list-based** (Spache): Directly measures vocabulary level against established benchmark for early readers
- **Polysyllable-focused** (SMOG, Gunning Fog): Identify complex terminology that challenges beginners
- **Primary-grade calibrated** (Spache): Specifically designed for grades 1-4, providing superior discrimination at A1 level

### Cross-Validation Benefits

Using four metrics enables:

- **Robust validation**: Consistent improvement across metrics indicates genuine intervention effectiveness
- **Trade-off identification**: Reveals if prompting reduces word difficulty but increases sentence length
- **Mechanism understanding**: Weighted vocabulary primarily affects Spache; prompting affects sentence structure metrics (Flesch-Kincaid, SMOG)
- **Bias avoidance**: No single formula is perfect; multiple metrics prevent over-reliance on one approach

### Statistical Analysis

The factorial design (4 models × 4 configs × N prompts) with four metrics enables:

- **MANOVA**: Test intervention effects across multiple dependent variables simultaneously
- **Interaction analysis**: Determine if weighting + prompting effects are additive or synergistic
- **Model comparison**: Identify which SLMs naturally produce simpler output
- **Metric correlation**: Understand relationships between different complexity dimensions

---

## Discarded Metrics

The following metrics were considered but excluded from the final analysis:

### Flesch Reading Ease

**Reason for exclusion:** Highly redundant with Flesch-Kincaid Grade Level. Both use nearly identical formulas (same inputs: Average Sentence Length and Average Syllables per Word) with different coefficients and inverse scaling. Flesch-Kincaid Grade Level is more interpretable for educational contexts ("5th grade level" vs. "score of 85").

### Dale-Chall Readability Score

**Reason for exclusion:** Redundant with Spache Readability. Both are word-list-based metrics measuring vocabulary difficulty. Dale-Chall uses a 3,000-word list covering grades 4-16+, but provides poor discrimination at A1 level (all scores below 4.9 are "4th grade or below"). Spache uses a ~1,000-word list specifically calibrated for grades 1-4, providing superior granularity for A1 learners (distinguishes 1st-2nd grade from 3rd-4th grade).

### Automated Readability Index (ARI)

**Reason for exclusion:** Redundant with Coleman-Liau Index. Both are character-based metrics that don't require syllable parsing. Coleman-Liau has stronger academic validation.

### Coleman-Liau Index

**Reason for exclusion:** While computationally efficient (character-based), it provides similar information to Flesch-Kincaid without additional insights. The four selected metrics already cover sentence structure adequately.

### Linsear Write Formula

**Reason for exclusion:** Designed specifically for technical writing and instructional manuals. Not appropriate for conversational chatbot responses typical of language learning interactions.

### McAlpine EFLAW (Easy Listening Formula for Auditory Writing)

**Reason for exclusion:** Designed for spoken/auditory content. This thesis focuses on text-based chatbot interactions without audio components.

### Text Statistics (Sentence Count, Character Count, Syllable Count, Polysyllable Count, Monosyllable Count)

**Reason for exclusion:** These are component metrics used in formulas rather than standalone measures. They're calculated internally for the primary metrics but don't provide independent insights. Reporting them would be redundant.

### Reading Time

**Reason for exclusion:** Linear transformation of word count (÷200 words/minute). Provides no additional statistical information beyond word count, which is already included as a secondary statistic.

---

## References

- Flesch, R. (1948). "A new readability yardstick." *Journal of Applied Psychology*, 32(3), 221-233.
- Kincaid, J.P., et al. (1975). "Derivation of new readability formulas for Navy enlisted personnel." *Research Branch Report 8-75*.
- Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
- McLaughlin, G.H. (1969). "SMOG grading: A new readability formula." *Journal of Reading*, 12(8), 639-646.
- Spache, G. (1953). "A new readability formula for primary-grade reading materials." *The Elementary School Journal*, 53(7), 410-413.

---

## Usage in Code

All metrics are calculated using the `TextEvaluator` class in `src/evaluation/text_complexity/text_evaluator.py`, which wraps the `textstat` library:

```python
from src.evaluation.text_complexity.text_evaluator import TextEvaluator

evaluator = TextEvaluator()
text = "The cat sat on the mat."

# Get comprehensive analysis (includes all metrics)
analysis = evaluator.evaluate_text_comprehensive(text)

# Access primary metrics
print(f"Flesch-Kincaid Grade: {analysis['flesch_kincaid_grade']}")
print(f"Gunning Fog: {analysis['gunning_fog']}")
print(f"SMOG Index: {analysis['smog_index']}")
print(f"Spache: {analysis['spache_readability']}")

# Access secondary statistics
print(f"Word Count: {analysis['word_count']}")
print(f"Difficult Words: {analysis['difficult_words']}")
```

Metrics are automatically calculated for all experiment results and exported to CSV files for statistical analysis.

---

## Summary

**Primary Analysis Metrics (4):**

1. Flesch-Kincaid Grade Level
2. Gunning Fog Index
3. SMOG Index
4. Spache Readability

**Secondary Descriptive Statistics (2):**

1. Word Count
2. Difficult Words Count

This streamlined set provides comprehensive text complexity assessment while eliminating redundancy:

- **Removed Flesch Reading Ease**: Redundant with Flesch-Kincaid (same formula, different scaling)
- **Removed Dale-Chall**: Redundant with Spache (both word-list-based; Spache provides superior A1-level discrimination)

The final four metrics maintain focus on A1 English learners with zero redundancy.

---

## Appendix: Complete Metric Evaluation

This appendix provides a comprehensive overview of all metrics originally considered for the experimental framework, including both selected and discarded options.

### Selected Metrics (4 Primary + 2 Secondary)

#### Primary Analysis Metrics

| Metric                               | Category          | Formula Components                          | Selection Rationale                                                                      |
| ------------------------------------ | ----------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Flesch-Kincaid Grade Level** | Grade Level Index | ASL, ASW                                    | Most established grade-level metric; balances sentence structure and syllabic complexity |
| **Gunning Fog Index**          | Grade Level Index | ASL, Complex Words (3+ syllables)           | Highly sensitive to polysyllabic words; ideal for vocabulary weighting evaluation        |
| **SMOG Index**                 | Grade Level Index | Polysyllable count                          | Reliable for short texts; less sensitive to sentence length variations                   |
| **Spache Readability**         | Readability Score | ASL, Unfamiliar Words (vs. 1,000-word list) | Specifically calibrated for grades 1-4; superior A1-level discrimination                 |

#### Secondary Descriptive Statistics

| Metric                          | Purpose                  | Selection Rationale                                                      |
| ------------------------------- | ------------------------ | ------------------------------------------------------------------------ |
| **Word Count**            | Verbosity measure        | Reveals conciseness differences; assesses cognitive load                 |
| **Difficult Words Count** | Vocabulary accessibility | Direct, interpretable metric; easier to communicate than abstract scores |

### Discarded Metrics with Detailed Rationale

#### Redundant with Selected Metrics

| Metric                                      | Redundant With             | Detailed Reason for Exclusion                                                                                                                                                                                                                                                                          |
| ------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Flesch Reading Ease**               | Flesch-Kincaid Grade Level | Uses identical formula inputs (ASL, ASW) with different coefficients. Both measure same underlying complexity but with different scaling (0-100 vs. grade level). Flesch-Kincaid preferred for educational interpretability.                                                                           |
| **Dale-Chall Readability Score**      | Spache Readability         | Both word-list-based vocabulary metrics. Dale-Chall uses 3,000-word list covering grades 4-16+ but provides poor discrimination at A1 level (all scores ≤4.9 = "4th grade or below"). Spache's 1,000-word list for grades 1-4 offers superior granularity (distinguishes 1st-2nd from 3rd-4th grade). |
| **Automated Readability Index (ARI)** | Coleman-Liau Index         | Both character-based metrics avoiding syllable parsing. ARI formula:`4.71 × (Chars/Words) + 0.5 × (Words/Sentences) - 21.43`. Coleman-Liau has stronger academic validation.                                                                                                                       |
| **Coleman-Liau Index**                | Flesch-Kincaid Grade Level | Character-based alternative to syllable-based metrics. Formula:`0.0588 × L - 0.296 × S - 15.8` (L=letters per 100 words, S=sentences per 100 words). Provides similar sentence structure information to Flesch-Kincaid without additional insights.                                                |

#### Not Relevant for Use Case

| Metric                          | Formula/Basis                                                         | Reason for Exclusion                                                                                                                                                          |
| ------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Linsear Write Formula** | Easy words (≤2 syllables) + Hard words (≥3 syllables) weighted by 3 | Designed specifically for technical writing and instructional manuals. Not appropriate for conversational chatbot responses typical of language learning interactions.        |
| **McAlpine EFLAW**        | (Words + Miniwords) / Sentences (Miniwords = 3+ syllables)            | Designed for broadcast and auditory content ("Easy Listening Formula for Auditory Writing"). This thesis focuses on text-based chatbot interactions without audio components. |

#### Component Metrics (Not Standalone)

| Metric                       | Used In                     | Reason for Exclusion                                                                                                         |
| ---------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Sentence Count**     | All grade-level indices     | Component metric calculated internally by formulas. Highly prompt-dependent; doesn't provide independent complexity insight. |
| **Character Count**    | ARI, Coleman-Liau           | Component metric for character-based formulas. Redundant with word-level metrics.                                            |
| **Syllable Count**     | Flesch-Kincaid, Gunning Fog | Component metric for syllable-based formulas. Total syllables less informative than syllables-per-word ratio.                |
| **Polysyllable Count** | SMOG, Gunning Fog           | Component metric for polysyllable-focused formulas. Already captured in SMOG and Gunning Fog calculations.                   |
| **Monosyllable Count** | Various                     | Inverse of polysyllable count. Redundant information; doesn't add independent insight.                                       |

#### Derived Metrics (Linear Transformations)

| Metric                                   | Derived From                   | Reason for Exclusion                                                                                                                                                                 |
| ---------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Reading Time (seconds/minutes)** | Word Count ÷ 200 words/minute | Linear transformation of word count. Provides no additional statistical information. Word count already included as secondary statistic. Formula:`Reading Time = Word Count / 200` |

### Metric Coverage Analysis

The final 4 primary metrics provide comprehensive coverage across multiple dimensions:

| Complexity Dimension            | Metrics Covering This Dimension |
| ------------------------------- | ------------------------------- |
| **Sentence Structure**    | Flesch-Kincaid, SMOG            |
| **Syllabic Complexity**   | Flesch-Kincaid, Gunning Fog     |
| **Polysyllabic Words**    | Gunning Fog, SMOG               |
| **Vocabulary Difficulty** | Spache                          |
| **Primary-Grade Focus**   | Spache                          |

### Comparison: Original vs. Final Metric Set

| Aspect                        | Original Set (18 metrics)                                                             | Final Set (4+2)                          | Benefit                               |
| ----------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------- |
| **Grade Level Indices** | 6 (FK, Gunning Fog, SMOG, ARI, Coleman-Liau, Dale-Chall)                              | 3 (FK, Gunning Fog, SMOG)                | Eliminated character-based redundancy |
| **Readability Scores**  | 4 (Flesch Reading Ease, Linsear Write, Spache, McAlpine EFLAW)                        | 1 (Spache)                               | Focused on A1-relevant metric         |
| **Text Statistics**     | 7 (sentences, words, chars, syllables, polysyllables, monosyllables, difficult words) | 2 (words, difficult words)               | Removed component metrics             |
| **Reading Time**        | 2 (seconds, minutes)                                                                  | 0 (deprecated)                           | Eliminated linear transformation      |
| **Redundancy**          | High (multiple overlapping metrics)                                                   | Zero (each metric unique)                | Clearer interpretation                |
| **A1 Discrimination**   | Mixed (some metrics poor at A1 level)                                                 | Excellent (all calibrated for beginners) | Better sensitivity                    |
| **Statistical Clarity** | 18 dependent variables                                                                | 4 dependent variables                    | Focused MANOVA analysis               |

### Historical Context

The original metric set was designed to be comprehensive, capturing all major readability formulas from the literature. However, analysis revealed:

1. **Redundancy Issues**: Multiple metrics measuring the same underlying construct (e.g., Flesch Reading Ease vs. Flesch-Kincaid)
2. **Discrimination Problems**: Some metrics (Dale-Chall) provided poor granularity at A1 level
3. **Relevance Concerns**: Metrics designed for technical writing (Linsear Write) or audio (McAlpine EFLAW) not appropriate for conversational text
4. **Statistical Complexity**: 18 metrics made interpretation difficult and increased multiple comparison issues

The streamlined set addresses all these issues while maintaining comprehensive coverage of text complexity dimensions relevant to A1 English learners.

### Validation of Metric Selection

The final 4 primary metrics were validated against three criteria:

1. **Non-Redundancy**: Correlation analysis confirmed each metric captures unique variance
2. **A1 Relevance**: All metrics provide meaningful discrimination at beginner proficiency levels
3. **Intervention Sensitivity**: Metrics responsive to both vocabulary weighting and prompting interventions

This rigorous selection process ensures the experimental framework focuses on meaningful, interpretable metrics that directly address the research questions.

---

**End of Appendix**
