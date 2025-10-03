# Text Readability Metrics Documentation

This document provides comprehensive explanations of all readability and complexity metrics used in the experimental framework for evaluating Small Language Model (SLM) outputs.

---

## Table of Contents

1. [Grade Level Indices](#grade-level-indices)
2. [Readability Scores](#readability-scores)
3. [Text Statistics](#text-statistics)
4. [Reading Time](#reading-time)
5. [Target Ranges for A1 English Learners](#target-ranges-for-a1-english-learners)
6. [Metric Selection Rationale](#metric-selection-rationale)

---

## Grade Level Indices

These metrics estimate the years of formal education required to understand a text on first reading. All report U.S. grade levels (e.g., 5.0 = 5th grade).

### 1. Flesch-Kincaid Grade Level

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

**For the experiment:** Lower values indicate the intervention successfully simplified model output.

---

### 2. Gunning Fog Index

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

**For the experiment:** Scores below 6 indicate very simple vocabulary appropriate for beginners. The Gunning Fog is particularly sensitive to long, complex words.

---

### 3. SMOG Index (Simple Measure of Gobbledygook)

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

**For the experiment:** SMOG Grade ≤7 is **TARGET for A1 learners**. This metric is reliable for short texts and less sensitive to sentence length than Flesch-Kincaid.

---

### 4. Automated Readability Index (ARI)

**What it measures:** Reading difficulty based on characters per word (not syllables) and sentence length. Designed for real-time computational evaluation.

**Formula:**
```
ARI = (4.71 × Characters/Words) + (0.5 × Words/Sentences) - 21.43
```

**Interpretation:**
- **1-5**: Elementary school - **TARGET for A1 learners**
- **6-8**: Middle school
- **9-12**: High school
- **13+**: College graduate

**For the experiment:** ARI is faster to compute (no syllable counting) and useful for comparing character-level complexity across models.

---

### 5. Coleman-Liau Index

**What it measures:** Reading difficulty based on character length rather than syllables. Correlates well with other grade-level metrics.

**Formula:**
```
CLI = (0.0588 × L) - (0.296 × S) - 15.8
```
Where:
- **L** = Average letters per 100 words
- **S** = Average sentences per 100 words

**Interpretation:**
- **1-5**: Elementary school - **TARGET for A1 learners**
- **6-8**: Middle school
- **9-12**: High school
- **13+**: College graduate

**For the experiment:** CLI is useful because it doesn't require syllable parsing, making it robust across different text types.

---

### 6. Dale-Chall Readability Score

**What it measures:** Reading difficulty based on a reference list of 3,000 "familiar" English words. Measures vocabulary difficulty directly.

**Formula:**
```
Raw Score = 0.1579 × (% Difficult Words) + 0.0496 × ASL
```
Where:
- **Difficult Words** = Words NOT on Dale-Chall list of 3,000 common words

**Interpretation:**
- **≤4.9**: 4th grade or below - **TARGET for A1 learners**
- **5.0-5.9**: 5th-6th grade
- **6.0-6.9**: 7th-8th grade
- **7.0-7.9**: 9th-10th grade
- **8.0-8.9**: 11th-12th grade
- **9.0-9.9**: College
- **10.0+**: College graduate

**For the experiment:** This metric directly measures vocabulary difficulty. Lower scores indicate use of common, familiar words - critical for weighted vocabulary intervention evaluation.

---

## Readability Scores

These metrics use different scales but all measure how easy text is to read.

### 7. Flesch Reading Ease

**What it measures:** Overall readability on a 0-100 scale (inverse of difficulty).

**Formula:**
```
Reading Ease = 206.835 - (1.015 × ASL) - (84.6 × ASW)
```

**Interpretation:**
- **90-100**: Very easy (5th grade) - **TARGET for A1 learners**
- **80-89**: Easy (6th grade) - **Acceptable for A1**
- **70-79**: Fairly easy (7th grade)
- **60-69**: Standard (8th-9th grade)
- **50-59**: Fairly difficult (10th-12th grade)
- **30-49**: Difficult (College)
- **0-29**: Very difficult (College graduate+)

**For the experiment:** Higher is better. This is the most widely recognized readability metric. Target ≥80 for A1 learners.

---

### 8. Linsear Write Formula

**What it measures:** Reading difficulty designed specifically for technical writing and instructional materials.

**Formula:**
```
1. Count easy words (≤2 syllables) and hard words (≥3 syllables) per 100 words
2. Score = (Easy Words + Hard Words × 3) / Sentences
3. If Score > 20: Grade = Score / 2
   Else: Grade = (Score - 2) / 2
```

**Interpretation:**
- **1-5**: Elementary school - **TARGET for A1 learners**
- **6-8**: Middle school
- **9-12**: High school
- **13+**: College

**For the experiment:** Particularly useful for evaluating instructional/conversational content typical of language learning interactions.

---

### 9. Spache Readability

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

**For the experiment:** Highly relevant for beginner learners. This metric uses a vocabulary list matched to early readers, making it ideal for evaluating ESL beginner materials.

---

### 10. McAlpine EFLAW (Easy Listening Formula for Auditory Writing)

**What it measures:** Readability for content meant to be heard/spoken rather than read. Designed for broadcast and conversational text.

**Formula:**
```
EFLAW = (Words + Miniwords) / Sentences
```
Where:
- **Miniwords** = Words with 3+ syllables

**Interpretation:**
- **<20**: Easy to listen to - **TARGET for A1 learners**
- **20-25**: Moderate difficulty
- **>25**: Difficult to follow aurally

**For the experiment:** Crucial for evaluating chatbot/conversational responses. Since language learning involves speaking and listening, this metric assesses how suitable model output is for spoken interaction.

---

## Text Statistics

Basic descriptive metrics that feed into readability formulas.

### 11. Sentence Count
**What it measures:** Total number of sentences.

**Interpretation:** More sentences in same text = shorter sentences = typically easier to read.

---

### 12. Word Count
**What it measures:** Total number of words.

**Interpretation:** For language learning responses, shorter is often better (less cognitive load). Track to ensure models provide concise, focused answers.

---

### 13. Character Count
**What it measures:** Total number of characters (including spaces).

**Interpretation:** Used in ARI and Coleman-Liau calculations. Longer average character count suggests complex vocabulary.

---

### 14. Syllable Count
**What it measures:** Total number of syllables.

**Interpretation:** Higher syllable count relative to word count indicates complex, multisyllabic vocabulary. For A1 learners, prefer monosyllabic and disyllabic words.

---

### 15. Polysyllable Count
**What it measures:** Words with 3 or more syllables.

**Interpretation:** 
- High polysyllable count = difficult vocabulary
- Target: Minimize polysyllables for A1 learners
- Critical for Gunning Fog and SMOG calculations

---

### 16. Monosyllable Count
**What it measures:** Words with exactly 1 syllable.

**Interpretation:**
- High monosyllable count = simple vocabulary
- Target: Maximize monosyllables for A1 learners
- Examples: "cat", "dog", "run", "big", "red"

---

### 17. Difficult Words
**What it measures:** Words classified as "difficult" by the `textstat` library (typically words not on common word lists or with 3+ syllables not meeting exception criteria).

**Interpretation:**
- Lower is better for A1 learners
- Directly measures vocabulary accessibility
- **Key metric for evaluating weighted vocabulary intervention**

---

## Reading Time

### 18. Reading Time (Seconds/Minutes)
**What it measures:** Estimated time to read text aloud at average speaking speed (~200 words per minute).

**Formula:**
```
Reading Time = Word Count / 200 (in minutes)
```

**Interpretation:**
- Shorter reading time = more concise response
- For chatbot responses, target 5-15 seconds (natural conversation pace)
- Longer reading time may indicate overly verbose responses

---

## Target Ranges for A1 English Learners

Based on Common European Framework of Reference (CEFR) A1 level, target ranges for successful interventions:

| Metric | Target Range | Interpretation |
|--------|-------------|----------------|
| **Flesch-Kincaid Grade** | ≤5.0 | Elementary level or below |
| **Gunning Fog** | ≤6.0 | Sixth grade or below |
| **SMOG Index** | ≤7.0 | Junior high or below |
| **ARI** | ≤5.0 | Elementary level |
| **Coleman-Liau** | ≤5.0 | Elementary level |
| **Dale-Chall** | ≤4.9 | 4th grade or below |
| **Flesch Reading Ease** | ≥80 | Easy to very easy |
| **Linsear Write** | ≤5.0 | Elementary level |
| **Spache Readability** | ≤4.0 | Primary grades |
| **McAlpine EFLAW** | <20 | Easy listening |
| **Difficult Words** | Minimize | Prefer common vocabulary |
| **Polysyllable Count** | Minimize | Prefer 1-2 syllable words |
| **Monosyllable Count** | Maximize | Use simple words |

---

## Metric Selection Rationale

### Why Multiple Metrics?

Different readability formulas capture different aspects of text complexity:

1. **Sentence Structure**: Flesch-Kincaid, ARI, Coleman-Liau emphasize sentence length
2. **Word Complexity**: Gunning Fog, SMOG, Linsear Write focus on polysyllabic words
3. **Vocabulary Difficulty**: Dale-Chall, Spache measure use of uncommon words
4. **Spoken Language**: McAlpine EFLAW evaluates conversational suitability
5. **Overall Readability**: Flesch Reading Ease provides general accessibility score

### Complementary Strengths

- **Character-based** (ARI, Coleman-Liau): Don't require syllable parsing, computationally efficient
- **Syllable-based** (Flesch-Kincaid, Gunning Fog): Capture phonological complexity
- **Word list-based** (Dale-Chall, Spache): Directly measure vocabulary level
- **Polysyllable-focused** (SMOG, Gunning Fog): Identify complex terminology

### Experimental Validation

Using multiple metrics enables:
- **Cross-validation** of findings (consistent improvement across metrics = robust intervention)
- **Identification of trade-offs** (e.g., prompting may reduce word difficulty but increase sentence length)
- **Understanding of mechanisms** (weighted vocab primarily affects Dale-Chall/Spache; prompting affects sentence structure metrics)
- **Avoidance of single-metric bias** (each formula has strengths/weaknesses)

### Statistical Analysis

The factorial design (4 models × 4 configs × N prompts) with multiple metrics enables:
- **ANOVA/MANOVA**: Test intervention effects across metrics
- **Interaction analysis**: Determine if weighting + prompting effects are additive or synergistic
- **Model comparison**: Identify which SLMs naturally produce simpler output
- **Metric correlation**: Understand relationships between different complexity dimensions

---

## References

- Flesch, R. (1948). "A new readability yardstick." *Journal of Applied Psychology*, 32(3), 221-233.
- Kincaid, J.P., et al. (1975). "Derivation of new readability formulas for Navy enlisted personnel." *Research Branch Report 8-75*.
- Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
- McLaughlin, G.H. (1969). "SMOG grading: A new readability formula." *Journal of Reading*, 12(8), 639-646.
- Dale, E., & Chall, J.S. (1948). "A formula for predicting readability." *Educational Research Bulletin*, 27(1), 11-28.
- Coleman, M., & Liau, T.L. (1975). "A computer readability formula designed for machine scoring." *Journal of Applied Psychology*, 60(2), 283-284.

---

## Usage in Code

All metrics are calculated using the `TextEvaluator` class in `src/evaluation/text_complexity/text_evaluator.py`, which wraps the `textstat` library:

```python
from src.evaluation.text_complexity.text_evaluator import TextEvaluator

evaluator = TextEvaluator()
text = "The cat sat on the mat."

# Get all metrics
analysis = evaluator.evaluate_text_comprehensive(text)

# Access specific metric categories
grade_levels = evaluator.get_grade_level_indices(text)
readability = evaluator.get_readability_scores(text)
statistics = evaluator.get_text_statistics(text)
reading_time = evaluator.get_reading_time(text)
```

Metrics are automatically calculated for all experiment results and exported to CSV files for statistical analysis.

