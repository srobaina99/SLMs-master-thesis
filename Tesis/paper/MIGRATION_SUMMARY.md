# Mean to Median Migration - Summary of Changes

## Overview
Successfully migrated all numeric values in `slm_complexity_control.tex` from mean-based statistics to median-based statistics using data from `combined_summary_MEDIAN.json`.

## Files Modified
1. **slm_complexity_control.tex** - Main paper with all values updated
2. **MEDIAN_MIGRATION_VERIFICATION.md** - Comprehensive verification report
3. **MIGRATION_SUMMARY.md** - This summary document

## Changes by Section

### Abstract (Line 23)
- **Percentage range**: 28-63% → **8-67%**
- **Qwen2 FK increase**: +33.5% → **+11%**
- **TinyLlama FK reduction**: -17.2% → **-25%** (changed to Qwen3 as the positive example)
- **Changed from**: "from negative (Qwen2: +33.5% FK) to positive (TinyLlama: -17.2% FK)"
- **Changed to**: "from negative (Qwen2: +11% FK) to positive (Qwen3: -25% FK)"

### Results Section

#### Table 1a: Aggregate Primary Metrics (Lines 269-272)
All values updated to medians:
- Control: FK 8.95→9.13, GF 11.06→10.98, SMOG 11.37→11.55, Spa 5.08→5.08
- Weighting: FK 9.11→9.47, GF 11.05→11.33, SMOG 11.04→11.21, Spa 5.13→5.31
- Prompting: FK 5.60→4.84*, GF 6.99→6.09, SMOG 8.34→8.16, Spa 3.60→3.29
- Both: FK 5.59*→4.93*, GF 7.29→6.72, SMOG 8.00→8.68, Spa 3.68→3.54

**Caption change**: Now both "Prompting" and "Both" meet A1 target (FK ≤ 5.0)

#### Table 1b: Secondary Metrics (Lines 286-289)
- Control: Words 82.3→83.5, Diff 15.7→15.0
- Weighting: Words 90.6→89.5, Diff 15.5→14.5
- Prompting: Words 69.2→63.5, Diff 6.2→4.0
- Both: Words 73.1→60.5, Diff 5.8→4.0

#### Aggregate Analysis Text (Line 261)
- **Reduction range**: "28-37%" → **"27-47%"**
- **Word count increase**: "10%" → **"7%"**

#### Aggregate Trade-off (Line 297)
- **MAJOR CHANGE**: "5.7% more words but 6.7% fewer difficult words" → **"5% fewer words with no change in difficult word count"**
- **Difficult word reduction**: "60-63%" → **"73%"**

#### Section Introduction (Line 311)
- **FK reduction**: "37%" → **"47%"**

#### Table 2: Model-Specific Deltas (Lines 319-345)
Complete table updated with median values and recalculated deltas:

**Phi3:**
- FK: 3.88→3.41, Δ: -0.35→-0.83, Δ%: -8.3%→-19.6%
- Words: 66.5→66.0, Δ: -1.5→0.0, Δ%: -2.2%→0.0%
- Difficult: 3.3→3.0, Δ: -0.3→0.0, Δ%: -9.2%→0.0%

**Qwen2:**
- FK: 8.02→6.34, Δ: +2.01→+0.62, Δ%: +33.5%→+10.8%
- Words: 75.9→61.0, Δ: +7.5→+3.0, Δ%: +10.9%→+5.2%
- Difficult: 4.8→5.0, Δ: -1.0→0.0, Δ%: -17.2%→0.0%

**Qwen3:**
- FK: 3.50→2.80, Δ: -0.29→-0.92, Δ%: -7.7%→-24.7%
- Words: 50.4→26.0, Δ: -8.7→-12.0, Δ%: -14.7%→-31.6%
- Difficult: 5.1→2.0, Δ: -0.5→0.0, Δ%: -9.4%→0.0%

**TinyLlama:**
- FK: 6.95→5.98, Δ: -1.44→-1.20, Δ%: -17.2%→-16.7%
- GF: 8.78→9.0, Δ: -1.13→+0.55, Δ%: -11.4%→+6.5%
- Words: 99.5→124.0, Δ: +18.4→+50.0, Δ%: +22.7%→+67.6%

#### Model-Specific Interpretations (Lines 356-362)
**Phi3**: Changed from "minimal complementary effects" to "positive complementary effects" (20% FK reduction)

**Qwen2**: 
- FK increase: 34%→11%
- Difficult word reduction: 17%→0%
- Changed from "catastrophic" to "modest" compensatory verbosity

**Qwen3**: 
- FK reduction: 7.7%→25% (much stronger)
- Word count reduction: 14.7%→32% (much more dramatic)
- Now described as "strong positive interaction"

**TinyLlama**:
- FK reduction: 17.2%→16.7% (similar)
- Word count increase: 22.7%→68% (MUCH more severe)
- GF now shows INCREASE (+6.5%) instead of decrease

#### Aggregate Trade-off Analysis (Line 366)
- **Complete rewrite** to reflect: 5% fewer words (not more), no change in difficult words
- **Classification changed**: Phi3 now "Beneficial" (was "Neutral")

### Discussion Section

#### Line 385 (Instruction-Following)
- "5.7% more words, 6.7% fewer difficult words" → **"5% fewer words, no change in difficult words"**

#### Line 389 (Architectural Implications)
- Range: "Qwen2: +33.5% FK to TinyLlama: -17.2% FK" → **"Qwen2: +11% FK to Qwen3: -25% FK"**

#### Line 391 (Instruction-following capability)
- Phi3 now shows "positive benefit" (20% FK reduction) instead of "minimal benefit"

#### Line 393 (Training data composition)
- Qwen3 weighting effect: 16.8%→9% FK reduction

#### Line 399 (Compensatory Verbosity)
- Word count increase: 10%→7%

#### Line 401 (Compensatory verbosity continued)
- TinyLlama verbosity: Removed "modest (Phi3, Qwen2)" examples
- TinyLlama word count increase: Implicit 22.7%→68%

### Conclusion Section

#### Line 423
- "5.7% more words, 6.7% fewer difficult words" → **"5% fewer words, no change in difficult words"**
- Changed from "do not synergize in most architectures" to "synergize only in specific architectures"

## Key Interpretive Changes

### 1. Aggregate Trade-off REVERSES
The most significant finding: Combined intervention now REDUCES verbosity rather than increasing it.
- **Mean narrative**: Adding weighting increases words but reduces difficult words
- **Median narrative**: Adding weighting reduces words with no effect on difficult words

### 2. Qwen3 Emerges as Clear Winner
- FK reduction improves from -7.7% to -24.7%
- Word count reduction improves from -14.7% to -31.6%
- Now the strongest positive example in abstract and throughout paper

### 3. Qwen2 Negative Interaction Weakens
- FK increase reduces from +33.5% to +10.8%
- Still negative but much less "catastrophic"

### 4. TinyLlama Compensatory Verbosity Much Worse
- Word count increase: +22.7% → +67.6%
- Compensatory verbosity is now 3x more severe

### 5. Phi3 Shows Meaningful Positive Interaction
- FK reduction: -8.3% → -19.6%
- Reclassified from "Neutral" to "Beneficial"

### 6. Difficult Word Effects Disappear
- At the median, all models show ZERO change in difficult words when combining interventions
- This is because median values are identical between "Prompting" and "Both" for all models

## Statistical Robustness

The median-based analysis provides:
1. **Greater robustness to outliers**: Extreme values don't skew results
2. **More representative central tendency**: Better reflects typical model behavior
3. **Clearer architectural differences**: Heterogeneity between models is more pronounced
4. **Stronger narrative**: Qwen3's superiority is more evident

## Verification Status

✅ All 16 table values in Table 1a verified
✅ All 8 table values in Table 1b verified
✅ All 24 model-specific values in Table 2 verified
✅ All 24 delta calculations verified
✅ All 24 percentage calculations verified
✅ 20+ inline text percentages recalculated and verified
✅ Abstract percentage range recalculated from effect_sizes
✅ No LaTeX linter errors
✅ All interpretations updated to match new evidence

## Files for Review

1. **slm_complexity_control.tex** - Updated paper
2. **MEDIAN_MIGRATION_VERIFICATION.md** - Detailed verification with all calculations
3. **MIGRATION_SUMMARY.md** - This summary

## Next Steps

The paper is now fully migrated to median-based statistics. All values have been verified against the JSON source. The interpretations have been updated to reflect the new findings, which strengthen the paper's core argument about architectural heterogeneity while providing a more robust statistical foundation.

