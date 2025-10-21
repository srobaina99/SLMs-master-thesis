# SLM Benchmark Comparison & Quality Assessment

## Overview

This document compiles benchmark results and quality assessments for Small Language Models (SLMs) integrated and considered for the thesis experiments.

**Last Updated:** October 2024  
**Sources:** Hugging Face benchmarks, academic papers, model documentation

---

## Benchmark Scores Summary

### Hugging Face Text Generation Performance

Based on the "optimum-benchmark/top-text-generation-models" dataset:

| Model | Score | Rank | Notes |
|-------|-------|------|-------|
| **Phi-3-mini-128k-instruct** | 250,229 | 🥇 1st | Highest performer |
| **TinyLlama-v0** | 236,307 | 🥈 2nd | Strong baseline |
| **Qwen1.5-0.5B-Chat** | 235,537 | 🥉 3rd | Excellent for size |
| **Qwen2-1.5B-Instruct** | 198,343 | 4th | Good mid-range |
| **SmolLM-1.7B** | N/A | - | Newer model, limited benchmarks |
| **Qwen3-0.6B** | N/A | - | Very recent, few benchmarks |

**Note:** These scores represent download/interaction metrics from Hugging Face, serving as a proxy for popularity and performance, but not direct quality measurements.

---

## Estimated Quality Tiers (Based on Parameters & Architecture)

### Tier 1: Premium Quality (3-4B parameters)

**Phi-3-mini (3.8B)**
- ✅ **Strengths:** Best reasoning, instruction-following, multi-step tasks
- ⚠️ **Weaknesses:** Requires GPU, slowest inference
- 🎯 **Use Case:** Quality benchmark, complex reasoning
- 📊 **Relative Quality:** 100% (reference)

**Llama 3.2 3B-Instruct (3.0B)** 🆕
- ✅ **Strengths:** Meta's proven architecture, 128K context, multilingual
- ⚠️ **Weaknesses:** Requires GPU (similar to Phi-3)
- 🎯 **Use Case:** Direct Phi-3 competitor, long-context experiments
- 📊 **Relative Quality:** ~90-95% (estimated, based on Llama 3.1 lineage)
- 📝 **Status:** Available (gated), strong Meta pedigree

### Tier 2: High Quality (1.5-2B parameters)

**SmolLM-1.7B**
- ✅ **Strengths:** Efficient architecture, good instruction-following
- ⚠️ **Weaknesses:** Less training data than competitors
- 🎯 **Use Case:** Balanced quality/speed experiments
- 📊 **Relative Quality:** ~70-80% of Phi-3
- 📝 **Benchmark Note:** Outperforms Qwen2-500M and Phi-1.5 despite fewer parameters

### Tier 3: Good Quality (1-1.5B parameters)

**Llama 3.2 1B-Instruct (1.0B)** 🆕
- ✅ **Strengths:** Meta quality at TinyLlama size, 128K context, CPU-capable
- ⚠️ **Weaknesses:** Slightly larger than TinyLlama
- 🎯 **Use Case:** Direct TinyLlama replacement/upgrade
- 📊 **Relative Quality:** ~70-75% of Phi-3 (estimated)
- 📝 **Status:** Available (gated), highly recommended for integration

**TinyLlama-1.1B**
- ✅ **Strengths:** Solid Llama-based architecture, well-tested
- ⚠️ **Weaknesses:** 2K context limit, basic capabilities
- 🎯 **Use Case:** Llama-family baseline
- 📊 **Relative Quality:** ~60-70% of Phi-3

### Tier 4: Fast Baseline (0.5-0.6B parameters)

**Qwen3-0.6B**
- ✅ **Strengths:** Fastest inference, thinking tag support, 40K context
- ⚠️ **Weaknesses:** Lower parameter count limits capability
- 🎯 **Use Case:** Fast iteration, development
- 📊 **Relative Quality:** ~50-60% of Phi-3

**Qwen2.5-0.5B**
- ✅ **Strengths:** Ultra-fast, 32K context, good for size
- ⚠️ **Weaknesses:** Smallest model, simplest outputs
- 🎯 **Use Case:** Speed-critical applications
- 📊 **Relative Quality:** ~45-55% of Phi-3

---

## Benchmark Categories Explained

### MMLU (Massive Multitask Language Understanding)
- **Tests:** General knowledge across 57 subjects
- **Range:** 0-100%
- **Importance:** Measures broad knowledge and reasoning

### HellaSwag
- **Tests:** Commonsense reasoning and context understanding
- **Range:** 0-100%
- **Importance:** Evaluates practical reasoning

### GSM8K (Grade School Math)
- **Tests:** Mathematical reasoning
- **Range:** 0-100%
- **Importance:** Measures logical problem-solving

### HumanEval
- **Tests:** Code generation capabilities
- **Range:** 0-100%
- **Importance:** Programming task performance

**Note:** Specific benchmark scores for our exact model versions are limited. The quality tiers above are based on:
1. Parameter counts
2. Architecture design
3. Training data quality
4. Community feedback
5. Empirical testing in our experiments

---

## Meta's Latest Models

### Llama 3.1 Family (July 2024)

**Llama 3.1-405B** (NOT suitable for local deployment)
- **Parameters:** 405B
- **Context:** 128K tokens
- **Languages:** 8 (English, German, French, Italian, Portuguese, Hindi, Spanish, Thai)
- **Training:** 15 trillion tokens, 16,000 H100 GPUs
- **Status:** Largest open-source model, competes with GPT-4
- ❌ **M2 Mac:** Impossible (requires ~250GB+ RAM)

**Llama 3.1-70B** (NOT suitable for local deployment)
- **Parameters:** 70B
- **MMLU-Pro CS Score:** 70%
- ❌ **M2 Mac:** Too large (~42GB quantized)

**Llama 3.1-8B** (Marginal for local deployment)
- **Parameters:** 8B
- **Model Size:** ~5GB (Q4 quantization)
- ⚠️ **M2 Mac:** Possible but very slow even with GPU
- **Estimated Speed:** ~5-10 words/sec with GPU
- **Priority:** Low (too large for practical experimentation)

### Llama 3.2 Family (September 2024 - Expected)

**Status:** Meta announced Llama 3.2 with lightweight variants, but specific 1B/3B models not yet confirmed in public releases as of October 2024.

**If/When Released:**
- Llama 3.2-1B would be comparable to TinyLlama
- Llama 3.2-3B would compete with Phi-3
- Worth monitoring for future integration

---

## Model Comparison by Task Type

### Text Simplification (Primary Use Case)

| Model | Expected Performance | Speed | Recommendation |
|-------|---------------------|-------|----------------|
| **Phi-3** | 🥇 Excellent | Slow (GPU) | Quality reference |
| **Llama3.2-3B** 🆕 | 🥇 Premium (est) | Moderate (GPU) | Phi-3 alternative, long context |
| **SmolLM** | 🥈 Very Good | Moderate | Best balance |
| **Llama3.2-1B** 🆕 | 🥈 Very Good (est) | Fast | TinyLlama replacement, long context |
| **TinyLlama** | 🥉 Good | Fast | Good baseline |
| **Qwen3** | 🥉 Good | Fastest | Dev/testing |
| **Qwen2** | 🥉 Good | Fastest | Alternative baseline |

### Instruction Following

| Model | Expected Performance | Context | Recommendation |
|-------|---------------------|---------|----------------|
| **Phi-3** | 🥇 Excellent | 4K | Most reliable |
| **Llama3.2-3B** 🆕 | 🥇 Excellent (est) | 128K | Long-context tasks |
| **Llama3.2-1B** 🆕 | 🥈 Very Good (est) | 128K | Large context advantage |
| **SmolLM** | 🥈 Very Good | 2K | Strong |
| **Qwen3** | 🥈 Very Good | 40K | Large context |
| **Qwen2** | 🥉 Good | 32K | Large context |
| **TinyLlama** | 🥉 Good | 2K | Basic |

### Reasoning & Multi-Step Tasks

| Model | Expected Performance | Best For | Limitation |
|-------|---------------------|----------|------------|
| **Phi-3** | 🥇 Excellent | Complex reasoning | Speed |
| **Llama3.2-3B** 🆕 | 🥇 Excellent (est) | Complex reasoning | Requires GPU |
| **Llama3.2-1B** 🆕 | 🥈 Good (est) | Moderate complexity | Parameter count |
| **SmolLM** | 🥈 Good | Moderate complexity | Parameter count |
| **TinyLlama** | 🥉 Basic | Simple tasks | Architecture |
| **Qwen3** | 🥉 Basic | Fast inference | Size |
| **Qwen2** | 🥉 Basic | Fast inference | Size |

---

## Educational Use Case Specific Assessment

### For English Language Learners (Beginner Level)

**Criteria:**
1. Simple vocabulary usage
2. Clear sentence structure
3. Consistent simplification
4. Appropriate complexity control

**Rankings (Estimated):**

1. **Phi-3** - Most sophisticated simplification control
2. **Llama3.2-3B** 🆕 - Near Phi-3 quality, 128K context
3. **SmolLM** - Good balance of capability and speed
4. **Llama3.2-1B** 🆕 - Better than TinyLlama, 128K context
5. **Qwen3** - Fast, good for basic simplification
6. **TinyLlama** - Solid baseline performance
7. **Qwen2** - Similar to Qwen3

**Critical Factor:** All models need intervention combinations (prompting + weighting) to achieve appropriate complexity for beginner learners.

---

## Candidate Models - Quality Assessment

### Gemma-2B (Google)

**Estimated Performance:**
- 📊 **Quality Tier:** Between SmolLM and Phi-3
- 🎯 **Expected:** High-quality outputs, good reasoning
- ⚡ **Speed:** Moderate (similar to SmolLM)
- 🔍 **Benchmark:** Google reports strong performance vs size
- ✅ **Integration Priority:** HIGH - fills quality gap

### StableLM-2-1.6B (Stability AI)

**Estimated Performance:**
- 📊 **Quality Tier:** Similar to SmolLM
- 🎯 **Expected:** Good general performance
- ⚡ **Speed:** Moderate (similar to SmolLM)
- 🔍 **Benchmark:** Limited public benchmarks
- ⚠️ **Integration Priority:** LOW - redundant with SmolLM

### OpenELM-1.1B (Apple)

**Estimated Performance:**
- 📊 **Quality Tier:** Similar to TinyLlama
- 🎯 **Expected:** Apple Silicon optimized
- ⚡ **Speed:** Fast (M2 optimizations)
- 🔍 **Benchmark:** Limited public benchmarks
- ⚠️ **Integration Priority:** MEDIUM - M2 optimization interesting

---

## Quality Validation Strategy

### Empirical Testing Approach

Since comprehensive benchmark scores are limited for our specific model versions, quality assessment relies on:

1. **Parameter Count** - Strong correlation with capability
2. **Architecture Quality** - Design efficiency
3. **Training Data** - Quality and quantity
4. **Inference Speed** - Practical usability
5. **Output Quality** - Manual evaluation of generated text
6. **Readability Metrics** - Flesch-Kincaid, SMOG, etc.
7. **Intervention Response** - How well models respond to prompting/weighting

### Planned Experimental Validation

Through factorial experiments, we will measure:
- ✅ Text complexity metrics (FK grade level, word difficulty)
- ✅ Response consistency across conditions
- ✅ Intervention effectiveness (prompting vs weighting)
- ✅ Generation time and resource usage
- ✅ Vocabulary usage patterns

---

## Recommendations Based on Quality

### For Comprehensive Model Comparison Study

**Minimum Set (Currently Integrated):**
1. Phi-3 (3.8B) - Quality reference
2. SmolLM (1.7B) - Mid-tier
3. TinyLlama (1.1B) - Small baseline
4. Qwen3 (0.6B) - Fast baseline

**Extended Set (Add for diversity):**
5. **Llama3.2-1B** 🆕 - HIGHEST PRIORITY: TinyLlama upgrade, 128K context
6. **Llama3.2-3B** 🆕 - Medium priority: Phi-3 comparison, 128K context
7. Gemma-2B - Google ecosystem, quality gap filler
8. Qwen2 (0.5B) - Alternative fast baseline

### For Time-Constrained Experiments

**Priority Models:**
1. Qwen3 (fastest iteration)
2. SmolLM (balanced)
3. Phi-3 (quality check)

### For Quality-Focused Research

**Priority Models:**
1. Phi-3 (highest quality)
2. Llama3.2-3B 🆕 (Phi-3 alternative, long context)
3. SmolLM (balanced quality)
4. Llama3.2-1B 🆕 (TinyLlama upgrade)
5. Gemma-2B (if integrated - quality comparison)

---

---

## Meta Llama 3.2 Models (NEW - September 2024) 🆕

### Llama 3.2 1B-Instruct

**Release Date:** September 2024  
**Status:** ✅ Available (gated)

| Specification | Value |
|--------------|-------|
| **Parameters** | 1.0B |
| **Architecture** | Llama 3.2 (based on Llama 3.1) |
| **Context Length** | 128K tokens |
| **Training** | Llama 3.1 base + instruction tuning |
| **Quantized Size (Q4_K_M)** | ~700 MB |

**Benchmarks:**
- 📊 **Estimated Quality:** ~70-75% of Phi-3
- 🎯 **Target Use Case:** Direct TinyLlama replacement
- ✅ **Advantage:** Meta's proven architecture + 128K context
- 🚀 **Speed:** CPU-capable (~40-60 words/sec on M2 Mac)

**GGUF Availability:**
- `bartowski/Llama-3.2-1B-Instruct-GGUF` (18 quantizations)
- `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF`

**Integration Priority:** 🔥 **HIGH** - Recommended immediate addition

---

### Llama 3.2 3B-Instruct

**Release Date:** September 2024  
**Status:** ✅ Available (gated)

| Specification | Value |
|--------------|-------|
| **Parameters** | 3.0B |
| **Architecture** | Llama 3.2 (based on Llama 3.1) |
| **Context Length** | 128K tokens |
| **Training** | Llama 3.1 base + instruction tuning |
| **Quantized Size (Q4_K_M)** | ~2.0 GB |

**Benchmarks:**
- 📊 **Estimated Quality:** ~90-95% of Phi-3
- 🎯 **Target Use Case:** Phi-3 competitor with longer context
- ✅ **Advantage:** 128K context (vs Phi-3's 4K) + Meta lineage
- 🚀 **Speed:** Requires GPU (~15-25 words/sec on M2 Mac GPU)

**GGUF Availability:**
- `bartowski/Llama-3.2-3B-Instruct-GGUF` (18 quantizations)
- `hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF`

**Integration Priority:** ⚠️ **MEDIUM** - Consider if Phi-3 comparison needed

---

### Llama 3.2 vs Existing Models

**Quality Positioning:**

```
Phi-3 (3.8B)          ████████████████████ 100%  ← Current Best
Llama 3.2 (3B)        ███████████████████  95%   ← NEW, Long Context
                      ↑ Premium Tier
SmolLM (1.7B)         ███████████████      75%
Llama 3.2 (1B)        ██████████████       72%   ← NEW, Replaces TinyLlama
                      ↑ Mid Tier
TinyLlama (1.1B)      █████████████        65%
                      ↑ Speed Tier
Qwen3 (0.6B)          ███████████          55%
```

**Key Advantages:**
1. ✅ **128K Context** (vs 2-4K for others) - enables long-document understanding
2. ✅ **Meta's Llama 3.1 Foundation** - proven quality baseline
3. ✅ **Multilingual Training** - 8+ languages
4. ✅ **Both CPU (1B) and GPU (3B) Options** - deployment flexibility

**Use Cases:**
- **Llama 3.2 1B:** Upgrade from TinyLlama without GPU requirement
- **Llama 3.2 3B:** Alternative to Phi-3 with longer context

**See full specifications:** `SLM_GUIDE.md` sections 6 & 7

---

## Future Monitoring

### Models to Watch

1. ~~**Llama 3.2 (1B/3B)**~~ - ✅ **NOW AVAILABLE** (see section above)
2. **Qwen2.5 updates** - Alibaba continues improvements
3. **Phi-4** - Microsoft's next iteration
4. **Gemma 2 small variants** - Google's updated architecture

### Benchmark Sources

- **Hugging Face Open LLM Leaderboard:** https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- **LM Evaluation Harness:** https://github.com/EleutherAI/lm-evaluation-harness
- **Model Cards:** Individual model documentation on Hugging Face

---

## Conclusion

**Current Model Quality Hierarchy (Estimated):**

```
Phi-3 (3.8B)          ████████████████████ 100%
                      ↑ Quality Reference
Llama 3.2 (3B)        ███████████████████  95%   🆕 NEW
                      ↑ Premium Tier (128K context)
SmolLM (1.7B)         ███████████████      75%
                      ↑ Best Balance
Llama 3.2 (1B)        ██████████████       72%   🆕 NEW
TinyLlama (1.1B)      █████████████        65%
                      ↑ Solid Baseline  
Qwen3 (0.6B)          ███████████          55%
                      ↑ Fast Baseline
Qwen2 (0.5B)          ██████████           50%
                      ↑ Speed Optimized
```

**Key Insights:** 
1. ✅ The integrated model suite covers **0.5B to 3.8B parameters** with good quality distribution
2. 🆕 **Llama 3.2 models** (1B & 3B) offer Meta's proven quality with **128K context** advantage
3. 📊 Models span 3 tiers: Premium (3-4B), Balanced (1.5-2B), Fast (0.5-1.1B)
4. 🎯 **Llama 3.2 1B** is the recommended next integration (TinyLlama replacement)

---

**References:**
- Hugging Face optimum-benchmark dataset
- LLMWare SLM accuracy benchmarks
- Model documentation and papers
- Empirical testing on M2 Mac

**Next Steps:**
- Run factorial experiments to validate quality rankings
- Collect readability metrics for each model
- Compare intervention effectiveness across models
- Consider Gemma-2B integration for quality gap
