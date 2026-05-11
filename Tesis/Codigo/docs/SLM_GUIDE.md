# Small Language Models (SLMs) Selection Guide

## Overview

This guide provides comprehensive specifications for Small Language Models (SLMs) suitable for local deployment on Apple M2 Mac hardware, with focus on educational applications requiring controlled text complexity.

**Last Updated:** October 2024

---

## Currently Integrated Models

*Models listed by size (smallest to largest)*

### 1. Qwen2.5-0.5B-Instruct

**Status:** ✅ Integrated and tested

| Specification | Value |
|--------------|-------|
| **Parameters** | 0.5B (500M) |
| **Architecture** | Qwen2.5 (Alibaba) |
| **Context Length** | 32K tokens |
| **Quantization** | Q4_0 |
| **Model File Size** | 409 MB |
| **Expected Memory** | ~600 MB |
| **CPU Speed** | ~80-100 words/sec |
| **GPU Speed** | ~80-100 words/sec (no benefit) |
| **Load Time** | 0.8s |
| **Template Format** | ChatML |

**Deployment:**
- ✅ **CPU-only** (optimal)
- ❌ GPU offloading unnecessary
- 🎯 **Best for:** Fast baseline experiments

**Recommendation:** Excellent for quick iterations. Already optimal on CPU.

---

### 2. Qwen3-0.6B

**Status:** ✅ Integrated and tested

| Specification | Value |
|--------------|-------|
| **Parameters** | 0.6B (600M) |
| **Architecture** | Qwen3 (Alibaba) |
| **Context Length** | 40K tokens |
| **Quantization** | Q4_0 |
| **Model File Size** | 409 MB |
| **Expected Memory** | ~600 MB |
| **CPU Speed** | ~80-100 words/sec |
| **GPU Speed** | ~80-100 words/sec (no benefit) |
| **Load Time** | 0.8s |
| **Template Format** | ChatML |
| **Special Feature** | Thinking tags support (`<think>`) |

**Deployment:**
- ✅ **CPU-only** (optimal)
- ❌ GPU offloading unnecessary
- 🎯 **Best for:** Fast experiments with reasoning capability

**Recommendation:** Fastest model in the suite. Ideal for development and testing.

---

### 3. TinyLlama-1.1B-Chat

**Status:** ✅ Integrated and tested

| Specification | Value |
|--------------|-------|
| **Parameters** | 1.1B |
| **Architecture** | Llama2-based |
| **Context Length** | 2K tokens |
| **Quantization** | Q4_0 |
| **Model File Size** | 608 MB |
| **Expected Memory** | ~800 MB |
| **CPU Speed** | ~60-80 words/sec |
| **GPU Speed** | ~60-80 words/sec (minimal benefit) |
| **Load Time** | 0.9s |
| **Template Format** | Llama2-style |

**Deployment:**
- ✅ **CPU-only** (optimal)
- ⚠️ GPU offloading provides <1.5x speedup
- 🎯 **Best for:** Llama-family comparison

**Recommendation:** Good middle-ground model. CPU sufficient for all use cases.

---

### 4. SmolLM-1.7B-Instruct

**Status:** ❌ Not integrated (benchmarked but no active wrapper in `src/framework/models/`)

| Specification | Value |
|--------------|-------|
| **Parameters** | 1.7B |
| **Architecture** | SmolLM (HuggingFace) |
| **Context Length** | 2K tokens |
| **Quantization** | Q4_K_M |
| **Model File Size** | 1.0 GB |
| **Expected Memory** | ~1.2 GB |
| **CPU Speed** | ~10-15 words/sec |
| **GPU Speed** | ~10-15 words/sec (1.3x speedup) |
| **Load Time** | 1.0s (CPU), 0.1s (GPU) |
| **Template Format** | ChatML |
| **Design Focus** | Efficient on-device deployment |

**Deployment:**
- ✅ **CPU-only** (acceptable)
- ⚠️ GPU offloading provides marginal benefit (1.3x)
- 🎯 **Best for:** Size/quality balance experiments

**Recommendation:** Works well on CPU. GPU optional but not essential.

---

### 5. Phi-3-mini-4k-Instruct

**Status:** ✅ Integrated and tested (GPU optimized)

| Specification | Value |
|--------------|-------|
| **Parameters** | 3.8B |
| **Architecture** | Phi-3 (Microsoft) |
| **Context Length** | 4K tokens |
| **Quantization** | Q4 |
| **Model File Size** | 2.2 GB |
| **Expected Memory** | ~2.5 GB |
| **CPU Speed** | ~0.1 words/sec ❌ |
| **GPU Speed** | ~18 words/sec ✅ (73x faster!) |
| **Load Time** | 2.4s (CPU), 5.4s (GPU) |
| **Template Format** | Phi-3 custom (`<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`) |
| **Design Focus** | Strong reasoning capabilities |

**Deployment:**
- ❌ **CPU-only** too slow (unusable)
- ✅ **GPU offloading REQUIRED** (`n_gpu_layers=-1`)
- 🎯 **Best for:** Quality comparison, reasoning tasks

**Recommendation:** MUST use GPU offloading. Provides highest quality output but slowest even with GPU.

---

## Candidate Models for Future Integration

### 6. Llama-3.2-1B-Instruct

**Status:** 🔥 **HIGH PRIORITY** - Recommended for integration

| Specification | Value |
|--------------|-------|
| **Parameters** | 1.0B |
| **Architecture** | Llama 3.2 (Meta, September 2024) |
| **Context Length** | 128K tokens |
| **Quantization** | Q4_K_M (estimated) |
| **Model File Size** | ~700 MB (estimated) |
| **Expected Memory** | ~900 MB-1.2 GB |
| **CPU Speed** | ~40-60 words/sec (estimated) |
| **GPU Speed** | ~40-60 words/sec (marginal benefit) |
| **Load Time** | ~1.0s (estimated) |
| **Template Format** | Llama 3 (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) |
| **Special Feature** | 128K context window (64x larger than TinyLlama) |

**Deployment:**
- ✅ **CPU-only** (optimal for this size)
- ❌ GPU offloading unnecessary (similar to SmolLM)
- 🎯 **Best for:** Direct TinyLlama replacement with better quality

**Recommendation:** Highest priority candidate. Meta's proven Llama 3.1 foundation at TinyLlama size with massive context upgrade.

**GGUF Availability:**
- `bartowski/Llama-3.2-1B-Instruct-GGUF` (18 quantizations)
- `hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF`

---

### 7. Llama-3.2-3B-Instruct

**Status:** ⚠️ **MEDIUM PRIORITY** - Consider for Phi-3 comparison

| Specification | Value |
|--------------|-------|
| **Parameters** | 3.0B |
| **Architecture** | Llama 3.2 (Meta, September 2024) |
| **Context Length** | 128K tokens |
| **Quantization** | Q4_K_M (estimated) |
| **Model File Size** | ~2.0 GB (estimated) |
| **Expected Memory** | ~2.2-2.5 GB |
| **CPU Speed** | ~2-5 words/sec (too slow) |
| **GPU Speed** | ~15-25 words/sec (estimated) |
| **Load Time** | ~2.5s (CPU), ~5.0s (GPU, estimated) |
| **Template Format** | Llama 3 (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) |
| **Special Feature** | 128K context window (32x larger than Phi-3) |

**Deployment:**
- ❌ **CPU-only** too slow (unusable)
- ✅ **GPU offloading REQUIRED** (`n_gpu_layers=-1`)
- 🎯 **Best for:** Head-to-head comparison with Phi-3

**Recommendation:** Consider if Phi-3 comparison needed. Offers 32x larger context at similar performance. May be redundant with Phi-3 already integrated.

**GGUF Availability:**
- `bartowski/Llama-3.2-3B-Instruct-GGUF` (18 quantizations)
- `hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF`

---

### 8. Gemma-2B-Instruct

**Status:** 🔄 Candidate

| Specification | Value |
|--------------|-------|
| **Parameters** | 2.0B |
| **Architecture** | Gemma (Google) |
| **Context Length** | 8K tokens |
| **Quantization** | Q4_K_M (estimated) |
| **Model File Size** | ~1.3 GB (estimated) |
| **Expected Memory** | ~1.5 GB |
| **CPU Speed** | ~15-25 words/sec (estimated) |
| **GPU Speed** | ~20-30 words/sec (estimated) |
| **Template Format** | Gemma-specific |

**Deployment Recommendation:**
- ✅ **CPU-capable** (likely acceptable)
- ⚠️ Test GPU benefit (size suggests marginal)
- 🎯 **Best for:** Google ecosystem comparison

**Integration Priority:** Medium - Good for diversity but similar to SmolLM

---

### 7. StableLM-2-1.6B

**Status:** 🔄 Candidate

| Specification | Value |
|--------------|-------|
| **Parameters** | 1.6B |
| **Architecture** | StableLM-2 (Stability AI) |
| **Context Length** | 4K tokens |
| **Quantization** | Q4_K_M (estimated) |
| **Model File Size** | ~1.0 GB (estimated) |
| **Expected Memory** | ~1.2 GB |
| **CPU Speed** | ~12-18 words/sec (estimated) |
| **GPU Speed** | ~15-20 words/sec (estimated) |
| **Template Format** | StableLM-specific |

**Deployment Recommendation:**
- ✅ **CPU-capable** (similar to SmolLM)
- ⚠️ GPU likely minimal benefit
- 🎯 **Best for:** Stability AI ecosystem comparison

**Integration Priority:** Low - Very similar specs to SmolLM

---

### 8. Mistral-7B-Instruct

**Status:** ⚠️ Not recommended

| Specification | Value |
|--------------|-------|
| **Parameters** | 7.0B |
| **Architecture** | Mistral (Mistral AI) |
| **Context Length** | 32K tokens |
| **Quantization** | Q4_K_M |
| **Model File Size** | ~4.1 GB |
| **Expected Memory** | ~5 GB |
| **CPU Speed** | ~0.05 words/sec ❌ |
| **GPU Speed** | ~5-10 words/sec (estimated) |
| **Load Time** | ~10s |

**Deployment Recommendation:**
- ❌ **CPU-only** impossible (too slow)
- ⚠️ **GPU required** but still slow
- 🎯 **Issue:** Too large for M2 Mac research workflow

**Integration Priority:** None - Too large for practical experimentation

---

### 9. OpenELM-1.1B

**Status:** 🔄 Candidate

| Specification | Value |
|--------------|-------|
| **Parameters** | 1.1B |
| **Architecture** | OpenELM (Apple) |
| **Context Length** | 2K tokens |
| **Quantization** | Q4_0 (estimated) |
| **Model File Size** | ~650 MB (estimated) |
| **Expected Memory** | ~850 MB |
| **CPU Speed** | ~50-70 words/sec (estimated) |
| **GPU Speed** | ~50-70 words/sec (estimated) |
| **Template Format** | Unknown |

**Deployment Recommendation:**
- ✅ **CPU-capable** (Apple Silicon optimized)
- ⚠️ GPU likely unnecessary
- 🎯 **Best for:** Apple ecosystem, M2 optimization

**Integration Priority:** Medium - Interesting for Apple Silicon focus

---

## Model Comparison Matrix

### By Size Category

| Size Class | Models | CPU Viable | GPU Benefit | Use Case |
|------------|--------|------------|-------------|----------|
| **Tiny** (<1GB) | Qwen2 (0.5B), Qwen3 (0.6B) | ✅ Excellent | ❌ None | Fast iteration |
| **Small** (1-1.5GB) | TinyLlama (1.1B), Llama3.2-1B 🆕 | ✅ Good | ⚠️ Minimal | Balanced experiments |
| **Medium** (2-4GB) | Llama3.2-3B 🆕, Phi-3 (3.8B) | ❌ Too slow | ✅ Required | Quality focus |
| **Large** (>4GB) | Mistral-7B | ❌ Impossible | ⚠️ Still slow | Not recommended |

### By Speed on M2 (CPU only)

| Model | Words/sec | Generation Time (100 words) | Viability |
|-------|-----------|----------------------------|-----------|
| **Qwen3** | 80-100 | 1-1.2s | ✅ Excellent |
| **Qwen2** | 80-100 | 1-1.2s | ✅ Excellent |
| **TinyLlama** | 60-80 | 1.2-1.7s | ✅ Good |
| **Llama3.2-1B** 🆕 | 40-60 (est) | 1.7-2.5s | ✅ Good |
| **SmolLM** | 10-15 | 6-10s | ✅ Acceptable |
| **Phi-3 (CPU)** | 0.1 | 1000s (~16 min) | ❌ Unusable |
| **Llama3.2-3B (CPU)** 🆕 | 2-5 (est) | 20-50s | ❌ Too slow |
| **Phi-3 (GPU)** | 18 | 5.5s | ✅ Acceptable |
| **Llama3.2-3B (GPU)** 🆕 | 15-25 (est) | 4-7s | ✅ Acceptable |

### By Quality (Estimated)

| Model | Parameters | Quality Tier | Reasoning | Best Use |
|-------|-----------|--------------|-----------|----------|
| **Phi-3** | 3.8B | 🥇 Highest (100%) | Strong | Quality benchmark |
| **Llama3.2-3B** 🆕 | 3.0B | 🥇 Premium (95%) | Meta proven | Phi-3 alternative |
| **SmolLM** | 1.7B | 🥈 High (75%) | Moderate | Balanced |
| **Llama3.2-1B** 🆕 | 1.0B | 🥉 Good (72%) | Meta quality | TinyLlama replacement |
| **TinyLlama** | 1.1B | 🥉 Good | Basic | Llama baseline |
| **Qwen3** | 0.6B | 🥉 Good | Basic | Fast baseline |
| **Qwen2** | 0.5B | 🥉 Good | Basic | Fast baseline |

---

## Deployment Decision Tree

```
Model Size?
    │
    ├─ < 1GB ──────────────────────────→ Use CPU (n_gpu_layers=0)
    │                                      ✅ Fast, efficient, low power
    │
    ├─ 1-2GB ──────────────────────────→ Use CPU (n_gpu_layers=0)
    │                                      ✅ Acceptable, test GPU if needed
    │
    └─ > 2GB ──────────────────────────→ MUST use GPU (n_gpu_layers=-1)
                                           ❌ CPU too slow, unusable
```

---

## Hardware Requirements Summary

### Minimum Requirements (CPU-only)

- **Processor:** Apple M1/M2 or equivalent
- **RAM:** 8 GB (16 GB recommended)
- **Storage:** 5 GB free space
- **Models:** Qwen2, Qwen3, TinyLlama

### GPU Requirements (for Phi-3)

- **Processor:** Apple M1/M2 with Metal support
- **RAM:** 16 GB (unified memory)
- **GPU Memory:** ~3 GB available (shared with system)
- **Models:** Phi-3 only

---

## Integration Checklist

When adding a new SLM to the codebase:

- [ ] Download GGUF model (Q4_0 or Q4_K_M quantization)
- [ ] Identify chat template format
- [ ] Create `{model}_llamacpp_wrapper.py` extending `LlamaCppBaseWrapper`
- [ ] Implement `_format_prompt()`, `_get_stop_tokens()`, `_extract_response()`
- [ ] Test on CPU with small prompt (10-30 tokens)
- [ ] If slow (>5s for 30 tokens), test GPU offloading
- [ ] Set appropriate `n_gpu_layers` (0 for CPU, -1 for GPU)
- [ ] Update `experiment_configs.py` with model entry
- [ ] Update `factorial_experiment.py` with wrapper import
- [ ] Update `run_experiment.py` with model choice
- [ ] Run full integration test (all intervention combinations)
- [ ] Document in this guide

---

## GGUF Quantization Reference

| Quantization | Bits | Size Impact | Quality | Recommendation |
|--------------|------|-------------|---------|----------------|
| **Q4_0** | 4-bit | Smallest | Good | ✅ Default for <1GB models |
| **Q4_K_M** | 4-bit mixed | Small | Better | ✅ Default for >1GB models |
| **Q5_K_M** | 5-bit mixed | Medium | High | ⚠️ If size permits |
| **Q8_0** | 8-bit | Large | Highest | ❌ Too large for M2 |
| **F16** | 16-bit | Huge | Reference | ❌ Not practical |

**Current Strategy:** Q4 quantization provides best size/quality tradeoff for M2 Mac.

---

## Recommendations by Use Case

### Fast Experimentation
**Best:** Qwen3 (0.6B)  
**Rationale:** Fastest model, good quality, minimal resource usage

### Balanced Quality/Speed (CPU-only)
**Best:** Llama3.2-1B 🆕  
**Rationale:** Meta quality, 128K context, TinyLlama-class speed  
**Alternative:** SmolLM (1.7B) - if slower speed acceptable

### Balanced Quality/Speed (GPU-enabled)
**Best:** Llama3.2-3B 🆕  
**Rationale:** Near Phi-3 quality, 128K context, faster with GPU  
**Alternative:** Phi-3 (3.8B) - highest quality

### Highest Quality
**Best:** Phi-3 (3.8B) with GPU  
**Rationale:** Best reasoning, but requires GPU acceleration

### Model Diversity
**Best:** All currently integrated (Qwen2, Qwen3, TinyLlama, Phi-3)  
**Rationale:** Covers range from 0.5B to 3.8B, different architectures

### Future Addition Priority
1. 🔥 **Llama3.2-1B** - HIGHEST PRIORITY: Meta quality, TinyLlama replacement, 128K context
2. **Llama3.2-3B** - Medium priority: Phi-3 alternative, 128K context
3. **Gemma-2B** - Google ecosystem, good size
4. **OpenELM-1.1B** - Apple Silicon optimized
5. **StableLM-2-1.6B** - Low priority (similar to SmolLM)

---

## Performance Expectations

### Typical Experiment (20 runs × ~100 words each)

| Model | Total Time | Per-Run Avg | Feasibility |
|-------|-----------|-------------|-------------|
| **Qwen3** | ~2 min | 6s | ✅ Excellent |
| **Qwen2** | ~2 min | 6s | ✅ Excellent |
| **TinyLlama** | ~3 min | 9s | ✅ Good |
| **Llama3.2-1B** 🆕 | ~5 min | 15s (est) | ✅ Good |
| **SmolLM** | ~12 min | 36s | ✅ Acceptable |
| **Llama3.2-3B (GPU)** 🆕 | ~10 min | 30s (est) | ✅ Good |
| **Phi-3 (GPU)** | ~15 min | 45s | ✅ Acceptable |
| **Phi-3 (CPU)** | ~5 hours | 900s | ❌ Impractical |
| **Llama3.2-3B (CPU)** 🆕 | ~30-60 min | 90-180s | ❌ Too slow |

---

## References

- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **Hugging Face Models:** https://huggingface.co/models
- **GGUF Format:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **GPU Optimization Guide:** `docs/GPU_OPTIMIZATION_GUIDE.md`

---

**Last Updated:** October 4, 2024  
**Hardware Tested:** MacBook M2, 16GB RAM  
**Framework:** llama.cpp with Metal acceleration
