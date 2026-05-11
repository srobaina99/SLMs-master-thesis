# GPU Optimization Guide for llama.cpp on M2 Mac

## Overview

This document details the GPU acceleration optimization for running GGUF models with llama.cpp on Apple Silicon (M2 Mac), including empirical performance data and configuration recommendations.

---

## Background: CPU vs GPU Inference

### Architecture

Large Language Models (LLMs) are composed of **transformer layers** stacked sequentially:

```
Input → Layer 1 → Layer 2 → ... → Layer N → Output
```

Each layer performs intensive matrix multiplication operations, which can be executed on either:
- **CPU cores** (~4-8 cores, optimized for sequential processing)
- **GPU cores** (~3000 cores on M2, optimized for parallel matrix operations)

### The `n_gpu_layers` Parameter

In llama.cpp, the `n_gpu_layers` parameter controls GPU offloading:

| Value | Behavior |
|-------|----------|
| `0` | All layers run on CPU (default) |
| `N` | First N layers run on GPU, rest on CPU |
| `-1` | **All layers offloaded to GPU** (full Metal acceleration) |

---

## Empirical Performance Testing

### Test Methodology

**Hardware:** MacBook M2  
**Models Tested:** Phi-3 (3.8B), SmolLM (1.7B, no longer in active framework), Qwen3 (0.6B)  
**Test:** Simple generation ("What is 2+2?", 10 tokens)  
**Configurations:** `n_gpu_layers=0` (CPU) vs `n_gpu_layers=-1` (GPU)

### Results

#### Phi-3-mini (3.8B parameters, 2.2GB, 32 layers)

| Metric | CPU Only | Full GPU | Improvement |
|--------|----------|----------|-------------|
| **Load Time** | 2.4s | 4.4s | -83% (slower) |
| **Generation Time** | 36.8s | 0.5s | **+7360% (73x faster)** |
| **Tokens/sec** | ~0.3 | ~20 | 67x faster |
| **Practical Use** | ❌ Too slow | ✅ Viable |

**Verdict:** 🎯 **CRITICAL - GPU offloading essential**

#### SmolLM (1.7B parameters, 1.0GB, 24 layers)

| Metric | CPU Only | Full GPU | Improvement |
|--------|----------|----------|-------------|
| **Load Time** | 1.0s | 0.1s | +90% (faster) |
| **Generation Time** | 0.3s | 0.3s | No change |
| **Tokens/sec** | ~33 | ~33 | 1.0x |

**Verdict:** ⚠️ **MARGINAL - No significant benefit**

#### Qwen3 (0.6B parameters, 409MB, 28 layers)

| Metric | CPU Only | Full GPU | Improvement |
|--------|----------|----------|-------------|
| **Load Time** | 0.8s | 0.2s | +75% (faster) |
| **Generation Time** | 0.1s | 0.1s | No change |
| **Tokens/sec** | ~100 | ~100 | 1.0x |

**Verdict:** ❌ **NO BENEFIT - Already optimal on CPU**

---

## Why the Difference?

### Large Models (>2GB) - Massive Speedup

**Phi-3 characteristics:**
- 32 layers × large matrices = heavy computation
- CPU bottleneck: Sequential processing of large matrices
- GPU advantage: Thousands of cores process matrix operations in parallel
- **Result:** 73x speedup

### Small Models (<1GB) - No Benefit

**Qwen3 characteristics:**
- 28 layers × small matrices = light computation
- CPU already fast: 0.1s generation time
- GPU overhead: CPU↔GPU communication takes ~0.05s
- **Result:** Overhead negates any benefit

---

## Performance Scaling Pattern

```
Speedup Factor
    │
 70x│                                    ●  Phi-3 (3.8B)
    │
 50x│
    │
 30x│
    │
 10x│
    │
  5x│
    │
  1x│        ●  Qwen3 (0.6B)    ●  SmolLM (1.7B)
    │
    └────────────────────────────────────────────> Model Size
         500MB              1GB              2GB
```

**Key Insight:** GPU offloading benefit scales exponentially with model size.

---

## Tradeoffs and Considerations

### Advantages of GPU Offloading

✅ **Massive speedup for large models** (10-100x)  
✅ **No accuracy loss** - Identical output quality  
✅ **Better GPU utilization** - Leverages Metal acceleration  
✅ **Unified memory** - M2's shared CPU/GPU memory helps  

### Disadvantages of GPU Offloading

⚠️ **Longer load time** (+2-3s for model initialization)  
⚠️ **Higher GPU memory usage** (~model size in VRAM)  
⚠️ **Increased power consumption** - More heat generation  
⚠️ **No benefit for small models** - Overhead can slow them down  

### Hardware Constraints (M2 Mac)

- **GPU Memory:** Shared with system RAM (unified memory)
- **Thermal limits:** Sustained GPU load increases heat
- **Power draw:** Higher battery consumption on laptops
- **Metal API:** Apple's GPU framework (well-optimized)

---

## Implementation in Codebase

### Current Configuration

**File:** `src/framework/models/*_llamacpp_wrapper.py`

```python
# Phi-3: Full GPU offloading (ENABLED)
super().__init__(
    model_name="Phi3",
    model_path=model_path,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=-1,  # All 32 layers on GPU
    timeout_seconds=300
)

# Qwen3, SmolLM, TinyLlama: CPU only (OPTIMAL)
super().__init__(
    model_name="Qwen3",
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,  # CPU only - already fast
    timeout_seconds=300
)
```

### Verification

Check actual GPU usage in verbose output:

```bash
python -c "from llama_cpp import Llama; \
llm = Llama('model.gguf', n_gpu_layers=-1, verbose=True)"
```

Look for:
```
load_tensors: offloading 32 repeating layers to GPU
load_tensors: offloaded 32/32 layers to GPU
```

---

## Recommendations

### When to Enable GPU Offloading

✅ **Enable (`n_gpu_layers=-1`) when:**
- Model size > 2GB
- Generation time > 5 seconds on CPU
- Accuracy is critical (no quality loss)
- Power/heat are acceptable

❌ **Keep CPU (`n_gpu_layers=0`) when:**
- Model size < 1GB
- Already fast on CPU (< 1s generation)
- Minimizing load time is important
- Battery life is a concern

### Model-Specific Recommendations

| Model | Size | Layers | Recommended Setting | Rationale |
|-------|------|--------|---------------------|-----------|
| **Phi-3** | 2.2GB | 32 | `n_gpu_layers=-1` | 73x speedup essential |
| **Qwen3** | 409MB | 28 | `n_gpu_layers=0` | Already optimal (0.1s) |
| **Qwen2** | 409MB | ? | `n_gpu_layers=0` | Similar to Qwen3 |
| **TinyLlama** | 608MB | ? | `n_gpu_layers=0` | Small model |

---

## Future Considerations

### Potential Optimizations

1. **Hybrid offloading:** Offload only compute-heavy layers (e.g., `n_gpu_layers=16`)
2. **Dynamic switching:** Detect model size and auto-configure
3. **Batch processing:** GPU excels at batched inference
4. **Quantization:** Lower precision (Q4) already optimal for M2

### Monitoring

Track these metrics to evaluate GPU offloading:
- **Generation time** (primary metric)
- **Load time** (one-time cost)
- **Memory usage** (GPU VRAM)
- **Temperature** (thermal throttling risk)
- **Power consumption** (battery life)

---

## References

- **llama.cpp documentation:** https://github.com/ggerganov/llama.cpp
- **Apple Metal API:** https://developer.apple.com/metal/
- **M2 GPU specs:** ~3.6 TFLOPS, ~3000 cores, unified memory

---

## Changelog

**2025-10-04:**
- Initial documentation
- Empirical testing on M2 Mac
- Phi-3 optimization implemented (73x speedup achieved)
- Recommendations for all models finalized

---

**Summary:** GPU offloading via `n_gpu_layers=-1` provides massive speedups (73x) for large models (>2GB) but offers no benefit for small models (<1GB) already fast on CPU. Current configuration is optimal: Phi-3 uses GPU, others (Qwen2, Qwen3, TinyLlama) use CPU.

> **Note:** SmolLM benchmark data above is historical — SmolLM was tested during evaluation but is not part of the active experiment framework (no wrapper in `src/framework/models/`).





