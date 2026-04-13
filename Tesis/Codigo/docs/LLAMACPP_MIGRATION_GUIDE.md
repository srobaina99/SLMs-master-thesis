# llama.cpp Migration Guide

## Overview

This guide documents the pattern for migrating models to llama.cpp GGUF backend for **4.4x faster inference** and **57% less memory** usage.

**Status:** All 4 models migrated ✅ (Qwen3, Qwen2, Phi3, TinyLlama)

---

## Why llama.cpp?

### Benchmark Results (Qwen3-0.6B)

| Metric | Transformers + MPS | llama.cpp + GGUF | Improvement |
|--------|-------------------|------------------|-------------|
| **Inference Speed** | 22.5 tok/s | **98.1 tok/s** | **4.4x faster** |
| **Memory Usage** | 1,137 MB | **491 MB** | **57% reduction** |
| **Load Time** | 7.1s | 16.7s | 2.3x slower (one-time cost) |
| **Weighting Support** | ✅ Native | ✅ logit_bias | Both work |

### Key Benefits
- **Faster experiments:** 1000-prompt run takes ~5 hours vs ~22 hours
- **Lower memory:** Can run multiple models simultaneously on M2
- **Official GGUF models:** Available for all target SLMs
- **Metal acceleration:** Native M-series optimization

---

## Architecture

### Class Hierarchy
```
BaseModelWrapper (abstract)
    └── LlamaCppBaseWrapper (abstract, reusable)
        ├── Qwen3LlamaCppWrapper ✅
        ├── Qwen2LlamaCppWrapper ✅
        ├── Phi3LlamaCppWrapper ✅ (GPU, n_gpu_layers=-1)
        └── TinyLlamaLlamaCppWrapper ✅
```

### Key Components

1. **`LlamaCppBaseWrapper`** (`src/framework/models/llamacpp_base.py`)
   - Reusable base class for all GGUF models
   - Handles model loading, logit_bias, response extraction
   - Subclasses only implement model-specific templates

2. **Model-Specific Wrapper** (e.g., `qwen3_llamacpp_wrapper.py`)
   - Extends `LlamaCppBaseWrapper`
   - Implements 3 abstract methods:
     - `_format_prompt()`: Chat template
     - `_get_stop_tokens()`: End-of-generation markers
     - `_extract_response()`: Clean output text

3. **Experiment Integration** (`factorial_experiment.py`)
   - Update `_model_classes` dictionary
   - No other changes needed

---

## Migration Steps

### Step 1: Find Official GGUF Model

```bash
# Search Hugging Face Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search='MODEL_NAME GGUF', limit=10)
print('\n'.join([m.id for m in models]))
"
```

**Trustworthy sources:**
- `ggml-org/*` (official GGML organization)
- `unsloth/*` (if you're already using their transformers version)
- `lmstudio-community/*` (verified community)

### Step 2: Download GGUF Model

```bash
cd Tesis/Codigo
mkdir -p models/gguf

python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='ggml-org/MODEL_NAME-GGUF',
    filename='MODEL_NAME-Q4_0.gguf',
    local_dir='models/gguf'
)
print(f'Downloaded to: {path}')
"
```

**Quantization recommendations:**
- **Q4_0**: Best balance (recommended for most models)
- **Q8_0**: Higher quality, 2x larger
- **f16**: Full precision, 4x larger (rarely needed)

### Step 3: Create Model Wrapper

**Template:** `src/framework/models/{model_name}_llamacpp_wrapper.py`

```python
"""
{ModelName} model wrapper using llama.cpp GGUF backend.
"""

import os
import sys
from typing import Dict, Any, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.append(project_root)

from .llamacpp_base import LlamaCppBaseWrapper


class {ModelName}LlamaCppWrapper(LlamaCppBaseWrapper):
    """
    {ModelName} model wrapper using llama.cpp GGUF backend.
    
    Model: ggml-org/{ModelName}-GGUF (Q4_0 quantization)
    Template: [ChatML / Llama2 / etc.]
    """
    
    def __init__(self, model_path: str = None):
        """Initialize {ModelName} llama.cpp wrapper."""
        # Default model path (project_root = .../Tesis)
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "Codigo",
                "models",
                "gguf",
                "{ModelName}-Q4_0.gguf"
            )
        
        super().__init__(
            model_name="{ModelName}",
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # Metal auto-detection
            timeout_seconds=300
        )
    
    def _format_prompt(self, user_input: str, system_prompt: str, enable_thinking: bool = False) -> str:
        """
        Format prompt using model-specific template.
        
        Examples:
        - ChatML (Qwen): <|im_start|>system\n{system}<|im_end|>\n...
        - Llama2: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
        - Alpaca: ### Instruction:\n{system}\n\n### Input:\n{user}\n\n### Response:\n
        """
        # TODO: Implement model-specific template
        pass
    
    def _get_stop_tokens(self) -> List[str]:
        """
        Get stop tokens for this model.
        
        Examples:
        - ChatML: ["<|im_end|>", "<|endoftext|>"]
        - Llama2: ["</s>"]
        - Alpaca: ["\n\n###"]
        """
        # TODO: Return model-specific stop tokens
        pass
    
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean response from model output.
        
        Common cleanup:
        - Remove thinking tags
        - Strip template markers
        - Clean whitespace
        """
        response = raw_output.strip()
        
        # TODO: Add model-specific cleanup
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model-specific information."""
        info = super().get_model_info()
        info.update({
            'model_id': 'ggml-org/{ModelName}-GGUF',
            'quantization': 'Q4_0',
            'template_format': 'ChatML',  # Update as needed
            'parameters': '0.6B'  # Update as needed
        })
        return info
```

### Step 4: Update Module Exports

**File:** `src/framework/models/__init__.py`

```python
from .{model_name}_llamacpp_wrapper import {ModelName}LlamaCppWrapper

__all__ = [
    # ... existing exports ...
    '{ModelName}LlamaCppWrapper',
]
```

### Step 5: Update Experiment Configuration

**File:** `src/framework/experiments/factorial_experiment.py`

```python
from src.framework.models import (
    # ... existing imports ...
    {ModelName}LlamaCppWrapper
)

class FactorialExperiment:
    def __init__(self, ...):
        self._model_classes = {
            # ... existing models ...
            "{ModelName}": {ModelName}LlamaCppWrapper,
        }
```

**File:** `src/framework/experiments/experiment_configs.py`

```python
MODEL_CONFIGS = {
    # ... existing configs ...
    "{ModelName}": {
        "model_name": "{ModelName}",
        "model_id": "ggml-org/{ModelName}-GGUF"  # Using llama.cpp
    },
}
```

### Step 6: Test Integration

Create test script: `scripts/test_{model_name}_llamacpp_integration.py`

```python
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework.models import {ModelName}LlamaCppWrapper
from src.framework.core.data_models import ExperimentConfig

def test_{model_name}_llamacpp():
    print("\n" + "="*60)
    print("{MODELNAME} LLAMACPP INTEGRATION TEST")
    print("="*60)
    
    wrapper = {ModelName}LlamaCppWrapper()
    
    if not wrapper.model_loaded:
        print("❌ FAILED: Model did not load")
        return False
    
    # Test all intervention combinations
    test_prompt = "What is a cat?"
    system_prompt = "You are a helpful English teacher."
    
    for weighting in [False, True]:
        for prompting in [False, True]:
            config = ExperimentConfig(
                model_name="{ModelName}",
                system_prompt=system_prompt,
                config_weighting=weighting,
                config_prompting=prompting,
                weight_factor=1.5,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                max_new_tokens=200
            )
            
            result = wrapper.generate_response(test_prompt, config)
            
            if not result['generation_successful']:
                print(f"❌ FAILED: {result['error_message']}")
                return False
    
    print("✅ ALL TESTS PASSED")
    return True

if __name__ == "__main__":
    success = test_{model_name}_llamacpp()
    sys.exit(0 if success else 1)
```

Run test:
```bash
cd Tesis/Codigo
source venv/bin/activate
python scripts/test_{model_name}_llamacpp_integration.py
```

---

## Common Chat Templates

### ChatML (Qwen, Phi-3)
```python
def _format_prompt(self, user_input: str, system_prompt: str) -> str:
    return (
        f"<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def _get_stop_tokens(self) -> List[str]:
    return ["<|im_end|>", "<|endoftext|>"]
```

### Llama2 / Mistral
```python
def _format_prompt(self, user_input: str, system_prompt: str) -> str:
    return (
        f"<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        f"<</SYS>>\n\n"
        f"{user_input} [/INST]"
    )

def _get_stop_tokens(self) -> List[str]:
    return ["</s>"]
```

### Gemma
```python
def _format_prompt(self, user_input: str, system_prompt: str) -> str:
    return (
        f"<start_of_turn>user\n"
        f"{system_prompt}\n\n"
        f"{user_input}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def _get_stop_tokens(self) -> List[str]:
    return ["<end_of_turn>", "<eos>"]
```

---

## Troubleshooting

### Issue: Model file not found
**Symptom:** `❌ Model file not found at .../models/gguf/...`

**Solution:**
1. Verify download location:
   ```bash
   find . -name "*.gguf"
   ```
2. Check `project_root` calculation in wrapper
3. Use absolute path for testing:
   ```python
   wrapper = ModelLlamaCppWrapper(model_path="/absolute/path/to/model.gguf")
   ```

### Issue: Metal warnings (bf16 kernels skipped)
**Symptom:** Many `ggml_metal_init: skipping kernel_*_bf16` warnings

**Impact:** None (model uses fallback kernels)  
**Action:** Ignore warnings (M2 Metal doesn't support bfloat16)

### Issue: Slow load time
**Symptom:** Model takes 15-20 seconds to load

**Impact:** Minimal (one-time cost per session)  
**Mitigation:** Keep model loaded during experiments (already done in `FactorialExperiment`)

### Issue: Weighting doesn't affect output
**Symptom:** Weighted and non-weighted responses are identical

**Debug:**
1. Check logit_bias is being created:
   ```python
   logit_bias = wrapper._create_logit_bias(vocab, 2.0)
   print(f"Logit bias tokens: {len(logit_bias)}")  # Should be > 0
   ```
2. Verify vocab file exists and is loaded
3. Try higher `weight_factor` (e.g., 5.0) for testing

---

## Migration Checklist

- [ ] Find official GGUF model on Hugging Face
- [ ] Download Q4_0 quantized version to `models/gguf/`
- [ ] Create `{model_name}_llamacpp_wrapper.py`
- [ ] Implement `_format_prompt()` with correct template
- [ ] Implement `_get_stop_tokens()`
- [ ] Implement `_extract_response()`
- [ ] Update `models/__init__.py` exports
- [ ] Update `factorial_experiment.py` model classes
- [ ] Update `experiment_configs.py` model ID
- [ ] Create integration test script
- [ ] Run integration test (all 4 intervention combinations)
- [ ] Run small factorial experiment to validate end-to-end
- [ ] Document any model-specific quirks

---

## Migration Status

All target models have been migrated:
- [x] **Qwen3-0.6B** — First migration, ChatML template
- [x] **Qwen2-0.5B** — ChatML template
- [x] **Phi-3-mini-3.8B** — Custom Phi3 template, requires GPU (`n_gpu_layers=-1`)
- [x] **TinyLlama-1.1B** — TinyLlama template

### Resources
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python docs](https://llama-cpp-python.readthedocs.io/)
- [GGUF format spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Hugging Face GGUF models](https://huggingface.co/models?library=gguf)

---

**Last Updated:** 2026-04-08  
**Status:** All 4 models migrated ✅ (Qwen3, Qwen2, Phi3, TinyLlama)
