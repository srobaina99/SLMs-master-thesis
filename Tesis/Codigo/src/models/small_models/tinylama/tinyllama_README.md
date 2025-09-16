# TinyLlama on Mac

This guide explains how to run TinyLlama models on macOS with Apple Silicon or Intel processors.

## Setup

### Prerequisites

- Python 3.8+
- macOS 12.3+ (for MPS acceleration)

### Installation

1. Install additional dependencies:

   ```bash
   # For transformers version
   pip install torch transformers sentencepiece accelerate

   # For llama.cpp version (recommended for better performance)
   pip install llama-cpp-python
   ```
2. Download the model:

   - For the transformers version, the model will be downloaded automatically
   - For the llama.cpp version, download a GGUF model from [TheBloke&#39;s TinyLlama models](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
     ```bash
     # Example using wget
     wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
     ```

## Usage

### Option 1: Using Transformers (with MPS acceleration)

Run the script:

```bash
python tinyllama_mac.py
```

This version:

- Uses PyTorch with MPS acceleration on Apple Silicon
- Falls back to CPU for unsupported operations
- Uses float16 instead of bfloat16 for compatibility

### Option 2: Using llama.cpp (recommended for better performance)

Run the script:

```bash
python tinyllama_llamacpp.py
```

This version:

- Uses the highly optimized llama.cpp backend
- Works well on older hardware and Intel Macs
- Requires downloading a GGUF model file first
- Update the `MODEL_PATH` variable in the script if needed

## Customization

You can customize the following parameters in both scripts:

- `system_prompt`: Change the system instructions
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = more random)
- `max_new_tokens` or `max_tokens`: Controls response length
- `top_k` and `top_p`: Controls diversity of responses

## Troubleshooting

If you encounter errors:

1. For MPS-related errors:

   - Make sure you're using macOS 12.3 or later
   - Try setting `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in the scripts)
   - Fall back to CPU by modifying the device settings
2. For llama.cpp errors:

   - Check that the model file exists and path is correct
   - Try a different quantization level (e.g., Q2_K, Q4_K_M, Q5_K_M)
   - Adjust `n_threads` based on your CPU core count

## Resources

- [TinyLlama GitHub Repository](https://github.com/jzhang38/TinyLlama)
- [TinyLlama on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [TheBloke&#39;s quantized models](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
