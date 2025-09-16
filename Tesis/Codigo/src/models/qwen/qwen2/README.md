# Qwen2.5-0.5B on Mac

This folder contains scripts to run the Qwen2.5-0.5B model efficiently on macOS with Apple Silicon or Intel processors.

## About Qwen2.5-0.5B

Qwen2.5-0.5B is a small but capable language model from the Qwen2.5 family developed by Alibaba Cloud. Despite its small size (only 0.5B parameters), it offers decent performance for various tasks and is ideal for running on consumer hardware like Macs.

## Setup

### Prerequisites

- Python 3.8+
- macOS 12.3+ (for MPS acceleration)

### Installation

1. Install dependencies based on which approach you want to use:

   ```bash
   # For transformers version
   pip install torch transformers sentencepiece accelerate
   
   # For CTransformers version (recommended for better performance)
   pip install ctransformers
   ```

2. Download the model:
   - For the transformers version, the model will be downloaded automatically
   - For the CTransformers version, download a GGUF model from [Qwen's official repository](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
     ```bash
     # Example using wget
     wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
     ```

## Usage

### Option 1: Using Transformers with MPS acceleration

Run the script:
```bash
python qwen2_mps.py
```

This version:
- Uses PyTorch with MPS acceleration on Apple Silicon
- Falls back to CPU for unsupported operations
- Uses float16 instead of bfloat16 for compatibility
- Automatically downloads the model

### Option 2: Using CTransformers (recommended for better performance)

Run the script:
```bash
python qwen2_ctransformers.py
```

This version:
- Uses the optimized CTransformers backend (based on GGML/GGUF)
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

2. For CTransformers errors:
   - Check that the model file exists and path is correct
   - Try a different quantization level (e.g., Q2_K, Q4_K_M, Q5_K_M)
   - Adjust `threads` based on your CPU core count

## Performance Notes

- The CTransformers version typically offers better performance and lower memory usage
- For Apple Silicon Macs, the MPS version can be quite efficient but may have compatibility issues with some operations
- If you have an M1/M2/M3 Mac with limited RAM, the CTransformers version with a quantized model is recommended

## Available GGUF Models

The official Qwen repository offers several quantization options:

| Model | Size | Description |
|-------|------|-------------|
| qwen2.5-0.5b-instruct-q2_k.gguf | 415 MB | Smallest file size, lower quality |
| qwen2.5-0.5b-instruct-q4_k_m.gguf | 491 MB | Good balance of quality and size |
| qwen2.5-0.5b-instruct-q5_k_m.gguf | 522 MB | Better quality, slightly larger |
| qwen2.5-0.5b-instruct-q8_0.gguf | 676 MB | High quality, larger size |
| qwen2.5-0.5b-instruct-fp16.gguf | 1.27 GB | Full precision, largest size |

## Resources

- [Qwen2.5 on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen2.5 GGUF models](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) 