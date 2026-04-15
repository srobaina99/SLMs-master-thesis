import torch
from llama_cpp import Llama

print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Check if llama-cpp-python was compiled with CUDA
try:
    # This will log if it's using BLAS/CUDA
    llm = Llama(model_path="", verbose=True) 
except Exception:
    # Expecting an error since no model path, but look for 'BLAS = 1' in output
    pass