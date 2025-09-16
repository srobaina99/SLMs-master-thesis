import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable MPS fallback for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Set device
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Set your prompt
prompt = "Write a short story about a robot who becomes friends with a human child"
print(f"\nPrompt: {prompt}")

# Process on CPU - this is the simplest approach that works reliably
input_ids = tokenizer.encode(prompt, return_tensors="pt")
attention_mask = torch.ones_like(input_ids)

# Generate text
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=200,
    num_beams=1
)
story = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated story:")
print(story) 