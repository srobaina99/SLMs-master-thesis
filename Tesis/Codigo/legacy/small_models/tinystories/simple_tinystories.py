import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable MPS fallback for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Set your prompt
prompt = "Once upon a time there was a little boy who"

# Generate text
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=200)
story = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated story:")
print(story) 