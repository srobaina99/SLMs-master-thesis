import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Enable MPS fallback for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading Qwen2.5-0.5B model and tokenizer...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Use float16 instead of bfloat16 for Mac compatibility
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto"  # This will use MPS if available
)

# Define chat template
def generate_response(user_input, system_prompt="You are a helpful AI assistant."):
    # Qwen2.5 uses ChatML format
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    print("\nGenerating response...")
    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    
    response = outputs[0]["generated_text"]
    # Extract just the assistant's response
    assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    return assistant_response

# Interactive chat loop
if __name__ == "__main__":
    system_prompt = "You are a helpful AI assistant who provides concise and accurate information."
    print("\n=== Qwen2.5-0.5B Chat ===")
    print(f"System: {system_prompt}")
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        response = generate_response(user_input, system_prompt)
        print(f"\nQwen2.5: {response}") 