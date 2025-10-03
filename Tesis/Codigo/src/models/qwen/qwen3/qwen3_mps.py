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
print("Loading Qwen3-4B model and tokenizer...")
model_id = "Qwen/Qwen3-4B"  # Using the standard 4B model which is the next size up

# Use float16 instead of bfloat16 for Mac compatibility
pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype=torch.float16,
    device_map="auto"  # This will use MPS if available
)

# Define chat template
def generate_response(user_input, system_prompt="You are a helpful AI assistant.", enable_thinking=True):
    # Qwen3 uses ChatML format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # Apply chat template with thinking mode option
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # Qwen3 supports thinking mode
    )
    
    print("\nGenerating response...")
    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        repetition_penalty=1.05,
    )
    
    response = outputs[0]["generated_text"]
    
    # Extract just the assistant's response
    # For thinking mode, we need to handle both thinking and response content
    if enable_thinking:
        try:
            # Extract thinking content (if present)
            thinking_marker = "<think>"
            thinking_end_marker = "</think>"
            
            if thinking_marker in response:
                thinking_start = response.find(thinking_marker) + len(thinking_marker)
                thinking_end = response.find(thinking_end_marker)
                thinking_content = response[thinking_start:thinking_end].strip()
                
                # Extract the actual response after thinking
                assistant_response = response[response.find(thinking_end_marker) + len(thinking_end_marker):].strip()
                
                print("\nThinking process:")
                print(thinking_content)
                return assistant_response
            else:
                # No thinking markers found, return the whole response
                return response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        except Exception as e:
            print(f"Error parsing thinking content: {e}")
            return response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    else:
        # Standard response extraction
        return response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()

# Interactive chat loop
if __name__ == "__main__":
    system_prompt = "You are a helpful AI assistant who provides concise and accurate information."
    print("\n=== Qwen3-4B Chat ===")
    print(f"System: {system_prompt}")
    print("Type 'exit' to quit")
    print("Type '/think' to enable thinking mode, '/no_think' to disable it")
    
    enable_thinking = True  # Default to thinking mode
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Check for mode switching commands
        if "/no_think" in user_input:
            enable_thinking = False
            user_input = user_input.replace("/no_think", "").strip()
            print("Thinking mode disabled")
        elif "/think" in user_input:
            enable_thinking = True
            user_input = user_input.replace("/think", "").strip()
            print("Thinking mode enabled")
            
        response = generate_response(user_input, system_prompt, enable_thinking)
        print(f"\nQwen3: {response}") 