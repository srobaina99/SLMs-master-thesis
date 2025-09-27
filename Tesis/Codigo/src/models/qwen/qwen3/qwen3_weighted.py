import os
import torch
import sys
import argparse
import time
from IPython.display import Markdown, display
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import LogitsProcessorList

# Add parent directory to path to import ProbabilityWeightingLogitsProcessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.models.probability_processor import ProbabilityWeightingLogitsProcessor

# Enable MPS fallback for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading Qwen3-0.6B model and tokenizer...")
model_id = "unsloth/Qwen3-0.6B"  # Using unsloth's version which is publicly available

# Load model and tokenizer separately to apply custom logits processor
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"  # This will use MPS if available
)

# Define chat template
def generate_response(user_input, system_prompt="You are a helpful AI assistant.", 
                      enable_thinking=False, weighted_words=None, weight_factor=1.0, verbose=False):
    # Qwen3 uses ChatML format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # Apply chat template with thinking mode option
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # Qwen3 supports thinking mode
    )
    
    print("\nGenerating response...")
    
    # Setup logits processors
    logits_processors = LogitsProcessorList()
    
    # Add word weighting processor if words are specified
    if weighted_words and len(weighted_words) > 0:
        logits_processors.append(
            ProbabilityWeightingLogitsProcessor(
                tokenizer=tokenizer,
                words_to_weight=weighted_words,
                weight_factor=weight_factor,
                verbose=verbose
            )
        )
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate with custom logits processor
    start_time = time.time()
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        logits_processor=logits_processors,
    )
    elapsed_time = time.time() - start_time
    print(f"Response time: {elapsed_time:.2f} seconds")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
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
def start_chat(system_prompt, weighted_words=None, weight_factor=2.0, enable_thinking=False, verbose=False):
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(f"User input: {user_input}")

        # Check for mode switching commands
        if "/no_think" in user_input:
            enable_thinking = False
            user_input = user_input.replace("/no_think", "").strip()
            print("Thinking mode disabled")
        elif "/think" in user_input:
            enable_thinking = True
            user_input = user_input.replace("/think", "").strip()
            print("Thinking mode enabled")
        
        # Check for verbose mode
        if "/verbose" in user_input:
            verbose = True
            user_input = user_input.replace("/verbose", "").strip()
            print("Verbose mode enabled")
        elif "/no_verbose" in user_input:
            verbose = False
            user_input = user_input.replace("/no_verbose", "").strip()
            print("Verbose mode disabled")
            
        weighted_words = weighted_words + [".", "," ,"<|im_end|>"]
        response = generate_response(
            user_input, 
            system_prompt, 
            enable_thinking, 
            weighted_words, 
            weight_factor,
            verbose
        )
        try:
            # Try to use IPython display for notebook environments
            display(Markdown(f"\nQwen3:\n{response}"))
        except:
            # Fall back to regular print if not in a notebook
            print(f"\nQwen3: {response}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Qwen3-0.6B Chat with Word Weighting')
    
    # Add arguments
    parser.add_argument('--system', type=str, default="You are a helpful AI assistant who provides concise and accurate information.",
                        help='System prompt for the chat')
    parser.add_argument('--words', type=str, default="",
                        help='Comma-separated list of words to weight (e.g., "positive,happy,good")')
    parser.add_argument('--factor', type=float, default=2.0,
                        help='Weight factor for the specified words (>1 increases probability, <1 decreases)')
    parser.add_argument('--thinking', action='store_true',
                        help='Enable thinking mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose mode for token probability information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process weighted words
    weighted_words = [word.strip() for word in args.words.split(',')] if args.words else []
    if '' in weighted_words:
        weighted_words.remove('')
    
    # Start chat with parsed arguments
    start_chat(
        system_prompt=args.system,
        weighted_words=weighted_words,
        weight_factor=args.factor,
        enable_thinking=args.thinking,
        verbose=args.verbose
    ) 