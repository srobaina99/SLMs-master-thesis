import os
from llama_cpp import Llama

# Set environment variables if needed
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Path to the downloaded GGUF model file
# You need to download this file from https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
MODEL_PATH = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Change this to your actual path

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
    print("Please download the model from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    print("For example: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    exit(1)

print(f"Loading model from {MODEL_PATH}...")

# Initialize the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Context window size
    n_threads=4,       # CPU threads to use
    n_gpu_layers=0,    # Set to higher number if you have a good GPU
    verbose=False      # Set to True for debugging
)

# Define chat template
def generate_response(user_input, system_prompt="You are a helpful AI assistant."):
    prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_input}
<|assistant|>"""
    
    print("\nGenerating response...")
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        stop=["<|user|>", "<|system|>"],
        echo=True  # Include the prompt in the output
    )
    
    # Extract just the assistant's response
    full_response = output["choices"][0]["text"]
    assistant_response = full_response.split("<|assistant|>")[-1].strip()
    return assistant_response

# Interactive chat loop
if __name__ == "__main__":
    system_prompt = "You are a helpful AI assistant who provides concise and accurate information."
    print("\n=== TinyLlama Chat (llama.cpp) ===")
    print(f"System: {system_prompt}")
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        response = generate_response(user_input, system_prompt)
        print(f"\nTinyLlama: {response}")