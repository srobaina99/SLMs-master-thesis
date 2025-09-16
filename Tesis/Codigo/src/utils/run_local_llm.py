import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import argparse
import os
import psutil

# Add explicit imports for Qwen2
try:
    from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
    QWEN2_AVAILABLE = True
except ImportError:
    QWEN2_AVAILABLE = False

# Check if MPS is available
MPS_AVAILABLE = torch.backends.mps.is_available()

# Force CPU usage instead of MPS only if explicitly requested
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def get_memory_info():
    """Get system memory information in GB"""
    mem = psutil.virtual_memory()
    total = mem.total / (1024 ** 3)
    available = mem.available / (1024 ** 3)
    used = mem.used / (1024 ** 3)
    return {
        "total": round(total, 2),
        "available": round(available, 2),
        "used": round(used, 2),
        "percent": mem.percent
    }

def list_recommended_models():
    """List recommended small models for Mac M2 with 8GB RAM"""
    models = {
        # Ultra Light Models (< 1B parameters)
        "qwen2-0.5b": {
            "model_id": "Qwen/Qwen2-0.5B-Instruct",
            "description": "0.5B parameter model, extremely lightweight, surprisingly capable, multilingual",
            "quantization": "4-bit",
            "size": "0.5B",
            "memory_usage": "~1GB"
        },
        "flan-t5-small": {
            "model_id": "google/flan-t5-small",
            "description": "80M parameter model, very lightweight, instruction-tuned",
            "quantization": "None",
            "size": "80M",
            "memory_usage": "~1GB"
        },
        
        # Light Models (1B-2B parameters)
        "tinyllama": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "description": "1.1B parameter model, very lightweight chat model",
            "quantization": "4-bit",
            "size": "1.1B",
            "memory_usage": "~2GB"
        },
        "phi-1.5": {
            "model_id": "microsoft/phi-1_5",
            "description": "1.3B parameter model, lightweight but capable",
            "quantization": "4-bit",
            "size": "1.3B",
            "memory_usage": "~2GB"
        },
        "qwen2-1.5b": {
            "model_id": "Qwen/Qwen2-1.5B-Instruct",
            "description": "1.5B parameter model, great balance of size and capability, multilingual",
            "quantization": "4-bit",
            "size": "1.5B",
            "memory_usage": "~2GB"
        },
        "stablelm2": {
            "model_id": "stabilityai/stablelm-2-1_6b-chat",
            "description": "1.6B parameter model, designed for chat and instruction following",
            "quantization": "4-bit",
            "size": "1.6B",
            "memory_usage": "~2GB"
        },
        
        # Medium Models (2B-3B parameters)
        "gemma-2b": {
            "model_id": "google/gemma-2b",
            "description": "2B parameter model, good for general text generation",
            "quantization": "4-bit",
            "size": "2B",
            "memory_usage": "~3GB"
        },
        "phi-2": {
            "model_id": "microsoft/phi-2",
            "description": "2.7B parameter model, good balance of size and capability",
            "quantization": "4-bit",
            "size": "2.7B",
            "memory_usage": "~3GB"
        },
        "openelm-3b": {
            "model_id": "apple/OpenELM-3B-Instruct",
            "description": "3B parameter model from Apple, designed for on-device efficiency",
            "quantization": "4-bit",
            "size": "3B",
            "memory_usage": "~3.5GB"
        },
        
        # Larger Models (>3B parameters) - Use with caution on 8GB RAM
        "phi-3-mini": {
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "description": "3.8B parameter model, highly capable, excellent for general tasks",
            "quantization": "4-bit",
            "size": "3.8B",
            "memory_usage": "~4GB"
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "7B parameter model, very capable but pushes RAM limits",
            "quantization": "4-bit",
            "size": "7B",
            "memory_usage": "~6GB",
            "warning": "May cause system slowdown due to high memory usage"
        }
    }
    
    print("\nRecommended models for Mac M2 with 8GB RAM:")
    print("-" * 100)
    
    # Helper function to convert size to float in billions
    def size_to_billions(size_str):
        if 'B' in size_str:
            return float(size_str.replace('B', ''))
        elif 'M' in size_str:
            return float(size_str.replace('M', '')) / 1000  # Convert millions to billions
        else:
            return float(size_str)
    
    categories = [
        ("Ultra Light Models (< 1B parameters)", lambda x: size_to_billions(x["size"]) < 1),
        ("Light Models (1B-2B parameters)", lambda x: 1 <= size_to_billions(x["size"]) < 2),
        ("Medium Models (2B-3B parameters)", lambda x: 2 <= size_to_billions(x["size"]) < 3),
        ("Larger Models (>3B parameters)", lambda x: size_to_billions(x["size"]) >= 3)
    ]
    
    for category_name, condition in categories:
        print(f"\n{category_name}:")
        print("-" * 50)
        for name, info in {k: v for k, v in models.items() if condition(v)}.items():
            print(f"- {name}")
            print(f"  Model ID: {info['model_id']}")
            print(f"  Size: {info['size']}")
            print(f"  Description: {info['description']}")
            print(f"  Recommended quantization: {info['quantization']}")
            print(f"  Estimated memory usage: {info['memory_usage']}")
            if "warning" in info:
                print(f"  WARNING: {info['warning']}")
            print()
    
    print("\nUsage example: python run_local_llm.py --model qwen2-0.5b")
    print("Note: First run will download the model from Hugging Face")
    print("-" * 100)

def load_model(model_name, use_4bit=True, use_gpu=True):
    """Load a model by name with appropriate quantization"""
    models = {
        "qwen2-0.5b": {
            "model_id": "Qwen/Qwen2-0.5B-Instruct",
            "description": "0.5B parameter model, extremely lightweight, surprisingly capable, multilingual",
            "quantization": "4-bit",
            "size": "0.5B",
            "memory_usage": "~1GB"
        },
        "flan-t5-small": {
            "model_id": "google/flan-t5-small",
            "description": "80M parameter model, very lightweight, instruction-tuned",
            "quantization": "None",
            "size": "80M",
            "memory_usage": "~1GB"
        },
        "tinyllama": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "description": "1.1B parameter model, very lightweight chat model",
            "quantization": "4-bit",
            "size": "1.1B",
            "memory_usage": "~2GB"
        },
        "phi-1.5": {
            "model_id": "microsoft/phi-1_5",
            "description": "1.3B parameter model, lightweight but capable",
            "quantization": "4-bit",
            "size": "1.3B",
            "memory_usage": "~2GB"
        },
        "qwen2-1.5b": {
            "model_id": "Qwen/Qwen2-1.5B-Instruct",
            "description": "1.5B parameter model, great balance of size and capability, multilingual",
            "quantization": "4-bit",
            "size": "1.5B",
            "memory_usage": "~2GB"
        },
        "stablelm2": {
            "model_id": "stabilityai/stablelm-2-1_6b-chat",
            "description": "1.6B parameter model, designed for chat and instruction following",
            "quantization": "4-bit",
            "size": "1.6B",
            "memory_usage": "~2GB"
        },
        "gemma-2b": {
            "model_id": "google/gemma-2b",
            "description": "2B parameter model, good for general text generation",
            "quantization": "4-bit",
            "size": "2B",
            "memory_usage": "~3GB"
        },
        "phi-2": {
            "model_id": "microsoft/phi-2",
            "description": "2.7B parameter model, good balance of size and capability",
            "quantization": "4-bit",
            "size": "2.7B",
            "memory_usage": "~3GB"
        },
        "openelm-3b": {
            "model_id": "apple/OpenELM-3B-Instruct",
            "description": "3B parameter model from Apple, designed for on-device efficiency",
            "quantization": "4-bit",
            "size": "3B",
            "memory_usage": "~3.5GB"
        },
        "phi-3-mini": {
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "description": "3.8B parameter model, highly capable, excellent for general tasks",
            "quantization": "4-bit",
            "size": "3.8B",
            "memory_usage": "~4GB"
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "7B parameter model, very capable but pushes RAM limits",
            "quantization": "4-bit",
            "size": "7B",
            "memory_usage": "~6GB",
            "warning": "May cause system slowdown due to high memory usage"
        }
    }
    
    if model_name not in models:
        print(f"Model {model_name} not found in recommended list.")
        list_recommended_models()
        return None, None
    
    model_id = models[model_name]["model_id"]
    print(f"Loading model: {model_id}")
    
    # Determine device
    if use_gpu and MPS_AVAILABLE:
        device = "mps"
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    else:
        device = "cpu"
        if use_gpu and not MPS_AVAILABLE:
            print("MPS not available. Falling back to CPU.")
        else:
            print("Using CPU as requested.")
    
    # Get memory before loading
    mem_before = get_memory_info()
    print(f"Memory before loading: {mem_before['available']:.2f}GB available")
    
    # Special handling for Qwen2 models
    if "qwen2" in model_name.lower():
        if not QWEN2_AVAILABLE:
            print("Qwen2 models require special tokenizer classes. Installing necessary packages...")
            print("Please run: pip install --upgrade transformers")
            return None, None
            
        tokenizer = Qwen2Tokenizer.from_pretrained(model_id)
        if use_4bit:
            model = Qwen2ForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if device == "mps" else torch.float32
                )
            )
        else:
            model = Qwen2ForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float16 if device == "mps" else torch.float32
            )
    # Special case for T5 models
    elif "t5" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16 if device == "mps" else torch.float32
        )
    else:
        # For other models, use 4-bit quantization if requested
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if device == "mps" else torch.float32
        ) if use_4bit else None
            
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            quantization_config=quantization_config
        )
    
    # Get memory after loading
    mem_after = get_memory_info()
    print(f"Memory after loading: {mem_after['available']:.2f}GB available")
    print(f"Memory used by model: {mem_before['available'] - mem_after['available']:.2f}GB")
    
    return model, tokenizer

def chat_with_model(model, tokenizer, max_new_tokens=512):
    """Interactive chat with the model"""
    print("\nStarting chat session (type 'exit' to quit)")
    print("-" * 50)
    
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        # Generate response
        response = generation_pipeline(
            user_input,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0]["generated_text"]
        
        # Print the model's response (remove the input prompt)
        model_response = response[len(user_input):].strip()
        print(f"\nModel: {model_response}")

def main():
    parser = argparse.ArgumentParser(description="Run small LLMs locally on Mac M2")
    parser.add_argument("--model", type=str, help="Model name to load")
    parser.add_argument("--list-models", action="store_true", help="List recommended models")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--system-info", action="store_true", help="Show system memory info")
    
    args = parser.parse_args()
    
    if args.system_info:
        mem_info = get_memory_info()
        print("\nSystem Memory Information:")
        print(f"Total: {mem_info['total']}GB")
        print(f"Available: {mem_info['available']}GB")
        print(f"Used: {mem_info['used']}GB")
        print(f"Usage: {mem_info['percent']}%")
        print()
    
    if args.list_models:
        list_recommended_models()
        return
        
    if not args.model:
        print("Please specify a model with --model or use --list-models to see options")
        list_recommended_models()
        return
        
    model, tokenizer = load_model(args.model, not args.no_4bit, not args.cpu_only)
    if model and tokenizer:
        chat_with_model(model, tokenizer)

if __name__ == "__main__":
    main() 