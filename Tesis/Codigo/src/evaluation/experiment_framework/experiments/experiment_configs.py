"""
Standard experiment configurations and prompts for factorial experiments.
"""

from typing import List, Dict, Any
import sys
import os

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.data_models import ExperimentConfig


# Standard prompts for English learning experiments
STANDARD_PROMPTS = [
    "What does the word 'library' mean?",
    "How do I introduce myself in English?",
    "What is a dog?",
    "Can you explain what 'breakfast' is?",
    "What is the difference between 'big' and 'large'?",
    "Can you tell me about your favorite animal?",
    "What colors do you see in a rainbow?",
    "Can you describe what happens in the morning?"
]

# Model configurations for factorial experiment
MODEL_CONFIGS = {
    "Phi3": {
        "model_name": "Phi3",
        "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf"  # Using llama.cpp GGUF (3.8B reasoning model)
    },
    "Qwen2": {
        "model_name": "Qwen2",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF"  # Using llama.cpp GGUF (4x faster)
    },
    "Qwen3": {
        "model_name": "Qwen3", 
        "model_id": "ggml-org/Qwen3-0.6B-GGUF"  # Using llama.cpp GGUF (4.4x faster)
    },
    "SmolLM": {
        "model_name": "SmolLM",
        "model_id": "MaziyarPanahi/SmolLM-1.7B-Instruct-GGUF"  # Using llama.cpp GGUF (efficient architecture)
    },
    "TinyLlama": {
        "model_name": "TinyLlama",
        "model_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"  # Using llama.cpp GGUF
    },
    "TinyStories": {
        "model_name": "TinyStories",
        "model_id": "roneneldan/TinyStories-33M"
    }
}


def create_factorial_configs() -> List[ExperimentConfig]:
    """
    Create all factorial experiment configurations.
    
    Returns 4 models × 4 intervention combinations = 16 configurations
    
    Returns:
        List of ExperimentConfig objects for factorial experiment
    """
    configs = []
    
    # Base system prompt for English learning
    system_prompt = "You are a helpful English teacher for beginner students. Answer with a paragraph only with plain text"
    
    # Intervention combinations: (weighting, prompting)
    interventions = [
        (False, False),  # Control
        (True, False),   # Weighting only
        (False, True),   # Prompting only  
        (True, True)     # Both interventions
    ]
    
    for model_name, model_info in MODEL_CONFIGS.items():
        for config_weighting, config_prompting in interventions:
            
            # Create configuration name
            config_name = f"{model_name}"
            if config_weighting and config_prompting:
                config_name += "_weighted_prompted"
            elif config_weighting:
                config_name += "_weighted"
            elif config_prompting:
                config_name += "_prompted"
            else:
                config_name += "_control"
            
            config = ExperimentConfig(
                model_name=model_info["model_name"],
                model_id=model_info["model_id"],
                system_prompt=system_prompt,
                config_weighting=config_weighting,
                config_prompting=config_prompting,
                
                # Legacy fields for backward compatibility
                weighted_words_enabled=config_weighting,
                weight_factor=2.0,  # Standard weighting factor
                enable_thinking=False,
                verbose=False,
                
                # Generation parameters
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                max_new_tokens=1024,
                
                # Experiment metadata
                experiment_name=config_name,
                description=f"Factorial experiment: {model_name} with weighting={config_weighting}, prompting={config_prompting}"
            )
            
            configs.append(config)
    
    return configs


def get_config_by_name(config_name: str) -> ExperimentConfig:
    """
    Get a specific configuration by name.
    
    Args:
        config_name: Name of the configuration (e.g., "Qwen2_weighted_prompted")
        
    Returns:
        ExperimentConfig object
        
    Raises:
        ValueError: If config_name not found
    """
    configs = create_factorial_configs()
    
    for config in configs:
        if config.experiment_name == config_name:
            return config
    
    available_names = [c.experiment_name for c in configs]
    raise ValueError(f"Configuration '{config_name}' not found. Available: {available_names}")


def get_configs_for_model(model_name: str) -> List[ExperimentConfig]:
    """
    Get all configurations for a specific model.
    
    Args:
        model_name: Name of the model ("Qwen2", "Qwen3", "TinyLlama", "TinyStories")
        
    Returns:
        List of ExperimentConfig objects for the specified model
    """
    configs = create_factorial_configs()
    return [c for c in configs if c.model_name == model_name]


def print_all_configs():
    """Print summary of all available configurations."""
    configs = create_factorial_configs()
    
    print(f"Total configurations: {len(configs)}")
    print("\nConfigurations by model:")
    
    for model_name in MODEL_CONFIGS.keys():
        model_configs = get_configs_for_model(model_name)
        print(f"\n{model_name} ({len(model_configs)} configs):")
        for config in model_configs:
            print(f"  - {config.experiment_name}")
    
    print(f"\nStandard prompts: {len(STANDARD_PROMPTS)}")
    for i, prompt in enumerate(STANDARD_PROMPTS, 1):
        print(f"  P{i}: {prompt}")


if __name__ == "__main__":
    # Print configuration summary when run directly
    print_all_configs()
