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
    "Can you describe what happens in the morning?",
    "What does 'happy' mean?",
    "How do you say goodbye in English?",
    "What is the weather like today?",
    "What is a 'friend'?",
    "What do you do at school?",
    "What is the difference between 'hot' and 'cold'?",
    "Can you describe your family?",
    "What foods do you eat for lunch?"
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
    "TinyLlama": {
        "model_name": "TinyLlama",
        "model_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"  # Using llama.cpp GGUF
    }
}


def create_factorial_configs() -> List[ExperimentConfig]:
    """
    Create all factorial experiment configurations.
    
    Returns 4 models × 4 intervention combinations = 16 configurations (excludes TinyStories)
    
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
                weight_factor=1.5,  # Optimal weighting factor from multi-weight experiment
                enable_thinking=False,
                verbose=False,
                
                # Generation parameters
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                max_new_tokens=200,  # ~150 words max for beginner-appropriate responses
                
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


def create_beam_search_configs(beam_width: int = 4, use_prompting: bool = True) -> List[ExperimentConfig]:
    """
    Create experiment configurations for beam search experiments.
    
    Beam search uses contextual prompting by default and tests standard beam selection
    criteria: A1 word ratio and cumulative log probability.
    
    Args:
        beam_width: Number of beams (default: 4)
        use_prompting: Whether to use contextual prompting intervention (default: True)
    
    Returns:
        List of ExperimentConfig objects for beam search experiment
    """
    configs = []
    
    # Base system prompt for English learning
    system_prompt = "You are a helpful English teacher for beginner students. Answer with a paragraph only with plain text"
    
    # Create configs for Qwen3 with beam search (can extend to all models)
    # Using first 5 prompts as specified
    beam_search_model = "Qwen3"
    
    # Config for beam search with A1 ratio selection
    configs.append(ExperimentConfig(
        model_name=beam_search_model,
        model_id="ggml-org/Qwen3-0.6B-GGUF",
        system_prompt=system_prompt,
        config_weighting=False,  # Weighting not used during beam generation
        config_prompting=use_prompting,  # Use contextual prompting
        
        weighted_words_enabled=False,
        weight_factor=1.5,
        enable_thinking=False,
        verbose=False,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=200,
        
        experiment_name=f"{beam_search_model}_beam_search_a1_ratio",
        description=f"Beam search (n={beam_width}) with A1 word ratio selection"
    ))
    
    # Config for beam search with max probability selection
    configs.append(ExperimentConfig(
        model_name=beam_search_model,
        model_id="ggml-org/Qwen3-0.6B-GGUF",
        system_prompt=system_prompt,
        config_weighting=False,
        config_prompting=use_prompting,
        
        weighted_words_enabled=False,
        weight_factor=1.5,
        enable_thinking=False,
        verbose=False,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=200,
        
        experiment_name=f"{beam_search_model}_beam_search_max_probability",
        description=f"Beam search (n={beam_width}) with max cumulative log probability selection"
    ))
    
    return configs


def create_multi_weight_configs(weight_factors: List[float] = [1.5, 2.0, 4.0]) -> List[ExperimentConfig]:
    """
    Create experiment configurations testing multiple weight factors.
    
    Simplified design: only tests weighting intervention with different factors.
    For each model, creates configs with weighting at each specified factor.
    
    Args:
        weight_factors: List of weight factors to test (default: [1.5, 2.0, 4.0])
    
    Returns:
        List of ExperimentConfig objects for multi-weight experiment
    """
    configs = []
    
    # Base system prompt for English learning
    system_prompt = "You are a helpful English teacher for beginner students. Answer with a paragraph only with plain text"
    
    for model_name, model_info in MODEL_CONFIGS.items():
        # Weighting only - for each weight factor
        for weight in weight_factors:
            weight_str = str(weight).replace('.', '_')
            configs.append(ExperimentConfig(
                model_name=model_info["model_name"],
                model_id=model_info["model_id"],
                system_prompt=system_prompt,
                config_weighting=True,
                config_prompting=False,
                weighted_words_enabled=True,
                weight_factor=weight,
                enable_thinking=False,
                verbose=False,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                max_new_tokens=200,
                experiment_name=f"{model_name}_weighted_{weight_str}",
                description=f"Multi-weight experiment: {model_name} with weighting (factor={weight})"
            ))
    
    return configs


def get_configs_for_model(model_name: str) -> List[ExperimentConfig]:
    """
    Get all configurations for a specific model.
    
    Args:
        model_name: Name of the model ("Qwen2", "Qwen3", "TinyLlama", "Phi3")
        
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
