#!/usr/bin/env python3
"""
Quick test script for multi-weight experiment configuration.
Demonstrates the new multi-weight experiment setup without running actual experiments.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.framework.experiments.experiment_configs import (
    create_multi_weight_configs, 
    MODEL_CONFIGS
)


def main():
    """Test multi-weight configuration generation."""
    
    print("=" * 70)
    print("MULTI-WEIGHT EXPERIMENT CONFIGURATION TEST")
    print("=" * 70)
    
    # Test with default weights
    print("\n📊 Testing with default weight factors: [1.5, 2.0, 4.0]")
    configs = create_multi_weight_configs()
    
    print(f"\n✅ Generated {len(configs)} configurations")
    print(f"   Models: {len(MODEL_CONFIGS)}")
    print(f"   Configs per model: {len(configs) // len(MODEL_CONFIGS)}")
    print(f"   Breakdown: 1 control + 1 prompted + 3 weighted + 3 weighted+prompted = 8")
    
    # Show configurations for one model
    print("\n" + "=" * 70)
    print("EXAMPLE: Qwen2 Configurations")
    print("=" * 70)
    
    qwen2_configs = [c for c in configs if c.model_name == "Qwen2"]
    
    for i, config in enumerate(qwen2_configs, 1):
        weighting_status = f"✓ (factor={config.weight_factor})" if config.config_weighting else "✗"
        prompting_status = "✓" if config.config_prompting else "✗"
        
        print(f"\n{i}. {config.experiment_name}")
        print(f"   Weighting: {weighting_status}")
        print(f"   Prompting: {prompting_status}")
        print(f"   Description: {config.description}")
    
    # Test with custom weights
    print("\n" + "=" * 70)
    print("CUSTOM WEIGHT FACTORS TEST")
    print("=" * 70)
    
    custom_weights = [1.0, 2.5, 5.0, 10.0]
    print(f"\n📊 Testing with custom weight factors: {custom_weights}")
    
    custom_configs = create_multi_weight_configs(weight_factors=custom_weights)
    print(f"\n✅ Generated {len(custom_configs)} configurations")
    print(f"   Configs per model: {len(custom_configs) // len(MODEL_CONFIGS)}")
    print(f"   Breakdown: 1 control + 1 prompted + {len(custom_weights)} weighted + {len(custom_weights)} weighted+prompted = {2 + 2*len(custom_weights)}")
    
    # Show weight distribution
    print("\n" + "=" * 70)
    print("WEIGHT FACTOR DISTRIBUTION")
    print("=" * 70)
    
    weight_counts = {}
    for config in custom_configs:
        if config.config_weighting:
            weight = config.weight_factor
            weight_counts[weight] = weight_counts.get(weight, 0) + 1
    
    print("\nWeight factor usage across all models:")
    for weight in sorted(weight_counts.keys()):
        count = weight_counts[weight]
        print(f"   {weight}: {count} configs ({count // len(MODEL_CONFIGS)} per model)")
    
    print("\n" + "=" * 70)
    print("✅ Multi-weight configuration test completed successfully!")
    print("=" * 70)
    
    print("\n💡 Usage examples:")
    print("   # Run with default weights (1.5, 2.0, 4.0):")
    print("   python scripts/run_experiment.py --experiment multi_weight --prompts 5")
    print()
    print("   # Run with custom weights:")
    print("   python scripts/run_experiment.py --experiment multi_weight --prompts 5 --weights 1.5,3.0,6.0")
    print()
    print("   # Run with single weight for comparison:")
    print("   python scripts/run_experiment.py --experiment multi_weight --prompts 3 --weights 2.0")
    print()


if __name__ == "__main__":
    main()

