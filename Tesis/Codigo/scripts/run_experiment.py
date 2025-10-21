#!/usr/bin/env python3
"""
Entry point script for running experiments with the experiment framework.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.experiment_runner import ExperimentRunner, run_quick_factorial_test, run_single_model_test
from src.evaluation.experiment_framework.experiments.experiment_configs import STANDARD_PROMPTS

def main():
    """
    Main entry point for running LLM evaluation experiments.
    
    Supports three experiment types:
    - quick_test: Fast evaluation with basic metrics
    - single_model: Test single model with factorial design
    - demo: Interactive demonstration of the framework
    
    Optional output file renaming with --output flag.
    """
    
    import argparse
    
    def normalize_model_name(name):
        """Normalize model name to proper case."""
        model_mapping = {
            'phi3': 'Phi3',
            'qwen2': 'Qwen2',
            'qwen3': 'Qwen3',
            'smollm': 'SmolLM',
            'tinyllama': 'TinyLlama',
            'tinystories': 'TinyStories'
        }
        return model_mapping.get(name.lower(), name)
    
    parser = argparse.ArgumentParser(description='Run LLM Evaluation Experiments')
    parser.add_argument('--experiment', type=str, default='quick_test',
                       choices=['quick_test', 'all', 'multi_weight', 'Phi3', 'phi3', 'Qwen2', 'qwen2', 'Qwen3', 'qwen3', 'SmolLM', 'smollm', 'TinyLlama', 'tinyllama', 'demo'],
                       help='Type of experiment to run (use "all" to run all models, "multi_weight" to test different weight factors)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename prefix')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip automatic plot generation after experiment')
    parser.add_argument('--prompts', type=str, default=None,
                       help='Number of prompts to use (e.g., "5", "10", "all"). If not specified, will prompt interactively.')
    parser.add_argument('--weights', type=str, default='1.5,2.0,4.0',
                       help='Comma-separated weight factors for multi_weight experiment (default: "1.5,2.0,4.0")')
    
    args = parser.parse_args()
    
    # Normalize experiment name
    experiment = normalize_model_name(args.experiment) if args.experiment not in ['quick_test', 'multi_weight', 'demo'] else args.experiment
    
    print(f"🧪 Running {experiment} experiment...")
    
    # Determine whether to generate plots
    generate_plots = not args.no_plots
    
    # Determine number of prompts to use
    def get_prompts_count():
        """Get number of prompts from args or interactive input."""
        if args.prompts:
            if args.prompts.lower() == 'all':
                return len(STANDARD_PROMPTS)
            else:
                return int(args.prompts)
        else:
            # Interactive prompt
            amount_prompts = input("Enter the number of prompts to use (or 'all'): ")
            if amount_prompts.lower() == 'all':
                return len(STANDARD_PROMPTS)
            else:
                return int(amount_prompts)
    
    if experiment == 'quick_test':
        runner = ExperimentRunner()
        prompts = STANDARD_PROMPTS[:3]  # First 3 prompts only
        print(f"🧪 Running quick factorial test with {len(prompts)} prompts")
        results_file = runner.run_factorial_experiment(prompts, "quick_factorial_test", generate_plots=generate_plots)
    
    elif experiment == 'multi_weight':
        # Parse weight factors
        weight_factors = [float(w.strip()) for w in args.weights.split(',')]
        
        # Confirm before running if interactive
        if not args.prompts:
            print(f"⚖️  Weight factors to test: {weight_factors}")
            confirm = input("This will run a multi-weight experiment. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Experiment cancelled")
                return
        
        # Get number of prompts
        num_prompts = get_prompts_count()
        prompts = STANDARD_PROMPTS[:num_prompts]
        
        runner = ExperimentRunner()
        results_file = runner.run_multi_weight_experiment(
            prompts, 
            weight_factors=weight_factors,
            generate_plots=generate_plots
        )
    
    elif experiment == 'all':
        # Confirm before running if interactive
        if not args.prompts:
            confirm = input("This will run a large experiment. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Experiment cancelled")
                return
        
        # Get number of prompts and run all models
        num_prompts = get_prompts_count()
        prompts = STANDARD_PROMPTS[:num_prompts]
        
        runner = ExperimentRunner()
        runner.run_all_models_experiment(prompts, generate_plots=generate_plots)
        return
    
    elif experiment in ['Phi3', 'Qwen2', 'Qwen3', 'SmolLM', 'TinyLlama']:
        # Get number of prompts
        num_prompts = get_prompts_count()
        prompts = STANDARD_PROMPTS[:num_prompts]
        
        runner = ExperimentRunner()
        print(f"🧪 Running test for {experiment} with {len(prompts)} prompts")
        results_file = runner.run_single_model_experiment(experiment, prompts, generate_plots=generate_plots)
    
    elif experiment == 'demo':
        from src.evaluation.experiment_framework.demo_factorial_experiment import main as run_demo
        run_demo()
        return
    
    # Handle results output
    try:
        if args.output and results_file:
            import shutil
            new_path = results_file.replace(os.path.basename(results_file), f"{args.output}.parquet")
            shutil.move(results_file, new_path)
            print(f"Results saved as: {new_path}")
        elif results_file:
            print(f"Results saved to: {results_file}")
    except NameError:
        print("⚠️  No results generated")
    
    print("✅ Experiment completed!")

if __name__ == "__main__":
    main()
