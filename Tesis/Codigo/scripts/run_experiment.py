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
            'qwen2': 'Qwen2',
            'qwen3': 'Qwen3', 
            'tinyllama': 'TinyLlama',
            'tinystories': 'TinyStories'
        }
        return model_mapping.get(name.lower(), name)
    
    parser = argparse.ArgumentParser(description='Run LLM Evaluation Experiments')
    parser.add_argument('--experiment', type=str, default='quick_test',
                       choices=['quick_test', 'Qwen2', 'qwen2', 'Qwen3', 'qwen3', 'TinyLlama', 'tinyllama', 'TinyStories', 'tinystories', 'demo'],
                       help='Type of experiment to run')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename prefix')
    
    args = parser.parse_args()
    
    # Normalize experiment name
    experiment = normalize_model_name(args.experiment) if args.experiment not in ['quick_test', 'demo'] else args.experiment
    
    print(f"🧪 Running {experiment} experiment...")
    
    if experiment == 'quick_test':
        results_file = run_quick_factorial_test()
    elif experiment in ['Qwen2', 'Qwen3', 'TinyLlama', 'TinyStories']:
        results_file = run_single_model_test(experiment)
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
