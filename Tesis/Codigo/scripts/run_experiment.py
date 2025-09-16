#!/usr/bin/env python3
"""
Entry point script for running experiments with the experiment framework.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.experiment_runner import ExperimentRunner, run_quick_test, run_weighted_comparison

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LLM Evaluation Experiments')
    parser.add_argument('--experiment', type=str, default='quick_test',
                       choices=['quick_test', 'weighted_comparison', 'demo'],
                       help='Type of experiment to run')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename prefix')
    
    args = parser.parse_args()
    
    print(f"🧪 Running {args.experiment} experiment...")
    
    if args.experiment == 'quick_test':
        results_file = run_quick_test()
    elif args.experiment == 'weighted_comparison':
        results_file = run_weighted_comparison()
    elif args.experiment == 'demo':
        from src.evaluation.experiment_framework.demo_experiment import run_full_demo
        run_full_demo()
        return
    
    if args.output and results_file:
        # Rename the results file if custom output name provided
        import shutil
        new_path = results_file.replace(os.path.basename(results_file), f"{args.output}.parquet")
        shutil.move(results_file, new_path)
        print(f"Results saved as: {new_path}")
    
    print("✅ Experiment completed!")

if __name__ == "__main__":
    main()
