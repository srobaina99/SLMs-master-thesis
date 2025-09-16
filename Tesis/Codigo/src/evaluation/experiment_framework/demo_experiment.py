#!/usr/bin/env python3
"""
Demo script to showcase the experiment framework capabilities.
Run this script to test the framework with your existing Qwen3 setup.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from core.experiment_runner import ExperimentRunner, run_quick_test, run_weighted_comparison
from core.data_models import ExperimentConfig
from prompts.english_learning_prompts import EnglishLearningPrompts


def demo_single_experiment():
    """Demonstrate running a single experiment."""
    print("=" * 60)
    print("DEMO 1: Single Experiment")
    print("=" * 60)
    
    runner = ExperimentRunner()
    
    # Create a simple configuration
    config = ExperimentConfig(
        system_prompt="You are a helpful English teacher for beginner students.",
        experiment_name="demo_single",
        description="Single experiment demo"
    )
    
    # Run single experiment
    result = runner.run_single_experiment(
        prompt="What does the word 'library' mean?",
        config=config,
        experiment_name="demo_single"
    )
    
    print(f"\nResult Summary:")
    print(f"  Response: {result.response[:100]}...")
    print(f"  Response time: {result.response_time_seconds:.2f} seconds")
    print(f"  Word count: {result.word_count}")
    print(f"  Flesch-Kincaid Grade: {result.flesch_kincaid_grade}")
    print(f"  Reading ease: {result.flesch_reading_ease}")
    
    return runner


def demo_batch_experiment():
    """Demonstrate running a batch of experiments."""
    print("\n" + "=" * 60)
    print("DEMO 2: Batch Experiment")
    print("=" * 60)
    
    runner = ExperimentRunner()
    
    # Use predefined prompts
    prompts = [
        "What does the word 'library' mean?",
        "How do I introduce myself to someone new?",
        "When do I use 'a' and when do I use 'an'?"
    ]
    
    config = ExperimentConfig(
        system_prompt=EnglishLearningPrompts.SYSTEM_PROMPTS['basic_teacher'],
        experiment_name="demo_batch",
        description="Batch experiment demo"
    )
    
    results = runner.run_batch_experiment(
        prompts=prompts,
        config=config,
        experiment_name="demo_batch"
    )
    
    print(f"\nBatch Results Summary:")
    print(f"  Total experiments: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"  Experiment {i}:")
        print(f"    Response time: {result.response_time_seconds:.2f}s")
        print(f"    Word count: {result.word_count}")
        print(f"    Grade level: {result.flesch_kincaid_grade:.1f}")
    
    return runner


def demo_parameter_sweep():
    """Demonstrate parameter sweep functionality."""
    print("\n" + "=" * 60)
    print("DEMO 3: Parameter Sweep")
    print("=" * 60)
    
    runner = ExperimentRunner()
    
    # Base configuration
    base_config = ExperimentConfig(
        system_prompt=EnglishLearningPrompts.SYSTEM_PROMPTS['basic_teacher'],
        experiment_name="demo_sweep"
    )
    
    # Test different system prompts
    parameter_variations = {
        'system_prompt': [
            EnglishLearningPrompts.SYSTEM_PROMPTS['basic_teacher'],
            EnglishLearningPrompts.SYSTEM_PROMPTS['detailed_teacher']
        ],
        'weighted_words_enabled': [False, True]
    }
    
    # Use a smaller set of prompts for demo
    prompts = ["What does the word 'library' mean?", "How do I say hello in English?"]
    
    results = runner.run_parameter_sweep(
        prompts=prompts,
        base_config=base_config,
        parameter_variations=parameter_variations,
        experiment_name="demo_sweep"
    )
    
    print(f"\nParameter Sweep Results:")
    print(f"  Total experiments: {len(results)}")
    
    # Group results by configuration
    config_groups = {}
    for result in results:
        config_key = f"weighted={result.weighted_words_enabled}"
        if config_key not in config_groups:
            config_groups[config_key] = []
        config_groups[config_key].append(result)
    
    for config_name, group_results in config_groups.items():
        avg_grade = sum(r.flesch_kincaid_grade for r in group_results) / len(group_results)
        avg_time = sum(r.response_time_seconds for r in group_results) / len(group_results)
        print(f"  {config_name}: Avg Grade={avg_grade:.1f}, Avg Time={avg_time:.2f}s")
    
    return runner


def demo_standard_experiments():
    """Demonstrate running standard predefined experiments."""
    print("\n" + "=" * 60)
    print("DEMO 4: Standard Experiments")
    print("=" * 60)
    
    runner = ExperimentRunner()
    
    # List available experiments
    available = runner.list_available_experiments()
    print(f"Available standard experiments: {available}")
    
    # Run the quick test
    print(f"\nRunning 'quick_test' standard experiment...")
    results = runner.run_standard_experiment("quick_test")
    
    print(f"\nStandard Experiment Results:")
    print(f"  Total experiments: {len(results)}")
    
    # Show some statistics
    if results:
        avg_grade = sum(r.flesch_kincaid_grade for r in results) / len(results)
        avg_words = sum(r.word_count for r in results) / len(results)
        avg_time = sum(r.response_time_seconds for r in results) / len(results)
        
        print(f"  Average grade level: {avg_grade:.2f}")
        print(f"  Average word count: {avg_words:.1f}")
        print(f"  Average response time: {avg_time:.2f}s")
    
    return runner


def demo_save_and_analyze():
    """Demonstrate saving results and getting analysis."""
    print("\n" + "=" * 60)
    print("DEMO 5: Save and Analyze Results")
    print("=" * 60)
    
    # Run a quick experiment
    runner = ExperimentRunner()
    runner.run_standard_experiment("quick_test")
    
    # Save results
    results_file = runner.save_results("demo_results")
    
    # Get summary statistics
    summary = runner.get_results_summary()
    
    print(f"\nSummary Statistics:")
    for metric, stats in summary.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std: {stats['std']:.2f}")
            print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
        else:
            print(f"  {metric}: {stats}")
    
    print(f"\nResults saved to: {results_file}")
    print("You can upload the Parquet file to Google Sheets for further analysis!")
    
    return runner, results_file


def run_full_demo():
    """Run all demo functions in sequence."""
    print("🚀 EXPERIMENT FRAMEWORK DEMO")
    print("This demo will showcase all framework capabilities.")
    print("Make sure your Qwen3 model is loaded before running!")
    
    try:
        # Check if model is available
        from utils.qwen3_wrapper import Qwen3ExperimentWrapper
        wrapper = Qwen3ExperimentWrapper()
        
        if not wrapper.model_loaded:
            print("\n❌ ERROR: Qwen3 model not loaded!")
            print("Please run the following first:")
            print("1. cd /Users/santiago/Documents/Personal/Tesis/Codigo")
            print("2. python -c \"from qwen3_mps.qwen3_weighted import *\"")
            print("3. Then run this demo again")
            return
        
        print("\n✅ Qwen3 model detected! Starting demo...")
        
        # Run all demos
        demo_single_experiment()
        demo_batch_experiment()
        demo_parameter_sweep()
        demo_standard_experiments()
        runner, results_file = demo_save_and_analyze()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final results saved to: {results_file}")
        print("\nNext steps:")
        print("1. Upload the .parquet file to Google Sheets")
        print("2. Create visualizations of your experiment results")
        print("3. Run your own experiments with different configurations!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure all dependencies are installed and Qwen3 model is loaded.")


if __name__ == "__main__":
    run_full_demo()
