"""
Demo script showing how to use the factorial experiment framework.
Demonstrates the complete 4×4×N experimental design from ExperimentSpecification.md.
"""

import sys
import os
from typing import List

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(project_root)

from src.evaluation.experiment_framework.experiments.factorial_experiment import FactorialExperiment
from src.evaluation.experiment_framework.experiments.experiment_configs import STANDARD_PROMPTS, print_all_configs
from src.evaluation.experiment_framework.core.experiment_runner import ExperimentRunner


def demo_model_status():
    """Demo: Check which models are available and loaded."""
    print("=" * 60)
    print("🔍 CHECKING MODEL STATUS")
    print("=" * 60)
    
    experiment = FactorialExperiment()
    status = experiment.get_model_status()
    
    print(f"Model Status Report:")
    for model_name, info in status.items():
        loaded_status = "✅ LOADED" if info['loaded'] else "❌ NOT LOADED"
        print(f"  {model_name}: {loaded_status}")
        
        if not info['loaded'] and info['error']:
            print(f"    Error: {info['error']}")
        
        model_info = info['model_info']
        print(f"    Vocab size: {model_info['vocab_size']} words")
        print(f"    Supports weighting: {model_info['supports_weighting']}")
        print(f"    Supports context prompting: {model_info['supports_context_prompting']}")
        print()


def demo_configurations():
    """Demo: Show all available configurations."""
    print("=" * 60)
    print("⚙️  EXPERIMENT CONFIGURATIONS")
    print("=" * 60)
    
    print_all_configs()


def demo_single_model_experiment():
    """Demo: Run factorial experiment for a single model."""
    print("=" * 60)
    print("🤖 SINGLE MODEL FACTORIAL EXPERIMENT")
    print("=" * 60)
    
    # Use a subset of prompts for demo
    demo_prompts = STANDARD_PROMPTS[:3]  # First 3 prompts only
    
    print(f"Running factorial experiment for Qwen3 with {len(demo_prompts)} prompts...")
    print("This will test all 4 intervention combinations:")
    print("  1. Control (no interventions)")
    print("  2. Weighting only")
    print("  3. Context prompting only")
    print("  4. Both interventions")
    print()
    
    experiment = FactorialExperiment()
    
    try:
        df = experiment.run_single_model_experiment("Qwen3", demo_prompts)
        files = experiment.save_results("demo_single_model")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Generated {len(df)} results")
        print(f"💾 Results saved to: {files['specification_csv']}")
        
        # Show sample of results
        if len(df) > 0:
            print(f"\n📋 Sample results:")
            print(df[['model', 'config_weighting', 'config_prompting', 'prompt_id', 'flesch_kincaid_grade']].head())
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This might be because the Qwen3 model is not loaded.")
        print("Make sure to load the model first before running experiments.")


def demo_full_factorial_experiment():
    """Demo: Run complete factorial experiment (all models, all configs)."""
    print("=" * 60)
    print("🚀 FULL FACTORIAL EXPERIMENT")
    print("=" * 60)
    
    # Use subset of prompts for demo to keep it manageable
    demo_prompts = STANDARD_PROMPTS[:2]  # First 2 prompts only
    
    print(f"Running FULL factorial experiment with {len(demo_prompts)} prompts...")
    print("This will test:")
    print("  - 5 models: Qwen2, Qwen3, TinyLlama, Phi3, SmolLM")
    print("  - 4 intervention combinations per model")
    print(f"  - Total experiments: {len(demo_prompts)} × 4 × 4 = {len(demo_prompts) * 16}")
    print()
    print("⚠️  WARNING: This may take several minutes and requires all models to be loaded!")
    
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Demo cancelled.")
        return
    
    experiment = FactorialExperiment()
    
    try:
        df = experiment.run_full_experiment(demo_prompts, "demo_full_factorial")
        files = experiment.save_results("demo_full_factorial")
        
        print(f"\n🎉 Full factorial demo completed!")
        print(f"📊 Generated {len(df)} results")
        print(f"💾 Results saved to: {files['specification_csv']}")
        
        # Show summary statistics
        if len(df) > 0:
            print(f"\n📈 Summary by model:")
            summary = df.groupby('model')['flesch_kincaid_grade'].agg(['count', 'mean', 'std']).round(2)
            print(summary)
            
            print(f"\n📈 Summary by intervention:")
            df['intervention'] = df['config_weighting'].astype(str) + '_' + df['config_prompting'].astype(str)
            intervention_summary = df.groupby('intervention')['flesch_kincaid_grade'].agg(['count', 'mean', 'std']).round(2)
            print(intervention_summary)
        
    except Exception as e:
        print(f"❌ Full factorial demo failed: {e}")
        print("This might be because some models are not loaded.")


def demo_experiment_runner_integration():
    """Demo: Use ExperimentRunner with factorial experiments."""
    print("=" * 60)
    print("🔧 EXPERIMENT RUNNER INTEGRATION")
    print("=" * 60)
    
    print("Using ExperimentRunner to run factorial experiment...")
    
    runner = ExperimentRunner()
    
    # Check model status
    print("Checking model status...")
    status = runner.get_model_status()
    loaded_models = [name for name, info in status.items() if info['loaded']]
    
    print(f"Available models: {loaded_models}")
    
    if not loaded_models:
        print("❌ No models loaded. Cannot run demo.")
        return
    
    # Run factorial experiment for first available model
    model_name = loaded_models[0]
    demo_prompts = STANDARD_PROMPTS[:2]
    
    try:
        results_file = runner.run_single_model_factorial(model_name, demo_prompts)
        print(f"✅ ExperimentRunner demo completed!")
        print(f"💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ ExperimentRunner demo failed: {e}")


def main():
    """Main demo function with interactive menu."""
    print("🧪 FACTORIAL EXPERIMENT FRAMEWORK DEMO")
    print("=" * 60)
    print("This demo shows the complete factorial experiment framework")
    print("implementing the design from ExperimentSpecification.md")
    print()
    
    while True:
        print("\nAvailable demos:")
        print("1. Check model status")
        print("2. Show experiment configurations")
        print("3. Run single model factorial experiment")
        print("4. Run full factorial experiment (all models)")
        print("5. Test ExperimentRunner integration")
        print("0. Exit")
        
        choice = input("\nSelect demo (0-5): ").strip()
        
        if choice == "0":
            print("👋 Demo finished!")
            break
        elif choice == "1":
            demo_model_status()
        elif choice == "2":
            demo_configurations()
        elif choice == "3":
            demo_single_model_experiment()
        elif choice == "4":
            demo_full_factorial_experiment()
        elif choice == "5":
            demo_experiment_runner_integration()
        else:
            print("❌ Invalid choice. Please select 0-5.")


if __name__ == "__main__":
    main()
