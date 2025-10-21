"""
Factorial Experiment Runner - Simplified interface for running factorial experiments.
Implements the 4×4×N experimental design from ExperimentSpecification.md.
"""

import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.data_models import ExperimentConfig, ExperimentResult, ExperimentDataManager
from src.evaluation.experiment_framework.experiments.factorial_experiment import FactorialExperiment
from src.evaluation.experiment_framework.experiments.experiment_configs import STANDARD_PROMPTS


class ExperimentRunner:
    """
    Simplified experiment runner focused on factorial experiments.
    
    Provides a clean interface to run the factorial design:
    - 5 models: Qwen2, Qwen3, TinyLlama, Phi3, SmolLM
    - 4 intervention combinations: control, weighting, prompting, both
    - N prompts: configurable set of test prompts
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize the factorial experiment runner.
        
        Args:
            results_dir: Directory to save experiment results (defaults to framework results dir)
        """
        # Use absolute path to results directory
        if results_dir is None:
            framework_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(framework_dir, "results")
        
        self.results_dir = os.path.abspath(results_dir)
        self.factorial_experiment = FactorialExperiment(self.results_dir)
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✅ ExperimentRunner initialized for factorial experiments")
        print(f"📁 Results directory: {self.results_dir}")
    
    def run_factorial_experiment(self, 
                                prompts: Optional[List[str]] = None,
                                experiment_name: str = "factorial_experiment",
                                generate_plots: bool = True) -> str:
        """
        Run the complete factorial experiment.
        
        Args:
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            experiment_name: Name for this experiment run
            generate_plots: Whether to automatically generate visualization plots
            
        Returns:
            Path to saved results file in specification format
        """
        print(f"🚀 Starting factorial experiment: {experiment_name}")
        
        # Use standard prompts if none provided
        if prompts is None:
            prompts = STANDARD_PROMPTS
            print(f"📝 Using {len(prompts)} standard prompts")
        else:
            print(f"📝 Using {len(prompts)} custom prompts")
        
        # Run the factorial experiment
        df = self.factorial_experiment.run_full_experiment(prompts, experiment_name)
        
        # Save results
        files = self.factorial_experiment.save_results(experiment_name)
        
        print(f"✅ Factorial experiment completed!")
        print(f"📊 Generated {len(df)} results")
        print(f"💾 Specification CSV: {files['specification_csv']}")
        
        # Generate visualizations if requested
        if generate_plots:
            self._generate_visualizations(files['specification_csv'])
        
        return files['specification_csv']
    
    def run_single_model_experiment(self, 
                                   model_name: str,
                                   prompts: Optional[List[str]] = None,
                                   experiment_name: Optional[str] = None,
                                   generate_plots: bool = True) -> str:
        """
        Run factorial experiment for a single model only.
        
        Args:
            model_name: Name of model to test ("Qwen2", "Qwen3", "TinyLlama", "Phi3", "SmolLM")
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            experiment_name: Name for experiment (auto-generated if None)
            generate_plots: Whether to automatically generate visualization plots
            
        Returns:
            Path to saved results file in specification format
        """
        if experiment_name is None:
            experiment_name = f"{model_name}_factorial"
        
        print(f"🚀 Starting single model factorial experiment: {model_name}")
        
        # Use standard prompts if none provided
        if prompts is None:
            prompts = STANDARD_PROMPTS
            print(f"📝 Using {len(prompts)} standard prompts")
        
        # Run single model experiment
        df = self.factorial_experiment.run_single_model_experiment(model_name, prompts)
        
        # Save results
        files = self.factorial_experiment.save_results(experiment_name)
        
        print(f"✅ Single model experiment completed for {model_name}!")
        print(f"📊 Generated {len(df)} results")
        print(f"💾 Specification CSV: {files['specification_csv']}")
        
        # Generate visualizations if requested
        if generate_plots:
            self._generate_visualizations(files['specification_csv'])
        
        return files['specification_csv']
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all available models.
        
        Returns:
            Dictionary with model status information
        """
        return self.factorial_experiment.get_model_status()
    
    def get_available_prompts(self) -> List[str]:
        """
        Get the list of standard prompts.
        
        Returns:
            List of standard prompts for experiments
        """
        return STANDARD_PROMPTS.copy()
    
    def run_all_models_experiment(self,
                                  prompts: Optional[List[str]] = None,
                                  generate_plots: bool = True,
                                  confirm: bool = True) -> Dict[str, str]:
        """
        Run factorial experiment for all available models.
        
        Args:
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            generate_plots: Whether to automatically generate visualization plots
            confirm: Whether to ask for confirmation before running (ignored if prompts specified)
            
        Returns:
            Dictionary mapping model names to their results file paths
        """
        all_models = ['Qwen2', 'Qwen3', 'TinyLlama', 'Phi3', 'SmolLM']
        
        # Use standard prompts if none provided
        if prompts is None:
            prompts = STANDARD_PROMPTS
        
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING ALL MODELS EXPERIMENT")
        print(f"{'='*60}")
        print(f"📊 Models: {', '.join(all_models)}")
        print(f"📝 Prompts: {len(prompts)}")
        print(f"🎨 Plots: {'Enabled' if generate_plots else 'Disabled'}")
        print(f"📈 Total experiments: {len(all_models)} models × 4 configs × {len(prompts)} prompts = {len(all_models) * 4 * len(prompts)}")
        print(f"{'='*60}\n")
        
        results_files = {}
        
        for idx, model in enumerate(all_models, 1):
            print(f"\n{'='*60}")
            print(f"🤖 [{idx}/{len(all_models)}] Running {model}")
            print(f"{'='*60}")
            
            try:
                results_file = self.run_single_model_experiment(
                    model, 
                    prompts, 
                    experiment_name=f"{model}_full_experiment",
                    generate_plots=generate_plots
                )
                results_files[model] = results_file
                print(f"✅ {model} completed successfully!")
            except Exception as e:
                print(f"❌ {model} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🎉 ALL MODELS EXPERIMENT COMPLETED")
        print(f"{'='*60}")
        print(f"✅ Successfully completed: {len(results_files)}/{len(all_models)} models")
        print(f"\n📊 Results files:")
        for model, file_path in results_files.items():
            print(f"   {model}: {file_path}")
        print(f"{'='*60}\n")
        
        return results_files
    
    def run_multi_weight_experiment(self,
                                    prompts: Optional[List[str]] = None,
                                    weight_factors: List[float] = [1.5, 2.0, 4.0],
                                    experiment_name: str = "multi_weight_experiment",
                                    generate_plots: bool = True) -> str:
        """
        Run experiment testing multiple weight factors.
        
        Tests different weighting strengths to understand the effect of weight_factor parameter.
        
        Args:
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            weight_factors: List of weight factors to test (default: [1.5, 2.0, 4.0])
            experiment_name: Name for this experiment run
            generate_plots: Whether to automatically generate visualization plots
            
        Returns:
            Path to saved results file in specification format
        """
        print(f"🚀 Starting multi-weight experiment: {experiment_name}")
        print(f"⚖️  Testing weight factors: {weight_factors}")
        
        # Use standard prompts if none provided
        if prompts is None:
            prompts = STANDARD_PROMPTS
            print(f"📝 Using {len(prompts)} standard prompts")
        else:
            print(f"📝 Using {len(prompts)} custom prompts")
        
        # Run the multi-weight experiment
        df = self.factorial_experiment.run_multi_weight_experiment(prompts, weight_factors, experiment_name)
        
        # Save results
        files = self.factorial_experiment.save_results(experiment_name)
        
        print(f"✅ Multi-weight experiment completed!")
        print(f"📊 Generated {len(df)} results")
        print(f"💾 Specification CSV: {files['specification_csv']}")
        
        # Generate visualizations if requested
        if generate_plots:
            self._generate_visualizations(files['specification_csv'])
        
        return files['specification_csv']
    
    def clear_results(self):
        """Clear all stored results."""
        self.factorial_experiment.clear_results()
        print("🗑️  All results cleared")
    
    def _generate_visualizations(self, csv_path: str):
        """
        Generate visualization plots for experiment results.
        
        Args:
            csv_path: Path to the CSV file with results
        """
        try:
            print("\n🎨 Generating visualizations...")
            
            # Import visualization module
            visualize_module_path = Path(self.results_dir) / "visualize_results.py"
            
            if not visualize_module_path.exists():
                print("⚠️  Visualization module not found. Skipping plot generation.")
                return
            
            # Import the plot_all_metrics function
            import importlib.util
            spec = importlib.util.spec_from_file_location("visualize_results", visualize_module_path)
            visualize_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(visualize_module)
            
            # Generate plots
            output_dir = visualize_module.plot_all_metrics(csv_path)
            print(f"✅ Visualizations saved to: {output_dir}/")
            
        except Exception as e:
            print(f"⚠️  Failed to generate visualizations: {e}")
            print("   You can manually generate plots by running:")
            print(f"   python {self.results_dir}/visualize_results.py {csv_path}")


# Convenience functions for quick usage
def run_quick_factorial_test(prompts: Optional[List[str]] = None) -> str:
    """
    Run a quick factorial experiment test.
    
    Args:
        prompts: List of prompts to test (uses first 3 standard prompts if None)
        
    Returns:
        Path to results file
    """
    runner = ExperimentRunner()
    
    # Use subset of prompts for quick test
    if prompts is None:
        prompts = STANDARD_PROMPTS[:3]  # First 3 prompts only
    
    print(f"🧪 Running quick factorial test with {len(prompts)} prompts")
    return runner.run_factorial_experiment(prompts, "quick_factorial_test")


def run_single_model_test(model_name: str, prompts: Optional[List[str]] = None) -> str:
    """
    Run a quick test for a single model.
    
    Args:
        model_name: Name of model to test
        prompts: List of prompts to test (uses first 5 standard prompts if None)
        
    Returns:
        Path to results file
    """
    runner = ExperimentRunner()
    
    # Use subset of prompts for quick test
    if prompts is None:
        prompts = STANDARD_PROMPTS[:5]  # First 5 prompts only
    
    print(f"🧪 Running quick test for {model_name} with {len(prompts)} prompts")
    return runner.run_single_model_experiment(model_name, prompts)


if __name__ == "__main__":
    # Run quick test when executed directly
    print("🧪 Running quick factorial test...")
    results_file = run_quick_factorial_test()
    print(f"✅ Results saved to: {results_file}")