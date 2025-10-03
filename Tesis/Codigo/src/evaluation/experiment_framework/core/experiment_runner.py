"""
Factorial Experiment Runner - Simplified interface for running factorial experiments.
Implements the 4×4×N experimental design from ExperimentSpecification.md.
"""

import os
import sys
from typing import List, Optional, Dict, Any

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
    
    Provides a clean interface to run the 4×4×N factorial design:
    - 4 models: Qwen2, Qwen3, TinyLlama, TinyStories
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
                                experiment_name: str = "factorial_experiment") -> str:
        """
        Run the complete factorial experiment.
        
        Args:
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            experiment_name: Name for this experiment run
            
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
        
        return files['specification_csv']
    
    def run_single_model_experiment(self, 
                                   model_name: str,
                                   prompts: Optional[List[str]] = None,
                                   experiment_name: Optional[str] = None) -> str:
        """
        Run factorial experiment for a single model only.
        
        Args:
            model_name: Name of model to test ("Qwen2", "Qwen3", "TinyLlama", "TinyStories")
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            experiment_name: Name for experiment (auto-generated if None)
            
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
    
    def clear_results(self):
        """Clear all stored results."""
        self.factorial_experiment.clear_results()
        print("🗑️  All results cleared")


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