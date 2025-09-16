"""
Main experiment runner that orchestrates the entire experiment framework.
Handles configuration, execution, and data collection for LLM evaluation experiments.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(project_root)

from .data_models import ExperimentConfig, ExperimentResult, ExperimentDataManager
from ..utils.qwen3_wrapper import Qwen3ExperimentWrapper
from ..prompts.english_learning_prompts import EnglishLearningPrompts, STANDARD_EXPERIMENT_CONFIGS


class ExperimentRunner:
    """
    Main class for running and managing LLM evaluation experiments.
    
    Coordinates between model wrapper, prompt templates, and data collection
    to provide a complete experiment framework.
    """
    
    def __init__(self, results_dir: str = "experiment_framework/results"):
        """
        Initialize the experiment runner.
        
        Args:
            results_dir: Directory to save experiment results
        """
        self.results_dir = results_dir
        self.data_manager = ExperimentDataManager()
        self.model_wrapper = Qwen3ExperimentWrapper()
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"ExperimentRunner initialized. Results will be saved to: {self.results_dir}")
    
    def run_single_experiment(self, 
                            prompt: str,
                            config: ExperimentConfig,
                            experiment_name: str = "single_experiment") -> ExperimentResult:
        """
        Run a single prompt-response experiment.
        
        Args:
            prompt: The input prompt for the model
            config: Experiment configuration
            experiment_name: Name for this experiment
            
        Returns:
            ExperimentResult containing all data and metrics
        """
        
        print(f"Running experiment: {experiment_name}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Generate response with metrics
        result_data = self.model_wrapper.generate_response_with_metrics(
            user_input=prompt,
            system_prompt=config.system_prompt,
            enable_thinking=config.enable_thinking,
            weighted_words=[] if not config.weighted_words_enabled else [".", ",", "<|im_end|>"],
            weight_factor=config.weight_factor,
            verbose=config.verbose
        )
        
        if not result_data['generation_successful']:
            print(f"Generation failed: {result_data['error_message']}")
            # Still create result object for failed generations
        
        # Create structured result
        experiment_result = ExperimentResult.create_from_response(
            prompt=prompt,
            response=result_data['response'],
            config=config,
            response_time=result_data['response_time_seconds'],
            text_metrics=result_data['text_metrics'],
            experiment_name=experiment_name,
            cleaned_response=result_data.get('cleaned_response', '')
        )
        
        # Add to data manager
        self.data_manager.add_result(experiment_result)
        
        print(f"Experiment completed. Response time: {result_data['response_time_seconds']:.2f}s")
        
        return experiment_result
    
    def run_batch_experiment(self,
                           prompts: List[str],
                           config: ExperimentConfig,
                           experiment_name: str = "batch_experiment") -> List[ExperimentResult]:
        """
        Run multiple prompts with the same configuration.
        
        Args:
            prompts: List of prompts to test
            config: Experiment configuration
            experiment_name: Name for this batch experiment
            
        Returns:
            List of ExperimentResult objects
        """
        
        print(f"Running batch experiment: {experiment_name}")
        print(f"Processing {len(prompts)} prompts with config: {config.experiment_name}")
        
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
            
            # Create unique name for each prompt in the batch
            prompt_experiment_name = f"{experiment_name}_prompt_{i+1}"
            
            result = self.run_single_experiment(
                prompt=prompt,
                config=config,
                experiment_name=prompt_experiment_name
            )
            
            results.append(result)
        
        print(f"\nBatch experiment completed. Processed {len(results)} prompts.")
        return results
    
    def run_parameter_sweep(self,
                          prompts: List[str],
                          base_config: ExperimentConfig,
                          parameter_variations: Dict[str, List[Any]],
                          experiment_name: str = "parameter_sweep") -> List[ExperimentResult]:
        """
        Run experiments with different parameter combinations.
        
        Args:
            prompts: List of prompts to test
            base_config: Base configuration to modify
            parameter_variations: Dict of parameter names to lists of values to test
            experiment_name: Name for this parameter sweep
            
        Returns:
            List of all ExperimentResult objects
        """
        
        print(f"Running parameter sweep: {experiment_name}")
        print(f"Parameters to vary: {list(parameter_variations.keys())}")
        
        all_results = []
        
        # Generate all parameter combinations
        param_names = list(parameter_variations.keys())
        param_values = list(parameter_variations.values())
        
        # Simple cartesian product for parameter combinations
        def generate_combinations(values_lists):
            if not values_lists:
                return [[]]
            
            first_values = values_lists[0]
            rest_combinations = generate_combinations(values_lists[1:])
            
            combinations = []
            for value in first_values:
                for rest_combo in rest_combinations:
                    combinations.append([value] + rest_combo)
            
            return combinations
        
        param_combinations = generate_combinations(param_values)
        
        print(f"Testing {len(param_combinations)} parameter combinations with {len(prompts)} prompts each")
        print(f"Total experiments: {len(param_combinations) * len(prompts)}")
        
        for combo_idx, param_combo in enumerate(param_combinations):
            print(f"\n=== Parameter Combination {combo_idx + 1}/{len(param_combinations)} ===")
            
            # Create modified config
            modified_config = ExperimentConfig(**base_config.to_dict())
            
            # Apply parameter modifications
            combo_description = []
            for param_name, param_value in zip(param_names, param_combo):
                setattr(modified_config, param_name, param_value)
                combo_description.append(f"{param_name}={param_value}")
            
            combo_name = f"{experiment_name}_" + "_".join(combo_description)
            modified_config.experiment_name = combo_name
            
            print(f"Configuration: {', '.join(combo_description)}")
            
            # Run batch with this configuration
            batch_results = self.run_batch_experiment(
                prompts=prompts,
                config=modified_config,
                experiment_name=combo_name
            )
            
            all_results.extend(batch_results)
        
        print(f"\nParameter sweep completed. Total results: {len(all_results)}")
        return all_results
    
    def run_standard_experiment(self, experiment_type: str = "quick_test") -> List[ExperimentResult]:
        """
        Run one of the predefined standard experiments.
        
        Args:
            experiment_type: Type of standard experiment to run
            
        Returns:
            List of ExperimentResult objects
        """
        
        if experiment_type not in STANDARD_EXPERIMENT_CONFIGS:
            available_types = list(STANDARD_EXPERIMENT_CONFIGS.keys())
            raise ValueError(f"Unknown experiment type: {experiment_type}. Available: {available_types}")
        
        experiment_config = STANDARD_EXPERIMENT_CONFIGS[experiment_type]
        
        # Create configuration
        config = ExperimentConfig(
            system_prompt=experiment_config['system_prompt'],
            experiment_name=experiment_type,
            description=experiment_config['description']
        )
        
        print(f"Running standard experiment: {experiment_type}")
        print(f"Description: {experiment_config['description']}")
        
        return self.run_batch_experiment(
            prompts=experiment_config['prompts'],
            config=config,
            experiment_name=experiment_type
        )
    
    def save_results(self, filename_prefix: str = None) -> str:
        """
        Save all collected results to Parquet file.
        
        Args:
            filename_prefix: Optional prefix for the filename
            
        Returns:
            Path to the saved file
        """
        
        if not filename_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"experiment_results_{timestamp}"
        
        parquet_path = os.path.join(self.results_dir, f"{filename_prefix}.parquet")
        csv_path = os.path.join(self.results_dir, f"{filename_prefix}.csv")
        
        # Save in both formats
        self.data_manager.save_to_parquet(parquet_path)
        self.data_manager.save_to_csv(csv_path)
        
        # Also save summary statistics
        summary = self.data_manager.get_summary_stats()
        summary_path = os.path.join(self.results_dir, f"{filename_prefix}_summary.json")
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Results saved:")
        print(f"  Parquet: {parquet_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Summary: {summary_path}")
        
        return parquet_path
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all collected results."""
        return self.data_manager.get_summary_stats()
    
    def clear_results(self):
        """Clear all stored results."""
        self.data_manager.clear()
        print("All results cleared.")
    
    def list_available_experiments(self) -> List[str]:
        """List all available standard experiment types."""
        return list(STANDARD_EXPERIMENT_CONFIGS.keys())


# Convenience functions for quick usage
def run_quick_test() -> str:
    """Run a quick test experiment and save results."""
    runner = ExperimentRunner()
    
    print("Running quick test experiment...")
    runner.run_standard_experiment("quick_test")
    
    results_file = runner.save_results("quick_test")
    
    # Print summary
    summary = runner.get_results_summary()
    print(f"\nExperiment Summary:")
    print(f"  Total experiments: {summary.get('total_experiments', 0)}")
    print(f"  Unique prompts: {summary.get('unique_prompts', 0)}")
    print(f"  Average response time: {summary.get('response_time_seconds', {}).get('mean', 0):.2f}s")
    
    return results_file


def run_weighted_comparison() -> str:
    """Compare performance with and without weighted words."""
    runner = ExperimentRunner()
    
    # Base configuration
    base_config = ExperimentConfig(
        system_prompt=EnglishLearningPrompts.SYSTEM_PROMPTS['basic_teacher'],
        experiment_name="weighted_comparison"
    )
    
    # Test with weighted words on/off
    parameter_variations = {
        'weighted_words_enabled': [False, True],
        'weight_factor': [1.0, 1.5, 2.0]
    }
    
    prompts = EnglishLearningPrompts.get_basic_test_set()
    
    print("Running weighted words comparison experiment...")
    runner.run_parameter_sweep(
        prompts=prompts,
        base_config=base_config,
        parameter_variations=parameter_variations,
        experiment_name="weighted_comparison"
    )
    
    results_file = runner.save_results("weighted_comparison")
    
    # Print summary
    summary = runner.get_results_summary()
    print(f"\nExperiment Summary:")
    print(f"  Total experiments: {summary.get('total_experiments', 0)}")
    print(f"  Average Flesch-Kincaid Grade: {summary.get('flesch_kincaid_grade', {}).get('mean', 0):.2f}")
    
    return results_file


if __name__ == "__main__":
    # Run quick test when executed directly
    print("Running quick test experiment...")
    results_file = run_quick_test()
    print(f"\nResults saved to: {results_file}")
    print("You can now upload the Parquet file to Google Sheets for analysis.")
