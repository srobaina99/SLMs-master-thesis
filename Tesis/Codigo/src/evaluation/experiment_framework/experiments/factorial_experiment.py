"""
Factorial experiment runner implementing the 4×4×N experimental design.
Runs all model-intervention combinations across multiple prompts.
"""

import sys
import os
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.data_models import ExperimentConfig, ExperimentResult, ExperimentDataManager
from src.evaluation.experiment_framework.models import (
    Qwen2Wrapper, Qwen3Wrapper, TinyLlamaWrapper, TinyStoriesWrapper
)
from src.evaluation.text_complexity.text_evaluator import TextEvaluator
from .experiment_configs import create_factorial_configs, STANDARD_PROMPTS


class FactorialExperiment:
    """
    Factorial experiment runner for the 4×4×N experimental design.
    
    Runs all combinations of:
    - 4 models: Qwen2, Qwen3, TinyLlama, TinyStories
    - 4 intervention combinations: control, weighting, prompting, both
    - N prompts: configurable set of test prompts
    """
    
    def __init__(self, results_dir: str = "experiment_framework/results"):
        """
        Initialize the factorial experiment runner.
        
        Args:
            results_dir: Directory to save experiment results
        """
        self.results_dir = results_dir
        self.data_manager = ExperimentDataManager()
        self.text_evaluator = TextEvaluator()
        
        # Initialize model wrappers
        self.models = {
            "Qwen2": Qwen2Wrapper(),
            "Qwen3": Qwen3Wrapper(),
            "TinyLlama": TinyLlamaWrapper(),
            "TinyStories": TinyStoriesWrapper()
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"FactorialExperiment initialized. Results will be saved to: {self.results_dir}")
        print(f"Loaded {len(self.models)} model wrappers")
    
    def run_full_experiment(self, 
                          prompts: Optional[List[str]] = None,
                          experiment_name: str = "factorial_experiment") -> pd.DataFrame:
        """
        Run the complete factorial experiment.
        
        Args:
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            experiment_name: Name for this experiment run
            
        Returns:
            DataFrame with all results in specification format
        """
        if prompts is None:
            prompts = STANDARD_PROMPTS
        
        print(f"\n🚀 Starting factorial experiment: {experiment_name}")
        print(f"📝 Testing {len(prompts)} prompts")
        print(f"🤖 Using {len(self.models)} models")
        print(f"⚙️  Testing 4 intervention combinations per model")
        print(f"📊 Total experiments: {len(prompts)} × {len(self.models)} × 4 = {len(prompts) * len(self.models) * 4}")
        
        configs = create_factorial_configs()
        results = []
        
        total_experiments = len(prompts) * len(configs)
        current_experiment = 0
        
        start_time = time.time()
        
        for prompt_idx, prompt in enumerate(prompts):
            prompt_id = f"P{prompt_idx + 1}"
            
            print(f"\n📝 Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
            
            for config in configs:
                current_experiment += 1
                
                print(f"  🤖 [{current_experiment}/{total_experiments}] {config.experiment_name}")
                
                # Update config with prompt ID
                config.prompt_id = prompt_id
                
                # Get the appropriate model wrapper
                model_wrapper = self.models[config.model_name]
                
                # Generate response
                response_data = model_wrapper.generate_response(prompt, config)
                
                if response_data['generation_successful']:
                    # Calculate text metrics
                    text_metrics = self.text_evaluator.evaluate_text_comprehensive(
                        response_data['cleaned_response']
                    )
                    
                    # Create experiment result
                    result = ExperimentResult.create_from_response(
                        prompt=prompt,
                        response=response_data['response'],
                        config=config,
                        response_time=response_data['time_spent'],
                        text_metrics=text_metrics,
                        experiment_name=config.experiment_name,
                        cleaned_response=response_data['cleaned_response']
                    )
                    
                    print(f"    ✅ Success ({response_data['time_spent']:.2f}s)")
                    
                else:
                    # Create result for failed generation
                    empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")
                    
                    result = ExperimentResult.create_from_response(
                        prompt=prompt,
                        response="",
                        config=config,
                        response_time=response_data['time_spent'],
                        text_metrics=empty_metrics,
                        experiment_name=config.experiment_name,
                        cleaned_response=""
                    )
                    
                    print(f"    ❌ Failed: {response_data['error_message']}")
                
                results.append(result)
                self.data_manager.add_result(result)
                
                # Brief pause between experiments
                time.sleep(0.1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Factorial experiment completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Generated {len(results)} results")
        
        # Convert to DataFrame in specification format
        df = self.data_manager.to_dataframe()
        
        return df
    
    def run_single_model_experiment(self, 
                                   model_name: str,
                                   prompts: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run factorial experiment for a single model only.
        
        Args:
            model_name: Name of model to test ("Qwen2", "Qwen3", "TinyLlama", "TinyStories")
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            
        Returns:
            DataFrame with results for the specified model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Choose from: {list(self.models.keys())}")
        
        if prompts is None:
            prompts = STANDARD_PROMPTS
        
        print(f"\n🚀 Starting single model experiment: {model_name}")
        
        # Filter configs to only the specified model
        all_configs = create_factorial_configs()
        model_configs = [c for c in all_configs if c.model_name == model_name]
        
        results = []
        model_wrapper = self.models[model_name]
        
        for prompt_idx, prompt in enumerate(prompts):
            prompt_id = f"P{prompt_idx + 1}"
            
            print(f"\n📝 Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
            
            for config in model_configs:
                config.prompt_id = prompt_id
                
                print(f"  ⚙️  {config.experiment_name}")
                
                # Generate response
                response_data = model_wrapper.generate_response(prompt, config)
                
                if response_data['generation_successful']:
                    text_metrics = self.text_evaluator.evaluate_text_comprehensive(
                        response_data['cleaned_response']
                    )
                    
                    result = ExperimentResult.create_from_response(
                        prompt=prompt,
                        response=response_data['response'],
                        config=config,
                        response_time=response_data['time_spent'],
                        text_metrics=text_metrics,
                        experiment_name=config.experiment_name,
                        cleaned_response=response_data['cleaned_response']
                    )
                    
                    print(f"    ✅ Success ({response_data['time_spent']:.2f}s)")
                    
                else:
                    empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")
                    
                    result = ExperimentResult.create_from_response(
                        prompt=prompt,
                        response="",
                        config=config,
                        response_time=response_data['time_spent'],
                        text_metrics=empty_metrics,
                        experiment_name=config.experiment_name,
                        cleaned_response=""
                    )
                    
                    print(f"    ❌ Failed: {response_data['error_message']}")
                
                results.append(result)
                self.data_manager.add_result(result)
        
        print(f"\n🎉 Single model experiment completed for {model_name}!")
        print(f"📊 Generated {len(results)} results")
        
        return self.data_manager.to_dataframe()
    
    def save_results(self, filename_prefix: str) -> Dict[str, str]:
        """
        Save experiment results in multiple formats.
        
        Args:
            filename_prefix: Prefix for output files
            
        Returns:
            Dictionary with paths to saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files = {}
        
        # Save in specification format (CSV)
        spec_csv_path = os.path.join(self.results_dir, f"{filename_prefix}_specification_{timestamp}.csv")
        self.data_manager.export_to_csv_specification_format(spec_csv_path)
        files['specification_csv'] = spec_csv_path
        
        # Save full data (Parquet)
        parquet_path = os.path.join(self.results_dir, f"{filename_prefix}_full_{timestamp}.parquet")
        self.data_manager.save_to_parquet(parquet_path)
        files['full_parquet'] = parquet_path
        
        # Save full data (CSV backup)
        full_csv_path = os.path.join(self.results_dir, f"{filename_prefix}_full_{timestamp}.csv")
        self.data_manager.save_to_csv(full_csv_path)
        files['full_csv'] = full_csv_path
        
        # Save summary statistics (JSON)
        summary = self.data_manager.get_summary_stats()
        summary_path = os.path.join(self.results_dir, f"{filename_prefix}_summary_{timestamp}.json")
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        files['summary_json'] = summary_path
        
        print(f"\n💾 Results saved:")
        for file_type, path in files.items():
            print(f"  {file_type}: {path}")
        
        return files
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all model wrappers.
        
        Returns:
            Dictionary with model status information
        """
        status = {}
        
        for model_name, wrapper in self.models.items():
            model_info = wrapper.get_model_info()
            
            # Test if model is actually loaded by trying a simple generation
            test_config = ExperimentConfig(
                model_name=model_name,
                config_weighting=False,
                config_prompting=False,
                prompt_id="test"
            )
            
            test_result = wrapper.generate_response("Hello", test_config)
            
            status[model_name] = {
                'model_info': model_info,
                'loaded': test_result['generation_successful'],
                'error': test_result.get('error_message', '') if not test_result['generation_successful'] else None
            }
        
        return status
    
    def clear_results(self):
        """Clear all stored results."""
        self.data_manager.clear()
        print("🗑️  All results cleared")
