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
from tqdm import tqdm

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.data_models import ExperimentConfig, ExperimentResult, ExperimentDataManager
from src.evaluation.experiment_framework.models import (
    BaseModelWrapper, Qwen2Wrapper, Qwen3Wrapper, TinyLlamaWrapper, TinyStoriesWrapper
)
from src.evaluation.text_complexity.text_evaluator import TextEvaluator
from src.evaluation.experiment_framework.experiments.experiment_configs import create_factorial_configs, STANDARD_PROMPTS


class FactorialExperiment:
    """
    Factorial experiment runner for the 4×4×N experimental design.
    
    Runs all combinations of:
    - 4 models: Qwen2, Qwen3, TinyLlama, TinyStories
    - 4 intervention combinations: control, weighting, prompting, both
    - N prompts: configurable set of test prompts
    """
    
    def __init__(self, results_dir: str = "src/evaluation/experiment_framework/results"):
        """
        Initialize the factorial experiment runner.
        
        Args:
            results_dir: Directory to save experiment results
        """
        self.results_dir = results_dir
        self.data_manager = ExperimentDataManager()
        self.text_evaluator = TextEvaluator()
        
        # Model wrappers - lazily initialized
        self._models = {}
        self._model_classes = {
            "Qwen2": Qwen2Wrapper,
            "Qwen3": Qwen3Wrapper,
            "TinyLlama": TinyLlamaWrapper,
            "TinyStories": TinyStoriesWrapper
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"FactorialExperiment initialized. Results will be saved to: {self.results_dir}")
    
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
        print(f"🤖 Using {len(self._model_classes)} models")
        print(f"⚙️  Testing 4 intervention combinations per model")
        print(f"📊 Total experiments: {len(prompts)} × {len(self._model_classes)} × 4 = {len(prompts) * len(self._model_classes) * 4}")
        
        configs = create_factorial_configs()
        results = []
        
        total_experiments = len(prompts) * len(configs)
        start_time = time.time()
        
        # Create progress bar
        with tqdm(total=total_experiments, desc="🧪 Factorial Experiment", 
                  unit="exp", ncols=100, colour="green") as pbar:
            
            for prompt_idx, prompt in enumerate(prompts):
                prompt_id = f"P{prompt_idx + 1}"
                
                # Update progress bar description with current prompt
                pbar.set_description(f"📝 P{prompt_idx + 1}/{len(prompts)}: {prompt[:30]}...")
                
                for config in configs:
                    # Update progress bar with current model
                    pbar.set_postfix(model=config.model_name, 
                                   config=f"W:{config.config_weighting} P:{config.config_prompting}")
                    
                    # Update config with prompt ID
                    config.prompt_id = prompt_id
                    
                    # Get the appropriate model wrapper (lazy initialization)
                    model_wrapper = self._get_model(config.model_name)
                    
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
                        
                        # Update progress bar with success
                        pbar.set_postfix(model=config.model_name, 
                                       status="✅", 
                                       time=f"{response_data['time_spent']:.1f}s")
                        
                    else:
                        # Create result for failed/timeout generation
                        empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")
                        
                        # Use None for response if it was a timeout or failure
                        response_value = response_data['response'] if response_data['response'] is not None else None
                        
                        result = ExperimentResult.create_from_response(
                            prompt=prompt,
                            response=response_value,
                            config=config,
                            response_time=response_data['time_spent'],
                            text_metrics=empty_metrics,
                            experiment_name=config.experiment_name,
                            cleaned_response=response_data['cleaned_response']
                        )
                        
                        # Update progress bar with failure/timeout
                        if "timed out" in response_data['error_message'].lower():
                            pbar.set_postfix(model=config.model_name, 
                                           status="⏰", 
                                           time=f"{response_data['time_spent']:.0f}s")
                        else:
                            pbar.set_postfix(model=config.model_name, 
                                           status="❌", 
                                           error="Failed")
                    
                    results.append(result)
                    self.data_manager.add_result(result)
                    
                    # Update progress bar
                    pbar.update(1)
                    
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
        if model_name not in self._model_classes:
            raise ValueError(f"Model '{model_name}' not available. Choose from: {list(self._model_classes.keys())}")
        
        if prompts is None:
            prompts = STANDARD_PROMPTS
        
        print(f"\n🚀 Starting single model experiment: {model_name}")
        
        # Filter configs to only the specified model
        all_configs = create_factorial_configs()
        model_configs = [c for c in all_configs if c.model_name == model_name]
        
        results = []
        model_wrapper = self._get_model(model_name)
        
        total_experiments = len(prompts) * len(model_configs)
        
        # Create progress bar for single model experiment
        with tqdm(total=total_experiments, desc=f"🤖 {model_name} Experiment", 
                  unit="exp", ncols=100, colour="blue") as pbar:
            
            for prompt_idx, prompt in enumerate(prompts):
                prompt_id = f"P{prompt_idx + 1}"
                
                # Update progress bar description with current prompt
                pbar.set_description(f"🤖 {model_name} - P{prompt_idx + 1}/{len(prompts)}: {prompt[:25]}...")
                
                for config in model_configs:
                    config.prompt_id = prompt_id
                    
                    # Update progress bar with current config
                    config_short = f"W:{config.config_weighting} P:{config.config_prompting}"
                    pbar.set_postfix(config=config_short)
                    
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
                        
                        # Update progress bar with success
                        pbar.set_postfix(config=config_short, 
                                       status="✅", 
                                       time=f"{response_data['time_spent']:.1f}s")
                        
                    else:
                        empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")
                        
                        # Use None for response if it was a timeout or failure
                        response_value = response_data['response'] if response_data['response'] is not None else None
                        
                        result = ExperimentResult.create_from_response(
                            prompt=prompt,
                            response=response_value,
                            config=config,
                            response_time=response_data['time_spent'],
                            text_metrics=empty_metrics,
                            experiment_name=config.experiment_name,
                            cleaned_response=response_data['cleaned_response']
                        )
                        
                        # Update progress bar with failure/timeout
                        if "timed out" in response_data['error_message'].lower():
                            pbar.set_postfix(config=config_short, 
                                           status="⏰", 
                                           time=f"{response_data['time_spent']:.0f}s")
                        else:
                            pbar.set_postfix(config=config_short, 
                                           status="❌", 
                                           error="Failed")
                    
                    results.append(result)
                    self.data_manager.add_result(result)
                    
                    # Update progress bar
                    pbar.update(1)
        
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
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        files = {}
        
        # Extract model name from filename_prefix for folder organization
        # Expected format: "ModelName_factorial" or similar
        model_name = filename_prefix.split('_')[0] if '_' in filename_prefix else filename_prefix
        
        # Create model-specific directory
        model_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create full_data subdirectory within model directory
        full_data_dir = os.path.join(model_dir, "full_data")
        os.makedirs(full_data_dir, exist_ok=True)
        
        # Save in specification format (CSV) - model directory
        spec_csv_path = os.path.join(model_dir, f"{filename_prefix}_specification_{timestamp}.csv")
        self.data_manager.export_to_csv_specification_format(spec_csv_path)
        files['specification_csv'] = spec_csv_path
        
        # Save full data (CSV backup) - full_data subdirectory
        full_csv_path = os.path.join(full_data_dir, f"{filename_prefix}_full_{timestamp}.csv")
        self.data_manager.save_to_csv(full_csv_path)
        files['full_csv'] = full_csv_path
        
        # Save summary statistics (JSON) - model directory
        summary = self.data_manager.get_summary_stats()
        summary_path = os.path.join(model_dir, f"{filename_prefix}_summary_{timestamp}.json")
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        files['summary_json'] = summary_path
        
        print(f"\n💾 Results saved:")
        for file_type, path in files.items():
            print(f"  {file_type}: {path}")
        
        return files
    
    def _get_model(self, model_name: str) -> BaseModelWrapper:
        """
        Get or lazily initialize a model wrapper.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            The model wrapper instance
        """
        if model_name not in self._models:
            print(f"Loading {model_name} model...")
            self._models[model_name] = self._model_classes[model_name]()
        return self._models[model_name]
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all model wrappers.
        
        Returns:
            Dictionary with model status information
        """
        status = {}
        
        for model_name in self._model_classes.keys():
            wrapper = self._get_model(model_name)
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
