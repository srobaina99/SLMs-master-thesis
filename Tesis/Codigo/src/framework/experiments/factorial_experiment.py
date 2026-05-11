"""
Factorial experiment runner implementing the 4×4×N experimental design.
Runs all model-intervention combinations across multiple prompts.
"""

import sys
import os
import time
import gc
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.framework.core.data_models import ExperimentConfig, ExperimentResult, ExperimentDataManager
from src.framework.models import (
    BaseModelWrapper, Phi3LlamaCppWrapper, Qwen2LlamaCppWrapper, Qwen3LlamaCppWrapper, TinyLlamaLlamaCppWrapper
)
from src.text_complexity.text_evaluator import TextEvaluator
from src.framework.experiments.experiment_configs import (
    create_factorial_configs, create_multi_weight_configs, create_beam_search_configs, STANDARD_PROMPTS
)


class FactorialExperiment:
    """
    Factorial experiment runner for the factorial experimental design.
    
    Runs all combinations of:
    - 4 models: Qwen2, Qwen3, TinyLlama, Phi3
    - 4 intervention combinations: control, weighting, prompting, both
    - N prompts: configurable set of test prompts
    """
    
    def __init__(self, results_dir: str = "results"):
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
            "Phi3": Phi3LlamaCppWrapper,  # Using llama.cpp (Microsoft 3.8B reasoning model)
            "Qwen2": Qwen2LlamaCppWrapper,  # Using llama.cpp for 4x speedup
            "Qwen3": Qwen3LlamaCppWrapper,  # Using llama.cpp for 4.4x speedup
            "TinyLlama": TinyLlamaLlamaCppWrapper,  # Using llama.cpp (standardized wrapper)
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"FactorialExperiment initialized. Results will be saved to: {self.results_dir}")
    
    def run_full_experiment(self, 
                          prompts: Optional[List[str]] = None,
                          experiment_name: str = "factorial_experiment") -> pd.DataFrame:
        """
        Run the complete factorial experiment with memory-efficient model cleanup.
        
        Groups experiments by model to minimize memory usage:
        - Load model → run all prompts/configs → cleanup → next model
        
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
        print(f"💡 Memory-efficient mode: Models loaded one at a time with cleanup")
        
        # Clear previous results to avoid accumulation
        self.data_manager.clear()
        
        configs = create_factorial_configs()
        results = []
        
        # Group configs by model for efficient memory management
        configs_by_model = {}
        for config in configs:
            if config.model_name not in configs_by_model:
                configs_by_model[config.model_name] = []
            configs_by_model[config.model_name].append(config)
        
        total_experiments = len(prompts) * len(configs)
        start_time = time.time()
        
        # Create progress bar
        with tqdm(total=total_experiments, desc="🧪 Factorial Experiment", 
                  unit="exp", ncols=100, colour="green") as pbar:
            
            # Process one model at a time
            for model_idx, (model_name, model_configs) in enumerate(configs_by_model.items()):
                print(f"\n🤖 [{model_idx + 1}/{len(configs_by_model)}] Loading {model_name}...")
                
                # Load model
                model_wrapper = self._get_model(model_name)
                
                # Run all prompts × configs for this model
                for prompt_idx, prompt in enumerate(prompts):
                    prompt_id = f"P{prompt_idx + 1}"
                    
                    # Update progress bar description
                    pbar.set_description(f"🤖 {model_name} - P{prompt_idx + 1}/{len(prompts)}")
                    
                    for config in model_configs:
                        # Update progress bar with current config
                        pbar.set_postfix(model=model_name, 
                                       config=f"W:{config.config_weighting} P:{config.config_prompting}")
                        
                        # Update config with prompt ID
                        config.prompt_id = prompt_id
                        
                        # Generate response
                        response_data = model_wrapper.generate_response(prompt, config)
                        
                        if response_data['generation_successful']:
                            # Calculate text metrics using model's text_evaluator (includes tokenizer)
                            text_metrics = model_wrapper.text_evaluator.evaluate_text_comprehensive(
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
                                cleaned_response=response_data['cleaned_response'],
                                generation_successful=True
                            )

                            # Update progress bar with success
                            pbar.set_postfix(model=model_name,
                                           status="✅",
                                           time=f"{response_data['time_spent']:.1f}s")

                        else:
                            # Create result for failed/timeout generation with empty metrics
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
                                cleaned_response=response_data['cleaned_response'],
                                generation_successful=False
                            )

                            # Update progress bar with failure/timeout
                            if "timed out" in response_data['error_message'].lower():
                                pbar.set_postfix(model=model_name,
                                               status="⏰",
                                               time=f"{response_data['time_spent']:.0f}s")
                            else:
                                pbar.set_postfix(model=model_name,
                                               status="❌",
                                               error="Failed")
                        
                        results.append(result)
                        self.data_manager.add_result(result)
                        
                        # Update progress bar
                        pbar.update(1)
                
                # Cleanup model after completing all experiments for this model
                print(f"🧹 Cleaning up {model_name}...")
                model_wrapper.cleanup()
                del self._models[model_name]
                
                # Force garbage collection to free memory
                gc.collect()
                
                # Brief pause to let system stabilize
                time.sleep(1)
        
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
            model_name: Name of model to test ("Qwen2", "Qwen3", "TinyLlama", "Phi3", "SmolLM")
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            
        Returns:
            DataFrame with results for the specified model
        """
        if model_name not in self._model_classes:
            raise ValueError(f"Model '{model_name}' not available. Choose from: {list(self._model_classes.keys())}")
        
        if prompts is None:
            prompts = STANDARD_PROMPTS
        
        print(f"\n🚀 Starting single model experiment: {model_name}")
        
        # Clear previous results to avoid accumulation
        self.data_manager.clear()
        
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
                        text_metrics = model_wrapper.text_evaluator.evaluate_text_comprehensive(
                            response_data['cleaned_response']
                        )

                        result = ExperimentResult.create_from_response(
                            prompt=prompt,
                            response=response_data['response'],
                            config=config,
                            response_time=response_data['time_spent'],
                            text_metrics=text_metrics,
                            experiment_name=config.experiment_name,
                            cleaned_response=response_data['cleaned_response'],
                            generation_successful=True
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
                            cleaned_response=response_data['cleaned_response'],
                            generation_successful=False
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
    
    def run_multi_weight_experiment(self,
                                    prompts: Optional[List[str]] = None,
                                    weight_factors: List[float] = [1.5, 2.0, 4.0],
                                    experiment_name: str = "multi_weight_experiment",
                                    model_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Run experiment testing multiple weight factors.

        Tests different weighting strengths to understand the effect of weight_factor parameter.
        For each model, tests:
        - Control (no interventions)
        - Prompting only
        - Weighting with each weight factor (alone and with prompting)

        Args:
            prompts: List of prompts to test (uses STANDARD_PROMPTS if None)
            weight_factors: List of weight factors to test (default: [1.5, 2.0, 4.0])
            experiment_name: Name for this experiment run
            model_filter: If provided, only run for this model (e.g. "Qwen3")

        Returns:
            DataFrame with all results in specification format
        """
        if prompts is None:
            prompts = STANDARD_PROMPTS

        if model_filter is not None:
            if model_filter not in self._model_classes:
                raise ValueError(f"Model '{model_filter}' not available. Choose from: {list(self._model_classes.keys())}")
            active_model_classes = {model_filter: self._model_classes[model_filter]}
        else:
            active_model_classes = self._model_classes

        configs = create_multi_weight_configs(weight_factors)
        configs = [c for c in configs if c.model_name in active_model_classes]

        # Calculate total experiments: models × len(weight_factors) × prompts
        configs_per_model = len(weight_factors)  # Only weighted configs
        total_configs = len(active_model_classes) * configs_per_model

        print(f"\n🚀 Starting multi-weight experiment: {experiment_name}")
        print(f"📝 Testing {len(prompts)} prompts")
        print(f"🤖 Using {len(active_model_classes)} models")
        print(f"⚖️  Testing {len(weight_factors)} weight factors: {weight_factors}")
        print(f"⚙️  {configs_per_model} configurations per model (weighted only)")
        print(f"📊 Total experiments: {len(prompts)} × {total_configs} = {len(prompts) * total_configs}")
        
        results = []
        total_experiments = len(prompts) * len(configs)
        start_time = time.time()
        
        # Create progress bar
        with tqdm(total=total_experiments, desc="🧪 Multi-Weight Experiment", 
                  unit="exp", ncols=100, colour="magenta") as pbar:
            
            for prompt_idx, prompt in enumerate(prompts):
                prompt_id = f"P{prompt_idx + 1}"
                
                # Update progress bar description with current prompt
                pbar.set_description(f"📝 P{prompt_idx + 1}/{len(prompts)}: {prompt[:30]}...")
                
                for config in configs:
                    # Update progress bar with current model and weight
                    weight_info = f"w={config.weight_factor}" if config.config_weighting else "w=off"
                    pbar.set_postfix(model=config.model_name, 
                                   weight=weight_info,
                                   prompt=f"P:{config.config_prompting}")
                    
                    # Update config with prompt ID
                    config.prompt_id = prompt_id
                    
                    # Get the appropriate model wrapper (lazy initialization)
                    model_wrapper = self._get_model(config.model_name)
                    
                    # Generate response
                    response_data = model_wrapper.generate_response(prompt, config)
                    
                    if response_data['generation_successful']:
                        # Calculate text metrics using model's text_evaluator (includes tokenizer)
                        text_metrics = model_wrapper.text_evaluator.evaluate_text_comprehensive(
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
                            cleaned_response=response_data['cleaned_response'],
                            generation_successful=True
                        )

                        # Update progress bar with success
                        pbar.set_postfix(model=config.model_name,
                                       weight=weight_info,
                                       status="✅",
                                       time=f"{response_data['time_spent']:.1f}s")

                    else:
                        # Create result for failed/timeout generation with empty metrics
                        empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")

                        response_value = response_data['response'] if response_data['response'] is not None else None

                        result = ExperimentResult.create_from_response(
                            prompt=prompt,
                            response=response_value,
                            config=config,
                            response_time=response_data['time_spent'],
                            text_metrics=empty_metrics,
                            experiment_name=config.experiment_name,
                            cleaned_response=response_data['cleaned_response'],
                            generation_successful=False
                        )

                        # Update progress bar with failure/timeout
                        if "timed out" in response_data['error_message'].lower():
                            pbar.set_postfix(model=config.model_name,
                                           weight=weight_info,
                                           status="⏰",
                                           time=f"{response_data['time_spent']:.0f}s")
                        else:
                            pbar.set_postfix(model=config.model_name,
                                           weight=weight_info,
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
        
        print(f"\n🎉 Multi-weight experiment completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Generated {len(results)} results")
        print(f"⚖️  Weight factors tested: {weight_factors}")
        
        # Convert to DataFrame in specification format
        df = self.data_manager.to_dataframe()
        
        return df
    
    def run_beam_search_experiment(self,
                                  prompts: Optional[List[str]] = None,
                                  beam_width: int = 4,
                                  experiment_name: str = "beam_search_experiment") -> pd.DataFrame:
        """
        Run beam search experiment testing two selection criteria: A1 word ratio and max probability.
        
        Uses first N prompts from STANDARD_PROMPTS and Qwen3 model with contextual prompting.
        Compares beam selection methods by generating readability metrics for each selected beam.
        
        Args:
            prompts: List of prompts to test (uses first 5 STANDARD_PROMPTS if None)
            beam_width: Number of beams to maintain (default: 4)
            experiment_name: Name for this experiment run
            
        Returns:
            DataFrame with all results in specification format
        """
        # Use first 5 prompts if not specified
        if prompts is None:
            prompts = STANDARD_PROMPTS[:5]
        else:
            prompts = prompts[:5]  # Limit to 5 prompts
        
        print(f"\n🚀 Starting beam search experiment: {experiment_name}")
        print(f"📝 Testing {len(prompts)} prompts")
        print(f"🤖 Model: Qwen3")
        print(f"🔦 Beam width: {beam_width}")
        print(f"⚙️  Selection methods: A1 word ratio, Max cumulative log probability")
        print(f"📊 Total experiments: {len(prompts)} × 2 selection methods = {len(prompts) * 2}")
        
        # Clear previous results
        self.data_manager.clear()
        
        results = []
        total_experiments = len(prompts) * 2  # 2 selection methods
        start_time = time.time()
        
        # Load Qwen3 model
        print(f"\n🤖 Loading Qwen3 model...")
        qwen3_wrapper = self._get_model("Qwen3")
        
        # Create beam search configs
        configs = create_beam_search_configs(beam_width=beam_width, use_prompting=True)
        
        # Create progress bar
        with tqdm(total=total_experiments, desc="🧪 Beam Search Experiment", 
                  unit="exp", ncols=100, colour="cyan") as pbar:
            
            for prompt_idx, prompt in enumerate(prompts):
                prompt_id = f"P{prompt_idx + 1}"
                
                # Update progress bar description
                pbar.set_description(f"📝 P{prompt_idx + 1}/{len(prompts)}: {prompt[:30]}...")
                
                # Test both selection methods
                for config in configs:
                    config.prompt_id = prompt_id
                    
                    # Determine selection method from config name
                    selection_method = "a1_ratio" if "a1_ratio" in config.experiment_name else "max_probability"
                    
                    # Update progress bar
                    pbar.set_postfix(method=selection_method)
                    
                    # Generate response with beam search
                    beam_response_data = qwen3_wrapper.generate_with_beam_search(
                        prompt=prompt,
                        config=config,
                        beam_width=beam_width,
                        selection_method=selection_method
                    )
                    
                    if beam_response_data['generation_successful']:
                        # Calculate text metrics
                        text_metrics = qwen3_wrapper.text_evaluator.evaluate_text_comprehensive(
                            beam_response_data['response']
                        )

                        # Create experiment result with beam-specific fields
                        result = ExperimentResult.create_from_beam_response(
                            prompt=prompt,
                            response=beam_response_data['response'],
                            config=config,
                            response_time=beam_response_data['time_spent'],
                            text_metrics=text_metrics,
                            experiment_name=config.experiment_name,
                            cleaned_response=beam_response_data['response'],
                            beam_selection_method=beam_response_data['beam_selection_method'],
                            beam_a1_ratio=beam_response_data['beam_a1_ratio'],
                            beam_a1_count=beam_response_data['beam_a1_count'],
                            beam_content_word_count=beam_response_data['beam_content_word_count'],
                            beam_cumulative_logprob=beam_response_data['beam_cumulative_logprob'],
                            beam_width=beam_width
                        )
                        
                        # Update progress bar with success
                        pbar.set_postfix(method=selection_method,
                                       status="✅",
                                       time=f"{beam_response_data['time_spent']:.1f}s")
                        
                    else:
                        # Create result for failed generation
                        empty_metrics = self.text_evaluator.evaluate_text_comprehensive("")

                        result = ExperimentResult.create_from_beam_response(
                            prompt=prompt,
                            response="",
                            config=config,
                            response_time=beam_response_data['time_spent'],
                            text_metrics=empty_metrics,
                            experiment_name=config.experiment_name,
                            cleaned_response="",
                            generation_successful=False,
                            beam_selection_method=selection_method,
                            beam_a1_ratio=0.0,
                            beam_a1_count=0,
                            beam_content_word_count=0,
                            beam_cumulative_logprob=0.0,
                            beam_width=beam_width
                        )
                        
                        # Update progress bar with failure
                        pbar.set_postfix(method=selection_method,
                                       status="❌",
                                       error=beam_response_data['error_message'][:20])
                    
                    results.append(result)
                    self.data_manager.add_result(result)
                    pbar.update(1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Beam search experiment completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Generated {len(results)} results")
        print(f"🔦 Selection methods: A1 word ratio, Max cumulative log probability")
        
        # Convert to DataFrame
        df = self.data_manager.to_dataframe()
        
        return df
    
    def clear_results(self):
        """Clear all stored results."""
        self.data_manager.clear()
        print("🗑️  All results cleared")
