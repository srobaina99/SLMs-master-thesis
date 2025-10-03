"""
Qwen3 model wrapper implementing BaseModelWrapper interface.
Integrates with existing Qwen3 implementation while providing standardized interface.
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.append(project_root)

from .base_model import BaseModelWrapper
from src.evaluation.experiment_framework.core.data_models import ExperimentConfig
from src.evaluation.text_complexity.text_evaluator import TextEvaluator
from src.evaluation.text_complexity.response_formatter import ResponseFormatter

try:
    from src.models.qwen.qwen3.qwen3_weighted import generate_response, model, tokenizer
except ImportError as e:
    print(f"Warning: Could not import Qwen3 modules: {e}")
    generate_response = None
    model = None
    tokenizer = None


class Qwen3Wrapper(BaseModelWrapper):
    """
    Qwen3 model wrapper implementing the standardized interface.
    Supports both probability weighting and context prompting interventions.
    """
    
    def __init__(self):
        """Initialize Qwen3 wrapper."""
        super().__init__("Qwen3")
        self.text_evaluator = TextEvaluator()
        self.response_formatter = ResponseFormatter()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen3 model."""
        self.model_loaded = self._check_model_availability()
        if self.model_loaded:
            print(f"Qwen3 model loaded successfully")
        else:
            print(f"Warning: Qwen3 model not available")
    
    def _check_model_availability(self) -> bool:
        """Check if Qwen3 model and dependencies are available."""
        try:
            if model is not None and tokenizer is not None and generate_response is not None:
                return True
            return False
        except:
            return False
    
    def _generate_response_impl(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate response using Qwen3 with the given configuration.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing response, timing, and success information
        """
        if not self.model_loaded:
            return {
                'response': '',
                'time_spent': 0.0,
                'generation_successful': False,
                'error_message': 'Qwen3 model not loaded',
                'cleaned_response': ''
            }
        
        start_time = time.time()
        
        try:
            # Apply context prompting intervention if enabled
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(prompt)
            
            # Apply probability weighting intervention if enabled
            weighted_words = None
            if config.config_weighting:
                weighted_words = self.target_vocabulary
            
            # Generate response using existing Qwen3 function
            response = generate_response(
                user_input=final_prompt,
                system_prompt=config.system_prompt,
                enable_thinking=config.enable_thinking,
                weighted_words=weighted_words or [],
                weight_factor=config.weight_factor,
                verbose=config.verbose
            )
            
            end_time = time.time()
            time_spent = end_time - start_time
            
            # Clean response for evaluation
            cleaned_response = self.response_formatter.clean_response_for_evaluation(response)
            
            return {
                'response': response,
                'time_spent': time_spent,
                'generation_successful': True,
                'error_message': '',
                'cleaned_response': cleaned_response
            }
            
        except Exception as e:
            end_time = time.time()
            time_spent = end_time - start_time
            
            return {
                'response': '',
                'time_spent': time_spent,
                'generation_successful': False,
                'error_message': str(e),
                'cleaned_response': ''
            }
    
    def generate_response_with_metrics(self, 
                                     user_input: str,
                                     system_prompt: str = "You are a helpful AI assistant.",
                                     enable_thinking: bool = False,
                                     weighted_words: Optional[List[str]] = None,
                                     weight_factor: float = 1.0,
                                     verbose: bool = False) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility with existing experiment framework.
        """
        # Create a temporary config for the legacy interface
        config = ExperimentConfig(
            model_name="Qwen3",
            system_prompt=system_prompt,
            config_weighting=weighted_words is not None and len(weighted_words) > 0,
            config_prompting=False,  # Not supported in legacy interface
            enable_thinking=enable_thinking,
            weight_factor=weight_factor,
            verbose=verbose
        )
        
        result = self.generate_response(user_input, config)
        
        # Add text metrics for legacy compatibility
        if result['generation_successful']:
            text_metrics = self.text_evaluator.evaluate_text_comprehensive(
                result['cleaned_response']
            )
            result['text_metrics'] = text_metrics
            result['response_time_seconds'] = result['time_spent']
        else:
            result['text_metrics'] = self.text_evaluator.evaluate_text_comprehensive("")
            result['response_time_seconds'] = result['time_spent']
        
        return result
