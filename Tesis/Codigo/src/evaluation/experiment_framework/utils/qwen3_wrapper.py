"""
Wrapper for Qwen3 model integration with experiment framework.
Provides a clean interface to the existing Qwen3 implementation with metrics collection.
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional, Tuple

# Add parent directories to path to import existing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(project_root)

try:
    from src.models.qwen.qwen3.qwen3_weighted import generate_response, model, tokenizer
    from src.evaluation.text_complexity.text_evaluator import TextEvaluator
    from src.evaluation.text_complexity.response_formatter import ResponseFormatter
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure you're running from the correct directory and dependencies are installed.")


class Qwen3ExperimentWrapper:
    """
    Wrapper class that integrates Qwen3 model with experiment framework.
    Handles response generation, timing, and automatic text evaluation.
    """
    
    def __init__(self):
        """Initialize the wrapper with text evaluator and response formatter."""
        self.text_evaluator = TextEvaluator()
        self.response_formatter = ResponseFormatter()
        self.model_loaded = self._check_model_availability()
    
    def _check_model_availability(self) -> bool:
        """Check if Qwen3 model and dependencies are available."""
        try:
            # Try to access the model and tokenizer
            _ = model
            _ = tokenizer
            return True
        except NameError:
            print("Warning: Qwen3 model not loaded. Make sure to run qwen3_weighted.py first.")
            return False
    
    def generate_response_with_metrics(self, 
                                     user_input: str,
                                     system_prompt: str = "You are a helpful AI assistant.",
                                     enable_thinking: bool = False,
                                     weighted_words: Optional[List[str]] = None,
                                     weight_factor: float = 1.0,
                                     verbose: bool = False,
                                     include_spanish_metrics: bool = False) -> Dict[str, Any]:
        """
        Generate response using Qwen3 and collect comprehensive metrics.
        
        Args:
            user_input: The prompt/question to send to the model
            system_prompt: System prompt to set model behavior
            enable_thinking: Whether to enable Qwen3's thinking mode
            weighted_words: List of words to weight in generation (if any)
            weight_factor: Factor for word weighting (>1 increases probability)
            verbose: Whether to show detailed generation info
            include_spanish_metrics: Whether to include Spanish readability metrics
            
        Returns:
            Dictionary containing response, timing, and text metrics
        """
        
        if not self.model_loaded:
            raise RuntimeError("Qwen3 model not available. Please load the model first.")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Generate response using existing Qwen3 function
            response = generate_response(
                user_input=user_input,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                weighted_words=weighted_words or [],
                weight_factor=weight_factor,
                verbose=verbose
            )
            
            # Record end time
            end_time = time.time()
            response_time = end_time - start_time
            
            # Clean response before evaluation to remove formatting interference
            cleaned_response = self.response_formatter.clean_response_for_evaluation(response)
            
            # Evaluate text complexity and readability on cleaned text
            text_metrics = self.text_evaluator.evaluate_text_comprehensive(
                text=cleaned_response,
                include_spanish=include_spanish_metrics
            )
            
            # Compile all results
            result = {
                'response': response,
                'cleaned_response': cleaned_response,
                'response_time_seconds': response_time,
                'text_metrics': text_metrics,
                'generation_successful': True,
                'error_message': None
            }
            
            return result
            
        except Exception as e:
            # Handle any errors during generation
            end_time = time.time()
            response_time = end_time - start_time
            
            error_result = {
                'response': "",
                'response_time_seconds': response_time,
                'text_metrics': self.text_evaluator.evaluate_text_comprehensive(""),
                'generation_successful': False,
                'error_message': str(e)
            }
            
            print(f"Error during response generation: {e}")
            return error_result
    
    def batch_generate_responses(self,
                               prompts: List[str],
                               system_prompt: str = "You are a helpful AI assistant.",
                               **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts with the same configuration.
        
        Args:
            prompts: List of prompts to process
            system_prompt: System prompt for all generations
            **generation_kwargs: Additional arguments for generate_response_with_metrics
            
        Returns:
            List of result dictionaries, one for each prompt
        """
        
        results = []
        total_prompts = len(prompts)
        
        print(f"Processing {total_prompts} prompts...")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Processing prompt {i}/{total_prompts}: {prompt[:50]}...")
            
            result = self.generate_response_with_metrics(
                user_input=prompt,
                system_prompt=system_prompt,
                **generation_kwargs
            )
            
            # Add prompt information to result
            result['prompt'] = prompt
            result['prompt_index'] = i - 1
            
            results.append(result)
            
            # Brief pause between generations to prevent overheating
            if i < total_prompts:
                time.sleep(0.5)
        
        print(f"Completed processing {total_prompts} prompts.")
        return results
    
    def test_single_prompt(self, 
                          prompt: str = "What does the word 'library' mean?",
                          system_prompt: str = "You are an English teacher for beginner students.") -> Dict[str, Any]:
        """
        Test the wrapper with a single prompt. Useful for debugging.
        
        Args:
            prompt: Test prompt
            system_prompt: System prompt for the test
            
        Returns:
            Result dictionary with response and metrics
        """
        
        print(f"Testing with prompt: '{prompt}'")
        print(f"System prompt: '{system_prompt}'")
        
        result = self.generate_response_with_metrics(
            user_input=prompt,
            system_prompt=system_prompt,
            verbose=True
        )
        
        if result['generation_successful']:
            print(f"\nResponse: {result['response']}")
            print(f"Response time: {result['response_time_seconds']:.2f} seconds")
            
            # Show key metrics
            metrics = result['text_metrics']
            stats = metrics.get('text_statistics', {})
            grades = metrics.get('grade_level_indices', {})
            
            print(f"\nKey Metrics:")
            print(f"  Word count: {stats.get('word_count', 0)}")
            print(f"  Sentence count: {stats.get('sentence_count', 0)}")
            print(f"  Flesch-Kincaid Grade: {grades.get('flesch_kincaid_grade', 0)}")
            print(f"  Reading ease: {metrics.get('readability_scores', {}).get('flesch_reading_ease', 0)}")
        else:
            print(f"Generation failed: {result['error_message']}")
        
        return result


# Convenience function for quick testing
def quick_test():
    """Quick test of the wrapper functionality."""
    wrapper = Qwen3ExperimentWrapper()
    
    if not wrapper.model_loaded:
        print("Cannot run test - Qwen3 model not loaded.")
        print("Please run the following first:")
        print("1. Navigate to the project root directory")
        print("2. Run: python -c \"from qwen3_mps.qwen3_weighted import *\"")
        return None
    
    return wrapper.test_single_prompt()


if __name__ == "__main__":
    # Run quick test when script is executed directly
    quick_test()
