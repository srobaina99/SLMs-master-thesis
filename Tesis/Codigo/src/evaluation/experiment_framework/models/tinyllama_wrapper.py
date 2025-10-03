"""
TinyLlama model wrapper implementing BaseModelWrapper interface.
Integrates with existing TinyLlama implementation while providing standardized interface.
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
from src.models.probability_processor import ProbabilityWeightingLogitsProcessor

try:
    from llama_cpp import Llama
except ImportError as e:
    print(f"Warning: Could not import llama-cpp-python: {e}")
    Llama = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import LogitsProcessorList
except ImportError as e:
    print(f"Warning: Could not import transformers: {e}")
    torch = None


class TinyLlamaWrapper(BaseModelWrapper):
    """
    TinyLlama model wrapper implementing the standardized interface.
    Supports both probability weighting and context prompting interventions.
    """
    
    def __init__(self):
        """Initialize TinyLlama wrapper."""
        super().__init__("TinyLlama")
        self.text_evaluator = TextEvaluator()
        self.response_formatter = ResponseFormatter()
        self.model = None
        self.tokenizer = None
        self.llama_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the TinyLlama model."""
        # Try to load with transformers first (for weighting support)
        if torch is not None:
            try:
                print("Loading TinyLlama with transformers...")
                model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    device_map="auto"
                )
                
                # Add pad token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print("TinyLlama loaded successfully with transformers")
                return
                
            except Exception as e:
                print(f"Failed to load TinyLlama with transformers: {e}")
        
        # Fallback to llama-cpp if available
        if Llama is not None:
            try:
                print("Loading TinyLlama with llama-cpp...")
                # This would need the actual GGUF model file path
                # For now, we'll just indicate it's not available
                print("Warning: llama-cpp fallback not configured - need GGUF model file")
                
            except Exception as e:
                print(f"Failed to load TinyLlama with llama-cpp: {e}")
        
        print("Warning: TinyLlama model not loaded")
    
    def _create_logits_processor(self, target_words: List[str], weight_factor: float) -> LogitsProcessorList:
        """Create logits processor for probability weighting."""
        if not target_words or not self.tokenizer:
            return LogitsProcessorList()
        
        processor = ProbabilityWeightingLogitsProcessor(
            tokenizer=self.tokenizer,
            target_words=target_words,
            weight_factor=weight_factor
        )
        return LogitsProcessorList([processor])
    
    def _generate_with_weighting(self, prompt: str, config: ExperimentConfig) -> str:
        """Generate response with probability weighting."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("TinyLlama model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Create logits processor for weighting
        logits_processor = self._create_logits_processor(
            self.target_vocabulary, 
            config.weight_factor
        )
        
        # Generate with custom logits processor
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove input prompt)
        input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if response.startswith(input_text):
            generated_text = response[len(input_text):]
        else:
            generated_text = response
        
        return generated_text.strip()
    
    def _generate_simple(self, prompt: str, config: ExperimentConfig) -> str:
        """Generate response without weighting."""
        if self.llama_model:
            # Use llama-cpp if available
            response = self.llama_model(
                prompt,
                max_tokens=1024,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                stop=["</s>", "<|endoftext|>"],
                echo=False
            )
            return response['choices'][0]['text'].strip()
        
        elif self.model and self.tokenizer:
            # Use transformers without weighting
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            elif torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            
            if response.startswith(input_text):
                generated_text = response[len(input_text):]
            else:
                generated_text = response
            
            return generated_text.strip()
        
        else:
            raise RuntimeError("No TinyLlama model available")
    
    def _generate_response_impl(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate response using TinyLlama with the given configuration.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing response, timing, and success information
        """
        if not self.model and not self.llama_model:
            return {
                'response': '',
                'time_spent': 0.0,
                'generation_successful': False,
                'error_message': 'TinyLlama model not loaded',
                'cleaned_response': ''
            }
        
        start_time = time.time()
        
        try:
            # Apply context prompting intervention if enabled
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(prompt)
            
            # Format prompt for TinyLlama (using ChatML-like format)
            formatted_prompt = f"<|system|>\n{config.system_prompt}</s>\n<|user|>\n{final_prompt}</s>\n<|assistant|>\n"
            
            # Generate response based on weighting configuration
            if config.config_weighting and self.model and self.tokenizer:
                response = self._generate_with_weighting(formatted_prompt, config)
            else:
                response = self._generate_simple(formatted_prompt, config)
            
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
