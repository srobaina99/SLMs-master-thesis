"""
TinyStories model wrapper implementing BaseModelWrapper interface.
Integrates with existing TinyStories implementation while providing standardized interface.
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import LogitsProcessorList
except ImportError as e:
    print(f"Warning: Could not import transformers: {e}")
    torch = None


class TinyStoriesWrapper(BaseModelWrapper):
    """
    TinyStories model wrapper implementing the standardized interface.
    Supports both probability weighting and context prompting interventions.
    """
    
    def __init__(self):
        """Initialize TinyStories wrapper."""
        super().__init__("TinyStories")
        self.text_evaluator = TextEvaluator()
        self.response_formatter = ResponseFormatter()
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the TinyStories model."""
        if torch is None:
            print("Warning: PyTorch not available, TinyStories model cannot be loaded")
            return
            
        try:
            # Enable MPS fallback for operations not supported on MPS
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            print("Loading TinyStories model...")
            
            # Load TinyStories model and compatible tokenizer
            self.model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("TinyStories model loaded successfully")
            
        except Exception as e:
            print(f"Error loading TinyStories model: {e}")
            self.model = None
            self.tokenizer = None
    
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
            raise RuntimeError("TinyStories model not loaded")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        
        # Move to appropriate device
        device = "cpu"  # TinyStories works best on CPU
        if torch.backends.mps.is_available():
            try:
                input_ids = input_ids.to("mps")
                attention_mask = attention_mask.to("mps")
                self.model = self.model.to("mps")
                device = "mps"
            except:
                # Fallback to CPU if MPS fails
                input_ids = input_ids.to("cpu")
                attention_mask = attention_mask.to("cpu")
                self.model = self.model.to("cpu")
        
        # Create logits processor for weighting
        logits_processor = self._create_logits_processor(
            self.target_vocabulary, 
            config.weight_factor
        )
        
        # Generate with custom logits processor
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=min(len(input_ids[0]) + config.max_new_tokens, 2048),  # Use config max_new_tokens
                do_sample=True,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove input prompt)
        if response.startswith(prompt):
            generated_text = response[len(prompt):]
        else:
            generated_text = response
        
        return generated_text.strip()
    
    def _generate_simple(self, prompt: str, config: ExperimentConfig) -> str:
        """Generate response without weighting."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("TinyStories model not loaded")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        
        # Move to appropriate device (CPU works best for TinyStories)
        device = "cpu"
        if torch.backends.mps.is_available():
            try:
                input_ids = input_ids.to("mps")
                attention_mask = attention_mask.to("mps")
                self.model = self.model.to("mps")
                device = "mps"
            except:
                input_ids = input_ids.to("cpu")
                attention_mask = attention_mask.to("cpu")
                self.model = self.model.to("cpu")
        
        # Generate without custom logits processor
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=min(len(input_ids[0]) + config.max_new_tokens, 2048),  # Use config max_new_tokens
                do_sample=True,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove input prompt)
        if response.startswith(prompt):
            generated_text = response[len(prompt):]
        else:
            generated_text = response
        
        return generated_text.strip()
    
    def _generate_response_impl(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate response using TinyStories with the given configuration.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing response, timing, and success information
        """
        if not self.model or not self.tokenizer:
            return {
                'response': '',
                'time_spent': 0.0,
                'generation_successful': False,
                'error_message': 'TinyStories model not loaded',
                'cleaned_response': ''
            }
        
        start_time = time.time()
        
        try:
            # Apply context prompting intervention if enabled
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(prompt)
            
            # TinyStories works best with simple prompts, no special formatting needed
            # Just add system context if provided
            if config.system_prompt and config.system_prompt != "You are a helpful AI assistant.":
                formatted_prompt = f"{config.system_prompt}\n\n{final_prompt}"
            else:
                formatted_prompt = final_prompt
            
            # Generate response based on weighting configuration
            if config.config_weighting:
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
