"""
Qwen2 model wrapper implementing BaseModelWrapper interface.
Integrates with existing Qwen2 implementation while providing standardized interface.
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from transformers.generation import LogitsProcessorList
except ImportError as e:
    print(f"Warning: Could not import transformers: {e}")
    torch = None


class Qwen2Wrapper(BaseModelWrapper):
    """
    Qwen2 model wrapper implementing the standardized interface.
    Supports both probability weighting and context prompting interventions.
    """
    
    def __init__(self):
        """Initialize Qwen2 wrapper."""
        super().__init__("Qwen2")
        self.text_evaluator = TextEvaluator()
        self.response_formatter = ResponseFormatter()
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen2 model."""
        if torch is None:
            print("Warning: PyTorch not available, Qwen2 model cannot be loaded")
            return
            
        try:
            # Enable MPS fallback for operations not supported on MPS
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Set device
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Initializing Qwen2 on device: {device}")
            
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            
            # Load model and tokenizer for weighting support
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Also create pipeline for simple generation
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print("Qwen2 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Qwen2 model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipe = None
    
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
            raise RuntimeError("Qwen2 model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
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
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove input prompt)
        generated_text = response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        
        return generated_text.strip()
    
    def _generate_simple(self, prompt: str, config: ExperimentConfig) -> str:
        """Generate response without weighting using pipeline."""
        if not self.pipe:
            raise RuntimeError("Qwen2 pipeline not loaded")
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=1024,
            do_sample=True,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=1.05
        )
        
        response = outputs[0]["generated_text"]
        
        # Extract just the assistant's response
        # Qwen2.5 uses ChatML format
        if "<|im_start|>assistant\n" in response:
            assistant_response = response.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|>")[0]
            return assistant_response.strip()
        
        # Fallback: remove the input prompt
        if len(response) > len(prompt):
            return response[len(prompt):].strip()
        
        return response.strip()
    
    def generate_response(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate response using Qwen2 with the given configuration.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing response, timing, and success information
        """
        if not self.model and not self.pipe:
            return {
                'response': '',
                'time_spent': 0.0,
                'generation_successful': False,
                'error_message': 'Qwen2 model not loaded',
                'cleaned_response': ''
            }
        
        start_time = time.time()
        
        try:
            # Apply context prompting intervention if enabled
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(prompt)
            
            # Format prompt using ChatML format
            formatted_prompt = f"<|im_start|>system\n{config.system_prompt}<|im_end|>\n<|im_start|>user\n{final_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate response based on weighting configuration
            if config.config_weighting and self.model:
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
