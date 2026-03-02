"""
Base class for llama.cpp model wrappers.

This provides a reusable foundation for all GGUF-based models, handling:
- Model loading and initialization
- Prompt formatting (ChatML, Llama2, etc.)
- Probability weighting via logit_bias
- Context prompting
- Response extraction and cleaning

All future llama.cpp models should extend this class.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from abc import abstractmethod

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from .base_model import BaseModelWrapper
from src.framework.core.data_models import ExperimentConfig
from src.text_complexity.text_evaluator import TextEvaluator
from src.text_complexity.response_formatter import ResponseFormatter

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    print("Warning: llama-cpp-python not available. Install with: pip install llama-cpp-python")


class LlamaCppBaseWrapper(BaseModelWrapper):
    """
    Base class for all llama.cpp GGUF model wrappers.
    
    Provides common functionality:
    - GGUF model loading
    - Prompt template formatting
    - logit_bias for vocabulary weighting
    - Response extraction
    
    Subclasses must implement:
    - _get_model_path(): Return path to GGUF file
    - _format_prompt(): Format prompt with model-specific template
    - _extract_response(): Extract assistant response from output
    """
    
    def __init__(self, 
                 model_name: str,
                 model_path: str,
                 n_ctx: int = 2048,
                 n_threads: int = 4,
                 n_gpu_layers: int = 0,
                 timeout_seconds: int = 300):
        """
        Initialize llama.cpp model wrapper.
        
        Args:
            model_name: Display name (e.g., "Qwen3", "SmolLM")
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: CPU threads to use
            n_gpu_layers: GPU layers (0 for CPU/Metal auto)
            timeout_seconds: Generation timeout
        """
        super().__init__(model_name, timeout_seconds)
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        
        # Initialize without tokenizer first, will be set after model loads
        self.text_evaluator = None
        self.response_formatter = ResponseFormatter()
        
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the GGUF model using llama.cpp."""
        if not LLAMACPP_AVAILABLE:
            print(f"❌ {self.model_name}: llama-cpp-python not installed")
            self.model_loaded = False
            return
        
        if not os.path.exists(self.model_path):
            print(f"❌ {self.model_name}: Model file not found at {self.model_path}")
            self.model_loaded = False
            return
        
        try:
            print(f"Loading {self.model_name} from {self.model_path}...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            self.model_loaded = True
            
            # Initialize text evaluator with tokenizer now that model is loaded
            self.text_evaluator = TextEvaluator(tokenizer=self._tokenize_text)
            
            print(f"✅ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"❌ {self.model_name}: Failed to load model: {e}")
            self.model_loaded = False
            # Fallback to evaluator without tokenizer
            self.text_evaluator = TextEvaluator()
    
    def _tokenize_text(self, text: str) -> list:
        """
        Tokenize text using the model's tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        if self.llm is None:
            return []
        return self.llm.tokenize(text.encode('utf-8'))
    
    @abstractmethod
    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        """
        Format prompt using model-specific template.
        
        Args:
            user_input: User's message
            system_prompt: System instruction
            
        Returns:
            Formatted prompt string
            
        Example for ChatML (Qwen):
            <|im_start|>system
            {system_prompt}<|im_end|>
            <|im_start|>user
            {user_input}<|im_end|>
            <|im_start|>assistant
        """
        pass
    
    @abstractmethod
    def _get_stop_tokens(self) -> List[str]:
        """
        Get stop tokens for this model.
        
        Returns:
            List of strings that signal end of generation
            
        Example for ChatML: ["<|im_end|>"]
        Example for Llama2: ["</s>"]
        """
        pass
    
    @abstractmethod
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean assistant response from raw model output.
        
        Args:
            raw_output: Raw text from model.generate()
            
        Returns:
            Cleaned response text
            
        Example: Remove thinking tags, template markers, etc.
        """
        pass
    
    def _create_logit_bias(self, vocab: List[str], weight_factor: float) -> Dict[int, float]:
        """
        Create logit_bias dictionary for vocabulary weighting.
        
        Args:
            vocab: List of words to weight
            weight_factor: Bias value (>1 increases probability)
            
        Returns:
            Dictionary mapping token IDs to bias values
        """
        if not self.llm or not vocab:
            return {}
        
        logit_bias = {}
        for word in vocab:
            try:
                # Tokenize the word
                tokens = self.llm.tokenize(word.encode('utf-8'))
                for token_id in tokens:
                    logit_bias[token_id] = weight_factor
            except Exception as e:
                # Skip words that fail to tokenize
                continue
        
        return logit_bias
    
    def _generate_response_impl(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate response using llama.cpp with the given configuration.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing response, timing, and success information
        """
        if not self.model_loaded or not self.llm:
            return {
                'response': '',
                'time_spent': 0.0,
                'generation_successful': False,
                'error_message': f'{self.model_name} model not loaded',
                'cleaned_response': ''
            }
        
        start_time = time.time()
        
        try:
            # Apply context prompting intervention if enabled
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(prompt)
            
            # Format prompt with model-specific template
            formatted_prompt = self._format_prompt(final_prompt, config.system_prompt)
            
            # Create logit_bias for probability weighting if enabled
            logit_bias = {}
            if config.config_weighting and self.target_vocabulary:
                logit_bias = self._create_logit_bias(
                    self.target_vocabulary,
                    config.weight_factor
                )
            
            # Generate response
            output = self.llm(
                formatted_prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop=self._get_stop_tokens(),
                echo=False,
                logit_bias=logit_bias if logit_bias else None
            )
            
            end_time = time.time()
            time_spent = end_time - start_time
            
            # Extract response text
            raw_response = output["choices"][0]["text"]
            response = self._extract_response(raw_response)
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            'backend': 'llama.cpp',
            'model_path': self.model_path,
            'model_format': 'GGUF',
            'context_window': self.n_ctx,
            'model_loaded': self.model_loaded
        })
        return info
    
    def cleanup(self):
        """
        Clean up llama.cpp model resources to free memory.
        
        Explicitly deletes the model instance and marks as unloaded.
        """
        if self.llm is not None:
            print(f"🧹 Cleaning up {self.model_name} model...")
            del self.llm
            self.llm = None
            self.model_loaded = False