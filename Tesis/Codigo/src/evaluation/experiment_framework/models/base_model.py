"""
Abstract base class for model wrappers.
Defines the standardized interface that all model wrappers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.data_models import ExperimentConfig


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.
    
    Provides a standardized interface for generating responses with different
    models while supporting the experimental interventions (weighting, prompting).
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the model (e.g., "Qwen2", "TinyLlama")
        """
        self.model_name = model_name
        self.vocab_path = os.path.join(project_root, "Tesis", "Codigo", "data", "vocabularies", "filtered_starters_vocab.txt")
        self.target_vocabulary = self._load_target_vocabulary()
    
    def _load_target_vocabulary(self) -> List[str]:
        """Load the target vocabulary for weighting."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab = [line.strip().lower() for line in f if line.strip()]
            print(f"Loaded {len(vocab)} target vocabulary words for {self.model_name}")
            return vocab
        except FileNotFoundError:
            print(f"Warning: Target vocabulary file not found at {self.vocab_path}")
            return []
    
    def _add_simplification_context(self, prompt: str) -> str:
        """
        Add simplification context to prompt for context prompting intervention.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Prompt with simplification context added
        """
        context = """# Context
Please respond using simple words that a young non-English speaking student can understand. 
Use vocabulary from basic English learning materials. Keep sentences short and clear.
Avoid complex grammar structures and difficult words.

"""
        return context + prompt
    
    @abstractmethod
    def generate_response(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate a response using the model with the given configuration.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing:
                - response: Generated text response
                - time_spent: Time taken to generate response (seconds)
                - generation_successful: Boolean indicating success
                - error_message: Error message if generation failed
                - cleaned_response: Response after any cleanup
        """
        pass
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the underlying model. Called once during wrapper creation."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'vocab_size': len(self.target_vocabulary),
            'supports_weighting': True,  # All wrappers should support this
            'supports_context_prompting': True
        }
