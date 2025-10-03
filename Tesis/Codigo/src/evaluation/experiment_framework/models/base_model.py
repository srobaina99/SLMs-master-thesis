"""
Abstract base class for model wrappers.
Defines the standardized interface that all model wrappers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
import sys
import time
import threading
from contextlib import contextmanager
try:
    import signal
    HAS_SIGNAL = True
except ImportError:
    HAS_SIGNAL = False

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 4 levels: models -> experiment_framework -> evaluation -> src -> Codigo
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(project_root)

from src.evaluation.experiment_framework.core.data_models import ExperimentConfig


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for implementing timeouts with fallback for different platforms."""
    if HAS_SIGNAL and hasattr(signal, 'SIGALRM'):
        # Unix-based systems with signal support
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore the old signal handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Fallback for systems without signal support (like Windows)
        # Note: This is a simpler approach that doesn't interrupt the actual generation
        # but at least provides timeout tracking
        start_time = time.time()
        
        class TimeoutTracker:
            def __init__(self):
                self.timed_out = False
                self.timer = None
            
            def timeout_callback(self):
                self.timed_out = True
            
            def start_timer(self):
                self.timer = threading.Timer(seconds, self.timeout_callback)
                self.timer.start()
            
            def stop_timer(self):
                if self.timer:
                    self.timer.cancel()
        
        tracker = TimeoutTracker()
        tracker.start_timer()
        
        try:
            yield
            if tracker.timed_out:
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
        finally:
            tracker.stop_timer()


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.
    
    Provides a standardized interface for generating responses with different
    models while supporting the experimental interventions (weighting, prompting).
    """
    
    def __init__(self, model_name: str, timeout_seconds: int = 300):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the model (e.g., "Qwen2", "TinyLlama")
            timeout_seconds: Maximum time to wait for generation (default: 5 minutes)
        """
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        # Vocab path: project_root is .../Codigo, vocab is at .../Codigo/data/vocabularies/
        self.vocab_path = os.path.join(project_root, "data", "vocabularies", "filtered_starters_vocab.txt")
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
    
    def generate_response(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Generate a response using the model with timeout protection.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing:
                - response: Generated text response (None if timeout)
                - time_spent: Time taken to generate response (timeout_seconds if timeout)
                - generation_successful: Boolean indicating success
                - error_message: Error message if generation failed
                - cleaned_response: Response after any cleanup
        """
        start_time = time.time()
        
        try:
            with timeout_context(self.timeout_seconds):
                return self._generate_response_impl(prompt, config)
        except TimeoutError as e:
            end_time = time.time()
            timeout_time = end_time - start_time
            
            print(f"⏰ {self.model_name} generation timed out after {timeout_time:.1f}s")
            
            return {
                'response': None,
                'time_spent': self.timeout_seconds,  # Record the timeout duration
                'generation_successful': False,
                'error_message': f'Generation timed out after {self.timeout_seconds} seconds',
                'cleaned_response': ''
            }
        except Exception as e:
            end_time = time.time()
            actual_time = end_time - start_time
            
            print(f"❌ {self.model_name} generation failed: {e}")
            
            return {
                'response': None,
                'time_spent': actual_time,
                'generation_successful': False,
                'error_message': str(e),
                'cleaned_response': ''
            }
    
    @abstractmethod
    def _generate_response_impl(self, prompt: str, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Internal implementation of response generation (to be implemented by subclasses).
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration containing intervention flags
            
        Returns:
            Dictionary containing response data
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
