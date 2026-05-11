"""
TinyLlama-1.1B model wrapper using llama.cpp GGUF backend.

This replaces the legacy TinyLlamaWrapper with a standardized
llama.cpp implementation following the LlamaCppBaseWrapper pattern.

Expected performance:
- Fast inference on M2 Mac
- Low memory footprint (~600MB)
- Supports probability weighting via logit_bias
"""

import os
import sys
from typing import Dict, Any, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from .llamacpp_base import LlamaCppBaseWrapper


class TinyLlamaLlamaCppWrapper(LlamaCppBaseWrapper):
    """
    TinyLlama-1.1B-Chat model wrapper using llama.cpp GGUF backend.
    
    Model: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (Q4_0 quantization)
    Template: TinyLlama custom format (simplified Llama2-style)
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize TinyLlama llama.cpp wrapper.
        
        Args:
            model_path: Path to GGUF file. If None, uses default location.
        """
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "models",
                "gguf",
                "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
            )
        
        super().__init__(
            model_name="TinyLlama",
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # Metal auto-detection
            timeout_seconds=300
        )
    
    def _format_prompt(self, user_input: str, system_prompt: str, enable_thinking: bool = False) -> str:
        """
        Format prompt using TinyLlama's custom template.
        
        TinyLlama format (from original code):
        <|system|>
        {system_prompt}
        <|user|>
        {user_input}
        <|assistant|>
        
        Args:
            user_input: User's message
            system_prompt: System instruction
            
        Returns:
            Formatted TinyLlama prompt
        """
        return (
            f"<|system|>\n"
            f"{system_prompt}\n"
            f"<|user|>\n"
            f"{user_input}\n"
            f"<|assistant|>"
        )
    
    def _get_stop_tokens(self) -> List[str]:
        """
        Get stop tokens for TinyLlama.
        
        Returns:
            List of TinyLlama stop tokens
        """
        return ["<|user|>", "<|system|>", "</s>"]
    
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean response from TinyLlama output.
        
        Handles:
        - Template markers
        - Whitespace cleanup
        
        Args:
            raw_output: Raw text from llama.cpp
            
        Returns:
            Cleaned response text
        """
        response = raw_output.strip()
        
        # Remove template markers
        response = response.replace("<|system|>", "")
        response = response.replace("<|user|>", "")
        response = response.replace("<|assistant|>", "")
        response = response.replace("</s>", "")
        
        # Clean up any remaining artifacts
        lines = response.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        return ' '.join(cleaned_lines)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get TinyLlama-specific model information.
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info.update({
            'model_id': 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
            'quantization': 'Q4_0',
            'template_format': 'TinyLlama',
            'parameters': '1.1B',
            'expected_tokens_per_sec': '~80-100',
            'expected_memory_mb': '~600'
        })
        return info
