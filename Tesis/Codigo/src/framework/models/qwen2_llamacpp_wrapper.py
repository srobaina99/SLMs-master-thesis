"""
Qwen2.5-0.5B model wrapper using llama.cpp GGUF backend.

This replaces the Transformers-based Qwen2Wrapper with a faster,
more memory-efficient llama.cpp implementation.

Expected performance (based on Qwen3 benchmarks):
- ~4x faster inference vs Transformers
- ~50% less memory usage
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


class Qwen2LlamaCppWrapper(LlamaCppBaseWrapper):
    """
    Qwen2.5-0.5B-Instruct model wrapper using llama.cpp GGUF backend.
    
    Model: Qwen/Qwen2.5-0.5B-Instruct-GGUF (Q4_0 quantization)
    Template: ChatML format (same as Qwen3)
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize Qwen2 llama.cpp wrapper.
        
        Args:
            model_path: Path to GGUF file. If None, uses default location.
        """
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "models",
                "gguf",
                "qwen2.5-0.5b-instruct-q4_0.gguf"
            )
        
        super().__init__(
            model_name="Qwen2",
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # Metal auto-detection
            timeout_seconds=300
        )
    
    def _format_prompt(self, user_input: str, system_prompt: str, enable_thinking: bool = False) -> str:
        """
        Format prompt using Qwen's ChatML template.
        
        ChatML format (same as Qwen3):
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_input}<|im_end|>
        <|im_start|>assistant
        
        Args:
            user_input: User's message
            system_prompt: System instruction
            
        Returns:
            Formatted ChatML prompt
        """
        return (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def _get_stop_tokens(self) -> List[str]:
        """
        Get stop tokens for Qwen2.
        
        Returns:
            List of ChatML stop tokens
        """
        return ["<|im_end|>", "<|endoftext|>"]
    
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean response from Qwen2 output.
        
        Handles:
        - Thinking tags (<think>...</think>)
        - ChatML markers
        - Whitespace cleanup
        
        Args:
            raw_output: Raw text from llama.cpp
            
        Returns:
            Cleaned response text
        """
        response = raw_output.strip()
        
        # Remove thinking tags if present
        if "<think>" in response and "</think>" in response:
            # Extract content after thinking
            think_end = response.find("</think>")
            response = response[think_end + len("</think>"):].strip()
        
        # Remove any remaining ChatML markers
        response = response.replace("<|im_start|>", "")
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|endoftext|>", "")
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Qwen2-specific model information.
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info.update({
            'model_id': 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
            'quantization': 'Q4_0',
            'template_format': 'ChatML',
            'parameters': '0.5B',
            'expected_tokens_per_sec': '~90-100',  # Based on Qwen3 benchmarks
            'expected_memory_mb': '~400-500'
        })
        return info
