"""
Qwen3 model wrapper using llama.cpp GGUF backend.

This replaces the Transformers-based Qwen3Wrapper with a faster,
more memory-efficient llama.cpp implementation.

Benchmark results (vs Transformers+MPS):
- 4.4x faster inference (98 vs 22 tok/s)
- 57% less memory (491MB vs 1137MB)
- Supports probability weighting via logit_bias
"""

import os
import sys
from typing import Dict, Any, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.append(project_root)

from .llamacpp_base import LlamaCppBaseWrapper


class Qwen3LlamaCppWrapper(LlamaCppBaseWrapper):
    """
    Qwen3-0.6B model wrapper using llama.cpp GGUF backend.
    
    Model: ggml-org/Qwen3-0.6B-GGUF (Q4_0 quantization)
    Template: ChatML format
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize Qwen3 llama.cpp wrapper.
        
        Args:
            model_path: Path to GGUF file. If None, uses default location.
        """
        # Default model path
        # project_root = .../Tesis (from base_model.py calculation)
        # Model is at: .../Tesis/Codigo/models/gguf/
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "Codigo",
                "models",
                "gguf",
                "Qwen3-0.6B-Q4_0.gguf"
            )
        
        super().__init__(
            model_name="Qwen3",
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # Metal auto-detection
            timeout_seconds=300
        )
    
    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        """
        Format prompt using Qwen's ChatML template.
        
        ChatML format:
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
        Get stop tokens for Qwen3.
        
        Returns:
            List of ChatML stop tokens
        """
        return ["<|im_end|>", "<|endoftext|>"]
    
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean response from Qwen3 output.
        
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
        Get Qwen3-specific model information.
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info.update({
            'model_id': 'ggml-org/Qwen3-0.6B-GGUF',
            'quantization': 'Q4_0',
            'template_format': 'ChatML',
            'parameters': '0.6B',
            'benchmark_tokens_per_sec': 98.1,
            'benchmark_memory_mb': 491
        })
        return info
