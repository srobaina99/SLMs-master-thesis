"""
Phi-3-mini model wrapper using llama.cpp GGUF backend.

Phi-3-mini is Microsoft's 3.8B parameter model optimized for strong reasoning
and instruction-following capabilities in a compact form factor.

Model characteristics:
- 3.8B parameters
- 4k context window
- Strong reasoning capabilities
- Efficient architecture
"""

import os
import sys
from typing import Dict, Any, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from .llamacpp_base import LlamaCppBaseWrapper


class Phi3LlamaCppWrapper(LlamaCppBaseWrapper):
    """
    Phi-3-mini-4k-instruct model wrapper using llama.cpp GGUF backend.
    
    Model: microsoft/Phi-3-mini-4k-instruct-gguf (Q4 quantization)
    Template: Phi-3 custom format with <|system|>, <|user|>, <|assistant|> tags
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize Phi-3 llama.cpp wrapper.
        
        Args:
            model_path: Path to GGUF file. If None, uses default location.
        """
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "models",
                "gguf",
                "Phi-3-mini-4k-instruct-q4.gguf"
            )
        
        super().__init__(
            model_name="Phi3",
            model_path=model_path,
            n_ctx=4096,  # 4k context
            n_threads=4,
            n_gpu_layers=-1,  # Offload all layers to Metal GPU (73x faster!)
            timeout_seconds=300
        )
    
    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        """
        Format prompt using Phi-3's custom template.
        
        Phi-3 format:
        <|system|>
        {system_prompt}<|end|>
        <|user|>
        {user_input}<|end|>
        <|assistant|>
        
        Args:
            user_input: User's message
            system_prompt: System instruction
            
        Returns:
            Formatted Phi-3 prompt
        """
        return (
            f"<|system|>\n"
            f"{system_prompt}<|end|>\n"
            f"<|user|>\n"
            f"{user_input}<|end|>\n"
            f"<|assistant|>\n"
        )
    
    def _get_stop_tokens(self) -> List[str]:
        """
        Get stop tokens for Phi-3.
        
        Returns:
            List of Phi-3 stop tokens
        """
        return ["<|end|>", "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>"]
    
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean response from Phi-3 output.
        
        Handles:
        - Phi-3 special markers
        - Whitespace cleanup
        
        Args:
            raw_output: Raw text from llama.cpp
            
        Returns:
            Cleaned response text
        """
        response = raw_output.strip()
        
        # Remove Phi-3 markers
        for marker in ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"]:
            response = response.replace(marker, "")
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Phi-3-specific model information.
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info.update({
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'official_model': 'microsoft/Phi-3-mini-4k-instruct',
            'quantization': 'Q4',
            'template_format': 'Phi-3 (custom)',
            'parameters': '3.8B',
            'model_family': 'Phi-3 (Microsoft)',
            'context_window': 4096,
            'expected_tokens_per_sec': '~40-60',
            'expected_memory_mb': '~2200',
            'design_focus': 'Strong reasoning in compact form'
        })
        return info
