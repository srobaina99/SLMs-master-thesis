"""
SmolLM-1.7B model wrapper using llama.cpp GGUF backend.

SmolLM is designed for efficient on-device deployment with strong performance
for its size. Part of HuggingFace's SmolLM family (135M, 360M, 1.7B).

Expected performance:
- Fast inference on M2 Mac
- Low memory footprint (~1GB)
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


class SmolLMLlamaCppWrapper(LlamaCppBaseWrapper):
    """
    SmolLM-1.7B-Instruct model wrapper using llama.cpp GGUF backend.
    
    Model: MaziyarPanahi/SmolLM-1.7B-Instruct-GGUF (Q4_K_M quantization)
    Template: ChatML format (same as Qwen)
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize SmolLM llama.cpp wrapper.
        
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
                "SmolLM-1.7B-Instruct.Q4_K_M.gguf"
            )
        
        super().__init__(
            model_name="SmolLM",
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # Metal auto-detection
            timeout_seconds=300
        )
    
    def _format_prompt(self, user_input: str, system_prompt: str) -> str:
        """
        Format prompt using SmolLM's ChatML template.
        
        ChatML format (same as Qwen):
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
        Get stop tokens for SmolLM.
        
        Returns:
            List of ChatML stop tokens
        """
        return ["<|im_end|>", "<|endoftext|>"]
    
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean response from SmolLM output.
        
        Handles:
        - ChatML markers
        - Whitespace cleanup
        
        Args:
            raw_output: Raw text from llama.cpp
            
        Returns:
            Cleaned response text
        """
        response = raw_output.strip()
        
        # Remove ChatML markers
        response = response.replace("<|im_start|>", "")
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|endoftext|>", "")
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get SmolLM-specific model information.
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info.update({
            'model_id': 'MaziyarPanahi/SmolLM-1.7B-Instruct-GGUF',
            'official_model': 'HuggingFaceTB/SmolLM-1.7B-Instruct',
            'quantization': 'Q4_K_M',
            'template_format': 'ChatML',
            'parameters': '1.7B',
            'model_family': 'SmolLM (HuggingFace)',
            'expected_tokens_per_sec': '~80-100',
            'expected_memory_mb': '~1000',
            'design_focus': 'Efficient on-device deployment'
        })
        return info
