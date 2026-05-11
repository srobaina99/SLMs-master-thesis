"""
Model wrapper layer for experiment framework.
Provides standardized interfaces for all supported models.

All primary models now use llama.cpp GGUF backend for optimal performance.
"""

from .base_model import BaseModelWrapper
from .llamacpp_base import LlamaCppBaseWrapper
from .phi3_llamacpp_wrapper import Phi3LlamaCppWrapper
from .qwen2_llamacpp_wrapper import Qwen2LlamaCppWrapper
from .qwen3_llamacpp_wrapper import Qwen3LlamaCppWrapper
from .tinyllama_llamacpp_wrapper import TinyLlamaLlamaCppWrapper
__all__ = [
    'BaseModelWrapper',
    'LlamaCppBaseWrapper',
    'Phi3LlamaCppWrapper',
    'Qwen2LlamaCppWrapper',
    'Qwen3LlamaCppWrapper',
    'TinyLlamaLlamaCppWrapper',
]
