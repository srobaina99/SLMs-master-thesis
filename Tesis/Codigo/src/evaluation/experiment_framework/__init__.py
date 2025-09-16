"""
English Learning Conversation Experiment Framework

A framework for evaluating small language models in English teaching scenarios.
Integrates with existing Qwen3 implementation and text complexity evaluation tools.
"""

__version__ = "1.0.0"
__author__ = "Santiago"

from .core.experiment_runner import ExperimentRunner
from .core.data_models import ExperimentResult, ExperimentConfig
from .prompts.english_learning_prompts import EnglishLearningPrompts

__all__ = [
    'ExperimentRunner',
    'ExperimentResult', 
    'ExperimentConfig',
    'EnglishLearningPrompts'
]
