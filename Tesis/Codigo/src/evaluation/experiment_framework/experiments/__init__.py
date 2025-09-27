"""
Experiment orchestration layer.
Contains classes for running structured experiments with multiple models and configurations.
"""

from .factorial_experiment import FactorialExperiment
from .experiment_configs import STANDARD_PROMPTS, create_factorial_configs

__all__ = [
    'FactorialExperiment',
    'STANDARD_PROMPTS',
    'create_factorial_configs'
]
