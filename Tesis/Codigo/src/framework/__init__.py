"""
Factorial Experiment Framework

A clean, focused framework for running factorial experiments with small language models.
Implements the 4×4×N experimental design from ExperimentSpecification.md.

Key Components:
- ExperimentRunner: Main interface for running factorial experiments
- FactorialExperiment: Core experiment logic
- Model Wrappers: Standardized interfaces for all models
- Data Models: Structured data handling and CSV export
"""

__version__ = "2.0.0"
__author__ = "Santiago"

from .core.experiment_runner import ExperimentRunner, run_quick_factorial_test, run_single_model_test
from .core.data_models import ExperimentResult, ExperimentConfig, ExperimentDataManager
from .experiments.factorial_experiment import FactorialExperiment
from .experiments.experiment_configs import STANDARD_PROMPTS, create_factorial_configs

__all__ = [
    # Main interfaces
    'ExperimentRunner',
    'FactorialExperiment',
    
    # Data models
    'ExperimentResult', 
    'ExperimentConfig',
    'ExperimentDataManager',
    
    # Configuration and prompts
    'STANDARD_PROMPTS',
    'create_factorial_configs',
    
    # Convenience functions
    'run_quick_factorial_test',
    'run_single_model_test'
]