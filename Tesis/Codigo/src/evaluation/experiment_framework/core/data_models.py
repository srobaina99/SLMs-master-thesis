"""
Data models for experiment framework.
Defines the structure for experiment configurations and results.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import uuid


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    # Model parameters
    model_id: str = "unsloth/Qwen3-0.6B"
    system_prompt: str = "You are a helpful English teacher for beginner students."
    weighted_words_enabled: bool = False
    weight_factor: float = 1.0
    enable_thinking: bool = False
    verbose: bool = False
    
    # Generation parameters
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 1024
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single prompt-response experiment."""
    
    # Identifiers
    experiment_id: str
    timestamp: datetime
    config_name: str
    
    # Input/Output
    prompt: str
    system_prompt: str
    response: str
    
    # Model parameters (flattened for easy analysis)
    model_id: str
    weighted_words_enabled: bool
    weight_factor: float
    enable_thinking: bool
    temperature: float
    
    # Performance metrics
    response_time_seconds: float
    
    # Text complexity metrics (from text_evaluator.py)
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    automated_readability_index: float = 0.0
    coleman_liau_index: float = 0.0
    dale_chall_readability_score: float = 0.0
    flesch_reading_ease: float = 0.0
    linsear_write_formula: float = 0.0
    spache_readability: float = 0.0
    mcalpine_eflaw: float = 0.0
    
    # Text statistics
    sentence_count: int = 0
    word_count: int = 0
    character_count: int = 0
    syllable_count: int = 0
    polysyllable_count: int = 0
    monosyllable_count: int = 0
    difficult_words: int = 0
    
    # Reading time
    reading_time_seconds: float = 0.0
    reading_time_minutes: float = 0.0
    
    # Response formatting
    cleaned_response: str = ""  # Response after formatting cleanup
    
    # Optional manual evaluation fields
    response_appropriateness: Optional[float] = None  # 1-5 scale
    vocabulary_level: Optional[str] = None  # 'beginner', 'elementary', etc.
    notes: Optional[str] = None
    
    @classmethod
    def create_from_response(cls, 
                           prompt: str,
                           response: str,
                           config: ExperimentConfig,
                           response_time: float,
                           text_metrics: Dict[str, Any],
                           experiment_name: str = "default",
                           cleaned_response: str = "") -> 'ExperimentResult':
        """Create ExperimentResult from response data and metrics."""
        
        # Extract metrics safely with defaults
        grade_indices = text_metrics.get('grade_level_indices', {})
        readability_scores = text_metrics.get('readability_scores', {})
        text_stats = text_metrics.get('text_statistics', {})
        reading_time = text_metrics.get('reading_time', {})
        
        return cls(
            experiment_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            config_name=experiment_name,
            prompt=prompt,
            system_prompt=config.system_prompt,
            response=response,
            cleaned_response=cleaned_response,
            model_id=config.model_id,
            weighted_words_enabled=config.weighted_words_enabled,
            weight_factor=config.weight_factor,
            enable_thinking=config.enable_thinking,
            temperature=config.temperature,
            response_time_seconds=response_time,
            
            # Grade level indices
            flesch_kincaid_grade=grade_indices.get('flesch_kincaid_grade', 0.0),
            gunning_fog=grade_indices.get('gunning_fog', 0.0),
            smog_index=grade_indices.get('smog_index', 0.0),
            automated_readability_index=grade_indices.get('automated_readability_index', 0.0),
            coleman_liau_index=grade_indices.get('coleman_liau_index', 0.0),
            dale_chall_readability_score=grade_indices.get('dale_chall_readability_score', 0.0),
            
            # Readability scores
            flesch_reading_ease=readability_scores.get('flesch_reading_ease', 0.0),
            linsear_write_formula=readability_scores.get('linsear_write_formula', 0.0),
            spache_readability=readability_scores.get('spache_readability', 0.0),
            mcalpine_eflaw=readability_scores.get('mcalpine_eflaw', 0.0),
            
            # Text statistics
            sentence_count=text_stats.get('sentence_count', 0),
            word_count=text_stats.get('word_count', 0),
            character_count=text_stats.get('character_count', 0),
            syllable_count=text_stats.get('syllable_count', 0),
            polysyllable_count=text_stats.get('polysyllable_count', 0),
            monosyllable_count=text_stats.get('monosyllable_count', 0),
            difficult_words=text_stats.get('difficult_words', 0),
            
            # Reading time
            reading_time_seconds=reading_time.get('reading_time_seconds', 0.0),
            reading_time_minutes=reading_time.get('reading_time_minutes', 0.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result_dict = asdict(self)
        # Convert datetime to string for serialization
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict


class ExperimentDataManager:
    """Manages experiment data storage and retrieval."""
    
    def __init__(self):
        self.results: List[ExperimentResult] = []
    
    def add_result(self, result: ExperimentResult):
        """Add a single experiment result."""
        self.results.append(result)
    
    def add_results(self, results: List[ExperimentResult]):
        """Add multiple experiment results."""
        self.results.extend(results)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        data = [result.to_dict() for result in self.results]
        return pd.DataFrame(data)
    
    def save_to_parquet(self, filename: str):
        """Save results to Parquet file for easy Google Sheets import."""
        df = self.to_dataframe()
        if df.empty:
            print("No results to save.")
            return
        
        df.to_parquet(filename, index=False)
        print(f"Saved {len(self.results)} results to {filename}")
    
    def save_to_csv(self, filename: str):
        """Save results to CSV file as backup option."""
        df = self.to_dataframe()
        if df.empty:
            print("No results to save.")
            return
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(self.results)} results to {filename}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all experiments."""
        if not self.results:
            return {}
        
        df = self.to_dataframe()
        
        # Key metrics to summarize
        numeric_columns = [
            'response_time_seconds', 'flesch_kincaid_grade', 'gunning_fog',
            'flesch_reading_ease', 'word_count', 'sentence_count'
        ]
        
        summary = {}
        for col in numeric_columns:
            if col in df.columns:
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        summary['total_experiments'] = len(self.results)
        summary['unique_prompts'] = df['prompt'].nunique()
        summary['configs_tested'] = df['config_name'].nunique()
        
        return summary
    
    def clear(self):
        """Clear all stored results."""
        self.results.clear()
