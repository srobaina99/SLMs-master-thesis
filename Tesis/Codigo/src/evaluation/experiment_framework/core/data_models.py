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
    
    # Model parameters - updated for specification compliance
    model_name: str = "Qwen3"  # "Qwen2", "Qwen3", "TinyLlama", "Phi3", "SmolLM"
    model_id: str = "unsloth/Qwen3-0.6B"  # Actual model path/identifier
    system_prompt: str = "You are a helpful English teacher for beginner students."
    
    # Intervention flags - specification format
    config_weighting: bool = False  # Probability weighting intervention
    config_prompting: bool = False  # Context prompting intervention
    
    # Legacy fields for backward compatibility
    weighted_words_enabled: bool = False
    weight_factor: float = 1.0
    enable_thinking: bool = False
    verbose: bool = False
    
    # Experiment tracking
    prompt_id: str = ""  # Identifier for the specific prompt used
    
    # Generation parameters
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 200  # ~150 words max for beginner-appropriate responses
    
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
    model: str  # Model name from specification
    model_id: str  # Actual model path
    config_weighting: bool  # Specification format
    config_prompting: bool  # Specification format
    prompt_id: str  # Prompt identifier
    
    # Legacy fields for backward compatibility
    weighted_words_enabled: bool
    weight_factor: float
    enable_thinking: bool
    temperature: float
    
    # Performance metrics
    response_time_seconds: float
    
    # PRIMARY TEXT COMPLEXITY METRICS (4 metrics, non-redundant)
    # Grade Level Indices (3)
    flesch_kincaid_grade: float = 0.0  # Sentence structure & syllabic complexity
    gunning_fog: float = 0.0  # Polysyllabic word emphasis
    smog_index: float = 0.0  # Polysyllable density
    # Readability Scores (1)
    spache_readability: float = 0.0  # Primary-grade vocabulary (A1-focused)
    
    # SECONDARY DESCRIPTIVE STATISTICS (3 statistics)
    word_count: int = 0  # Verbosity/conciseness
    token_count: Optional[int] = None  # Actual tokens generated (if tokenizer available)
    difficult_words: int = 0  # Vocabulary accessibility
    
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
            
            # Specification format fields
            model=config.model_name,
            model_id=config.model_id,
            config_weighting=config.config_weighting,
            config_prompting=config.config_prompting,
            prompt_id=config.prompt_id,
            
            # Legacy fields for backward compatibility
            weighted_words_enabled=config.weighted_words_enabled,
            weight_factor=config.weight_factor,
            enable_thinking=config.enable_thinking,
            temperature=config.temperature,
            response_time_seconds=response_time,
            
            # PRIMARY METRICS - Grade level indices
            flesch_kincaid_grade=grade_indices.get('flesch_kincaid_grade', 0.0),
            gunning_fog=grade_indices.get('gunning_fog', 0.0),
            smog_index=grade_indices.get('smog_index', 0.0),
            
            # PRIMARY METRICS - Readability scores
            spache_readability=readability_scores.get('spache_readability', 0.0),
            
            # SECONDARY STATISTICS
            word_count=text_stats.get('word_count', 0),
            token_count=text_stats.get('token_count', None),
            difficult_words=text_stats.get('difficult_words', 0)
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
    
    def export_to_csv_specification_format(self, filename: str):
        """Export in exact format from ExperimentSpecification.md"""
        df = self.to_dataframe()
        if df.empty:
            print("No results to save.")
            return
        
        # Rename columns to match specification
        df_spec = df.copy()
        # Round time_spent to 1 decimal place (tenths of seconds)
        df_spec['time_spent'] = df_spec['response_time_seconds'].round(1)
        df_spec['answer'] = df_spec['response']
        
        # Select columns for specification format
        # Following streamlined metric set from docs/text_metrics.md
        spec_columns = [
            # Experiment configuration
            'model', 'config_weighting', 'config_prompting', 'weight_factor', 'prompt_id',
            # Response data
            'answer', 'time_spent',
            # PRIMARY METRICS: Grade Level Indices (3)
            'flesch_kincaid_grade', 'gunning_fog', 'smog_index',
            # PRIMARY METRICS: Readability Scores (1)
            'spache_readability',
            # SECONDARY STATISTICS (2)
            'word_count', 'difficult_words'
        ]
        
        # Filter to available columns
        available_columns = [col for col in spec_columns if col in df_spec.columns]
        df_export = df_spec[available_columns]
        
        # Save with comma as decimal separator (European format)
        df_export.to_csv(filename, index=False, decimal=',')
        print(f"Saved {len(self.results)} results in specification format to {filename}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all experiments."""
        if not self.results:
            return {}
        
        df = self.to_dataframe()
        
        # Key metrics to summarize (streamlined set from docs/text_metrics.md)
        numeric_columns = [
            'response_time_seconds',
            # PRIMARY METRICS
            'flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'spache_readability',
            # SECONDARY STATISTICS
            'word_count', 'difficult_words'
        ]
        
        # Overall summary
        summary = {
            'overall': {},
            'by_config': {},
            'metadata': {}
        }
        
        # Overall stats
        for col in numeric_columns:
            if col in df.columns:
                summary['overall'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        # Stats by intervention configuration
        if 'config_weighting' in df.columns and 'config_prompting' in df.columns:
            # Create intervention labels
            def get_config_label(row):
                if row['config_weighting'] and row['config_prompting']:
                    return 'both'
                elif row['config_weighting']:
                    return 'weighting_only'
                elif row['config_prompting']:
                    return 'prompting_only'
                else:
                    return 'control'
            
            df['intervention_config'] = df.apply(get_config_label, axis=1)
            
            # Calculate stats for each configuration
            for config in ['control', 'weighting_only', 'prompting_only', 'both']:
                config_df = df[df['intervention_config'] == config]
                if len(config_df) > 0:
                    summary['by_config'][config] = {
                        'count': len(config_df)
                    }
                    for col in numeric_columns:
                        if col in config_df.columns:
                            summary['by_config'][config][col] = {
                                'mean': config_df[col].mean(),
                                'std': config_df[col].std(),
                                'min': config_df[col].min(),
                                'max': config_df[col].max()
                            }
        
        # Metadata
        summary['metadata']['total_experiments'] = len(self.results)
        summary['metadata']['unique_prompts'] = df['prompt'].nunique()
        summary['metadata']['configs_tested'] = df['config_name'].nunique()
        if 'model' in df.columns:
            summary['metadata']['models_tested'] = df['model'].unique().tolist()
        
        return summary
    
    def clear(self):
        """Clear all stored results."""
        self.results.clear()
