"""
Recalculate Combined Analysis with Median Statistics
Generates JSON summary from CSV results including both mean and median
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_data():
    """Load all model datasets from latest experiment results"""
    base_path = Path(__file__).parent.parent / "src/evaluation/experiment_framework/results"
    
    qwen2 = pd.read_csv(base_path / "Qwen2/full_data/Qwen2_full_experiment_full_1023_0025.csv")
    qwen3 = pd.read_csv(base_path / "Qwen3/full_data/Qwen3_full_experiment_full_1023_0027.csv")
    phi3 = pd.read_csv(base_path / "Phi3/full_data/Phi3_full_experiment_full_1023_0034.csv")
    tinyllama = pd.read_csv(base_path / "TinyLlama/full_data/TinyLlama_full_experiment_full_1023_0029.csv")
    
    # Combine all datasets
    combined = pd.concat([qwen2, qwen3, phi3, tinyllama], ignore_index=True)
    
    return combined

def create_config_labels(df):
    """Create readable config labels"""
    def label_config(row):
        if not row['config_weighting'] and not row['config_prompting']:
            return 'Control'
        elif row['config_weighting'] and not row['config_prompting']:
            return 'Weighting Only'
        elif not row['config_weighting'] and row['config_prompting']:
            return 'Prompting Only'
        else:
            return 'Both'
    
    df['config_label'] = df.apply(label_config, axis=1)
    return df

def generate_summary_json_with_median(df):
    """Generate JSON summary of results with median statistics"""
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    config_order = ['Control', 'Weighting Only', 'Prompting Only', 'Both']
    
    summary = {
        'overall': {},
        'by_config': {},
        'by_model_and_config': {},
        'effect_sizes': {}
    }
    
    # Streamlined metrics from docs/text_metrics.md
    metrics = ['flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'spache_readability',
              'response_time_seconds', 'word_count', 'difficult_words']
    
    # Overall statistics with median
    for metric in metrics:
        summary['overall'][metric] = {
            'mean': float(df[metric].mean()),
            'median': float(df[metric].median()),
            'std': float(df[metric].std()),
            'min': float(df[metric].min()),
            'max': float(df[metric].max())
        }
    
    # By config with median
    for config in config_order:
        config_data = df[df['config_label'] == config]
        summary['by_config'][config] = {
            'n': len(config_data),
            'models': {
                'Qwen2': len(config_data[config_data['model'] == 'Qwen2']),
                'Qwen3': len(config_data[config_data['model'] == 'Qwen3'])
            }
        }
        
        for metric in metrics:
            summary['by_config'][config][metric] = {
                'mean': float(config_data[metric].mean()),
                'median': float(config_data[metric].median()),
                'std': float(config_data[metric].std())
            }
    
    # By model and config with median
    for model in ['Qwen2', 'Qwen3', 'Phi3', 'TinyLlama']:
        summary['by_model_and_config'][model] = {}
        for config in config_order:
            data = df[(df['model'] == model) & (df['config_label'] == config)]
            summary['by_model_and_config'][model][config] = {
                'n': len(data)
            }
            for metric in metrics:
                if len(data) > 0:
                    summary['by_model_and_config'][model][config][metric] = {
                        'mean': float(data[metric].mean()),
                        'median': float(data[metric].median()),
                        'std': float(data[metric].std())
                    }
    
    # Effect sizes (vs control) with median
    for model in ['Qwen2', 'Qwen3', 'Phi3', 'TinyLlama']:
        control_data = df[(df['model'] == model) & (df['config_label'] == 'Control')]
        summary['effect_sizes'][model] = {}
        
        for config in ['Weighting Only', 'Prompting Only', 'Both']:
            config_data = df[(df['model'] == model) & (df['config_label'] == config)]
            summary['effect_sizes'][model][config] = {}
            
            # Calculate effect sizes for primary complexity metrics
            for metric in ['flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'spache_readability']:
                if len(control_data) > 0 and len(config_data) > 0:
                    control_mean = control_data[metric].mean()
                    control_median = control_data[metric].median()
                    config_mean = config_data[metric].mean()
                    config_median = config_data[metric].median()
                    
                    # Percent change (based on mean)
                    pct_change = ((config_mean - control_mean) / control_mean) * 100
                    
                    # Percent change for median
                    pct_change_median = ((config_median - control_median) / control_median) * 100
                    
                    # Cohen's d
                    pooled_std = np.sqrt(
                        (control_data[metric].std()**2 + config_data[metric].std()**2) / 2
                    )
                    cohens_d = (config_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                    
                    summary['effect_sizes'][model][config][metric] = {
                        'control_mean': float(control_mean),
                        'control_median': float(control_median),
                        'intervention_mean': float(config_mean),
                        'intervention_median': float(config_median),
                        'percent_change': float(pct_change),
                        'percent_change_median': float(pct_change_median),
                        'cohens_d': float(cohens_d)
                    }
    
    # Save with MEDIAN in filename
    output_path = output_dir / 'combined_summary_MEDIAN.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved: {output_path}")
    return summary

def main():
    """Main pipeline"""
    
    print("Loading data from latest experiment results...")
    df = load_data()
    df = create_config_labels(df)
    
    print(f"Total observations: {len(df)}")
    print(f"Models: {df['model'].unique()}")
    print(f"Configs: {df['config_label'].unique()}")
    
    print("\nGenerating JSON summary with median statistics...")
    summary = generate_summary_json_with_median(df)
    
    print("\n" + "="*80)
    print("SUMMARY GENERATION COMPLETE")
    print("="*80)
    print(f"Generated file: paper/results/combined_summary_MEDIAN.json")
    print(f"Total metrics processed: 7")
    print(f"Configurations: 4 (Control, Weighting Only, Prompting Only, Both)")
    print(f"Models: 4 (Qwen2, Qwen3, Phi3, TinyLlama)")

if __name__ == "__main__":
    main()

