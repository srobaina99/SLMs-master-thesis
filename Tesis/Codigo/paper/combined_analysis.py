"""
Combined Analysis of Qwen2 and Qwen3 Factorial Experiments
Generates cross-model by-config comparisons for paper presentation
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_data():
    """Load both model datasets"""
    base_path = Path(__file__).parent.parent / "src/evaluation/experiment_framework/results/full_data"
    
    qwen2 = pd.read_csv(base_path / "Qwen2_factorial_full_20251002_083001.csv")
    qwen3 = pd.read_csv(base_path / "Qwen3_factorial_full_20251002_083455.csv")
    
    # Combine datasets
    combined = pd.concat([qwen2, qwen3], ignore_index=True)
    
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

def compute_statistics(df, groupby_cols, metrics):
    """Compute mean, std, min, max for metrics"""
    stats = {}
    
    for metric in metrics:
        grouped = df.groupby(groupby_cols)[metric]
        stats[metric] = pd.DataFrame({
            'mean': grouped.mean(),
            'std': grouped.std(),
            'min': grouped.min(),
            'max': grouped.max(),
            'count': grouped.count()
        })
    
    return stats

def generate_by_config_comparison(df):
    """Generate statistics by config across both models"""
    
    metrics = [
        'response_time_seconds',
        'flesch_kincaid_grade',
        'gunning_fog',
        'flesch_reading_ease',
        'word_count',
        'sentence_count',
        'difficult_words',
        'polysyllable_count',
        'monosyllable_count'
    ]
    
    # Overall by config (across both models)
    print("\n" + "="*80)
    print("BY-CONFIG ANALYSIS (QWEN2 + QWEN3 COMBINED)")
    print("="*80)
    
    config_order = ['Control', 'Weighting Only', 'Prompting Only', 'Both']
    
    for config in config_order:
        config_data = df[df['config_label'] == config]
        print(f"\n{'='*80}")
        print(f"CONFIG: {config.upper()}")
        print(f"{'='*80}")
        print(f"Sample size: {len(config_data)} observations")
        print(f"Models: Qwen2 ({len(config_data[config_data['model']=='Qwen2'])}) + " +
              f"Qwen3 ({len(config_data[config_data['model']=='Qwen3'])})")
        print()
        
        for metric in metrics:
            data = config_data[metric]
            print(f"{metric}:")
            print(f"  Mean: {data.mean():.2f}")
            print(f"  Std:  {data.std():.2f}")
            print(f"  Min:  {data.min():.2f}")
            print(f"  Max:  {data.max():.2f}")
        print()
    
    # By config AND model
    print("\n" + "="*80)
    print("BY-CONFIG AND MODEL (SEPARATE)")
    print("="*80)
    
    for config in config_order:
        print(f"\n{'='*80}")
        print(f"CONFIG: {config.upper()}")
        print(f"{'='*80}")
        
        for model in ['Qwen2', 'Qwen3']:
            config_model_data = df[(df['config_label'] == config) & (df['model'] == model)]
            print(f"\n  {model} (n={len(config_model_data)}):")
            
            for metric in ['flesch_kincaid_grade', 'gunning_fog', 'flesch_reading_ease', 
                          'response_time_seconds', 'word_count']:
                data = config_model_data[metric]
                if len(data) > 0:
                    print(f"    {metric}: {data.mean():.2f} ± {data.std():.2f}")

def create_comparison_visualizations(df):
    """Create comprehensive comparison visualizations"""
    
    # Create output directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    config_order = ['Control', 'Weighting Only', 'Prompting Only', 'Both']
    colors = {'Control': '#66c2a5', 'Weighting Only': '#fc8d62', 
              'Prompting Only': '#8da0cb', 'Both': '#e78ac3'}
    
    # Figure 1: Main complexity metrics by config (combined models)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('flesch_kincaid_grade', 'Flesch-Kincaid Grade Level', 'lower is better'),
        ('gunning_fog', 'Gunning Fog Index', 'lower is better'),
        ('flesch_reading_ease', 'Flesch Reading Ease', 'higher is better'),
        ('response_time_seconds', 'Response Time (seconds)', 'lower is better')
    ]
    
    for idx, (metric, title, note) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Create boxplot
        data_to_plot = [df[df['config_label'] == config][metric].values 
                       for config in config_order]
        
        bp = ax.boxplot(data_to_plot, labels=config_order, patch_artist=True)
        
        # Color boxes
        for patch, config in zip(bp['boxes'], config_order):
            patch.set_facecolor(colors[config])
            patch.set_alpha(0.7)
        
        ax.set_title(f'{title}\n({note})', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add target line for complexity metrics
        if metric in ['flesch_kincaid_grade', 'gunning_fog']:
            target = 5.0 if metric == 'flesch_kincaid_grade' else 6.0
            ax.axhline(y=target, color='red', linestyle='--', linewidth=2, 
                      label=f'A1 Target (≤{target})')
            ax.legend()
        elif metric == 'flesch_reading_ease':
            ax.axhline(y=80, color='green', linestyle='--', linewidth=2, 
                      label='A1 Target (≥80)')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_by_config_main_metrics.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'combined_by_config_main_metrics.png'}")
    
    # Figure 2: Model comparison for key metric (FK Grade)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(config_order))
    width = 0.35
    
    qwen2_means = [df[(df['model'] == 'Qwen2') & (df['config_label'] == config)]['flesch_kincaid_grade'].mean() 
                   for config in config_order]
    qwen3_means = [df[(df['model'] == 'Qwen3') & (df['config_label'] == config)]['flesch_kincaid_grade'].mean() 
                   for config in config_order]
    
    qwen2_stds = [df[(df['model'] == 'Qwen2') & (df['config_label'] == config)]['flesch_kincaid_grade'].std() 
                  for config in config_order]
    qwen3_stds = [df[(df['model'] == 'Qwen3') & (df['config_label'] == config)]['flesch_kincaid_grade'].std() 
                  for config in config_order]
    
    bars1 = ax.bar(x - width/2, qwen2_means, width, yerr=qwen2_stds, label='Qwen2', 
                   color='#1f77b4', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, qwen3_means, width, yerr=qwen3_stds, label='Qwen3', 
                   color='#ff7f0e', alpha=0.8, capsize=5)
    
    ax.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='A1 Target (≤5.0)')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Flesch-Kincaid Grade Level', fontsize=12, fontweight='bold')
    ax.set_title('Text Complexity by Model and Configuration\n(Lower is Simpler)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_fk_grade.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison_fk_grade.png'}")
    
    # Figure 3: Scatter plot - Response Time vs Complexity Trade-off
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for config in config_order:
        config_data = df[df['config_label'] == config]
        ax.scatter(config_data['response_time_seconds'], 
                  config_data['flesch_kincaid_grade'],
                  label=config, alpha=0.6, s=100, color=colors[config])
    
    # Add target zones
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.5, 
              label='A1 Complexity Target')
    ax.axvline(x=20, color='orange', linestyle='--', linewidth=2, alpha=0.5, 
              label='Response Time Threshold')
    
    # Shade acceptable region (low complexity, reasonable time)
    ax.fill_between([0, 20], 0, 5.0, color='green', alpha=0.1, label='Ideal Zone')
    
    ax.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Flesch-Kincaid Grade Level', fontsize=12, fontweight='bold')
    ax.set_title('Simplicity vs Speed Trade-off\n(Target: Bottom-Left Quadrant)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_vs_time_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'complexity_vs_time_tradeoff.png'}")
    
    plt.close('all')

def generate_summary_json(df):
    """Generate JSON summary of results"""
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    config_order = ['Control', 'Weighting Only', 'Prompting Only', 'Both']
    
    summary = {
        'overall': {},
        'by_config': {},
        'by_model_and_config': {},
        'effect_sizes': {}
    }
    
    # Overall statistics
    metrics = ['flesch_kincaid_grade', 'gunning_fog', 'flesch_reading_ease', 
              'response_time_seconds', 'word_count', 'difficult_words']
    
    for metric in metrics:
        summary['overall'][metric] = {
            'mean': float(df[metric].mean()),
            'std': float(df[metric].std()),
            'min': float(df[metric].min()),
            'max': float(df[metric].max())
        }
    
    # By config
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
                'std': float(config_data[metric].std())
            }
    
    # By model and config
    for model in ['Qwen2', 'Qwen3']:
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
                        'std': float(data[metric].std())
                    }
    
    # Effect sizes (vs control)
    for model in ['Qwen2', 'Qwen3']:
        control_data = df[(df['model'] == model) & (df['config_label'] == 'Control')]
        summary['effect_sizes'][model] = {}
        
        for config in ['Weighting Only', 'Prompting Only', 'Both']:
            config_data = df[(df['model'] == model) & (df['config_label'] == config)]
            summary['effect_sizes'][model][config] = {}
            
            for metric in ['flesch_kincaid_grade', 'flesch_reading_ease']:
                if len(control_data) > 0 and len(config_data) > 0:
                    control_mean = control_data[metric].mean()
                    config_mean = config_data[metric].mean()
                    
                    # Percent change
                    pct_change = ((config_mean - control_mean) / control_mean) * 100
                    
                    # Cohen's d
                    pooled_std = np.sqrt(
                        (control_data[metric].std()**2 + config_data[metric].std()**2) / 2
                    )
                    cohens_d = (config_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                    
                    summary['effect_sizes'][model][config][metric] = {
                        'control_mean': float(control_mean),
                        'intervention_mean': float(config_mean),
                        'percent_change': float(pct_change),
                        'cohens_d': float(cohens_d)
                    }
    
    # Save
    with open(output_dir / 'combined_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: {output_dir / 'combined_summary.json'}")

def generate_example_table(df):
    """Generate example outputs for visual hook"""
    
    # Get one example from each config for the same prompt
    prompt = "What does the word 'library' mean?"
    
    print("\n" + "="*80)
    print("EXAMPLE OUTPUTS: VISUAL HOOK")
    print("="*80)
    print(f"\nPrompt: '{prompt}'\n")
    
    config_order = ['Control', 'Both']
    
    for config in config_order:
        examples = df[(df['config_label'] == config) & (df['prompt'] == prompt)]
        
        if len(examples) > 0:
            # Take first example
            example = examples.iloc[0]
            
            print(f"\n{'='*80}")
            print(f"{config.upper()} ({example['model']})")
            print(f"{'='*80}")
            print(f"Response: {example['cleaned_response'][:300]}...")
            print(f"\nMetrics:")
            print(f"  Flesch-Kincaid Grade: {example['flesch_kincaid_grade']:.1f}")
            print(f"  Flesch Reading Ease: {example['flesch_reading_ease']:.1f}")
            print(f"  Word Count: {int(example['word_count'])}")
            print(f"  Difficult Words: {int(example['difficult_words'])}")

def main():
    """Main analysis pipeline"""
    
    print("Loading data...")
    df = load_data()
    df = create_config_labels(df)
    
    print(f"\nTotal observations: {len(df)}")
    print(f"Models: {df['model'].unique()}")
    print(f"Configs: {df['config_label'].unique()}")
    
    # Generate analyses
    generate_by_config_comparison(df)
    create_comparison_visualizations(df)
    generate_summary_json(df)
    generate_example_table(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - paper/figures/*.png (3 visualizations)")
    print("  - paper/results/combined_summary.json")

if __name__ == "__main__":
    main()


