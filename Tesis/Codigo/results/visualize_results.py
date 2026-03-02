#!/usr/bin/env python3
"""
Visualization script for factorial experiment results.
Creates boxplots for each metric grouped by intervention configurations.

Works with the new model-based folder structure:
    results/
    ├── Qwen2/
    │   ├── *_specification_*.csv
    │   └── plots/  (generated)
    ├── Qwen3/
    ├── SmolLM/
    └── TinyLlama/

Usage:
    # Visualize all models
    python visualize_results.py
    
    # Visualize specific model
    python visualize_results.py --model SmolLM
    python visualize_results.py --model all
    
    # Visualize specific CSV file
    python visualize_results.py path/to/file.csv
    
    # Visualize by model name shorthand
    python visualize_results.py SmolLM
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results CSV with proper decimal handling."""
    try:
        # Try loading with comma as decimal separator
        df = pd.read_csv(csv_path, decimal=',')
    except:
        # Fallback to period as decimal separator
        df = pd.read_csv(csv_path)
    
    return df


def create_config_label(row) -> str:
    """Create readable configuration label from weighting/prompting flags."""
    if row['config_weighting'] and row['config_prompting']:
        return 'Both'
    elif row['config_weighting']:
        return 'Weighting'
    elif row['config_prompting']:
        return 'Prompting'
    else:
        return 'Control'


def create_single_metric_subplot(df: pd.DataFrame, metric: str, ax, config_order: List[str]):
    """
    Create a single boxplot on a given axis.
    
    Args:
        df: DataFrame with results
        metric: Column name to plot
        ax: Matplotlib axis to plot on
        config_order: Order of configurations
    """
    if metric not in df.columns:
        ax.set_visible(False)
        return False
    
    # Create boxplot
    sns.boxplot(
        data=df,
        x='config_label',
        y=metric,
        hue='config_label',
        order=config_order,
        palette='Set2',
        legend=False,
        ax=ax
    )
    
    # Add individual points
    sns.stripplot(
        data=df,
        x='config_label',
        y=metric,
        order=config_order,
        color='black',
        alpha=0.3,
        size=3,
        ax=ax
    )
    
    # Formatting
    ax.set_title(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')
    ax.set_xlabel('', fontsize=9)
    ax.set_ylabel('', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    return True


def plot_all_metrics(csv_path: str, output_subdir: str = 'plots',
                     metrics: Optional[List[str]] = None):
    """
    Generate single comprehensive visualization with all metrics from a results CSV.
    
    Args:
        csv_path: Path to the CSV file
        output_subdir: Subdirectory name for plots (default: 'plots')
        metrics: Optional list of specific metrics to plot (default: all readability metrics)
    """
    csv_path = Path(csv_path)
    
    # Load data
    print(f"\n📊 Loading results from: {csv_path.name}")
    df = load_results(csv_path)
    print(f"   Loaded {len(df)} rows")
    
    # Extract model name and experiment name from parent directory or filename
    if csv_path.parent.name in ['Qwen2', 'Qwen3', 'SmolLM', 'TinyLlama', 'Phi3']:
        model_name = csv_path.parent.name
    else:
        model_name = csv_path.stem.split('_')[0] if '_' in csv_path.stem else 'Model'
    
    # Extract experiment name from filename (e.g., "Qwen2_multi_weight_specification_1005_1848.csv" -> "multi_weight")
    filename_parts = csv_path.stem.split('_')
    if 'specification' in filename_parts:
        spec_idx = filename_parts.index('specification')
        experiment_name = '_'.join(filename_parts[1:spec_idx])  # Everything between model name and "specification"
    else:
        experiment_name = 'factorial'
    
    # Create configuration labels
    df['config_label'] = df.apply(create_config_label, axis=1)
    config_order = ['Control', 'Weighting', 'Prompting', 'Both']
    config_order = [c for c in config_order if c in df['config_label'].values]
    
    # Create output directory in the same model folder
    output_dir = csv_path.parent / output_subdir
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Saving plots to: {csv_path.parent.name}/{output_subdir}/")
    
    # Define default metrics if not specified
    # Following streamlined metric set from docs/text_metrics.md
    if metrics is None:
        metrics = [
            # Performance metric
            'time_spent',
            
            # PRIMARY METRICS: Grade Level Indices (3)
            'flesch_kincaid_grade',
            'gunning_fog',
            'smog_index',
            
            # PRIMARY METRICS: Readability Scores (1)
            'spache_readability',
            
            # SECONDARY STATISTICS (2)
            'word_count',
            'difficult_words'
        ]
    
    # Filter to metrics present in the data
    available_metrics = [m for m in metrics if m in df.columns]
    
    print(f"\n🎨 Generating comprehensive plot with {len(available_metrics)} metrics...")
    
    # Calculate grid dimensions
    n_metrics = len(available_metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 4 * n_rows))
    
    # Create subplots
    for idx, metric in enumerate(available_metrics):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        create_single_metric_subplot(df, metric, ax, config_order)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), n_rows * n_cols):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.set_visible(False)
    
    # Add main title with experiment name
    title_text = f'{model_name} - {experiment_name.replace("_", " ").title()} Experiment'
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save with descriptive filename
    output_path = output_dir / f'{csv_path.stem}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path.name}")
    print(f"   Location: {output_dir}/")
    
    return output_dir




def plot_aggregate_across_models(results_base_dir: Path, output_subdir: str = 'plots',
                                  metrics: Optional[List[str]] = None,
                                  experiment_pattern: str = '*_specification_*.csv'):
    """
    Generate aggregate boxplots combining data from all models.
    
    Args:
        results_base_dir: Base directory containing model subdirectories
        output_subdir: Subdirectory name for plots (default: 'plots')
        metrics: Optional list of specific metrics to plot
        experiment_pattern: Glob pattern to match CSV files
    """
    print("\n" + "="*80)
    print("📊 GENERATING AGGREGATE VISUALIZATION ACROSS ALL MODELS")
    print("="*80)
    
    # Find all model directories
    model_dirs = [d for d in results_base_dir.iterdir() 
                 if d.is_dir() and d.name in ['Qwen2', 'Qwen3', 'SmolLM', 'TinyLlama', 'Phi3']]
    
    if not model_dirs:
        print("❌ No model directories found")
        return None
    
    print(f"\n📁 Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")
    
    # Collect all data
    all_data = []
    for model_dir in model_dirs:
        # Find the most recent specification CSV for this model
        csv_files = sorted(model_dir.glob(experiment_pattern), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not csv_files:
            print(f"⚠️  No CSV files found in {model_dir.name}/")
            continue
        
        # Use the most recent file
        csv_file = csv_files[0]
        print(f"   Loading {model_dir.name}: {csv_file.name}")
        
        try:
            df = load_results(csv_file)
            df['model'] = model_dir.name  # Add model identifier
            df['config_label'] = df.apply(create_config_label, axis=1)
            all_data.append(df)
            print(f"      ✅ Loaded {len(df)} rows")
        except Exception as e:
            print(f"      ❌ Error loading: {e}")
    
    if not all_data:
        print("❌ No data loaded from any model")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Combined dataset: {len(combined_df)} total rows from {len(all_data)} models")
    
    # Configuration order
    config_order = ['Control', 'Weighting', 'Prompting', 'Both']
    config_order = [c for c in config_order if c in combined_df['config_label'].values]
    
    # Define metrics
    if metrics is None:
        metrics = [
            'flesch_kincaid_grade',
            'gunning_fog',
            'smog_index',
            'spache_readability',
            'word_count',
            'difficult_words'
        ]
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in combined_df.columns]
    print(f"\n🎨 Generating aggregate plots for {len(available_metrics)} metrics...")
    
    # Calculate grid dimensions
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(14, 9 * n_rows))
    
    # Define target values for A1 learners
    metric_targets = {
        'flesch_kincaid_grade': 5.0,
        'gunning_fog': 6.0,
        'smog_index': 7.0,
        'spache_readability': 4.0
    }
    
    # Create subplots
    for idx, metric in enumerate(available_metrics):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        if metric not in combined_df.columns:
            ax.set_visible(False)
            continue
        
        # Create boxplot
        sns.boxplot(
            data=combined_df,
            x='config_label',
            y=metric,
            hue='config_label',
            order=config_order,
            palette='Set2',
            legend=False,
            ax=ax
        )
        
        # Add individual points
        sns.stripplot(
            data=combined_df,
            x='config_label',
            y=metric,
            order=config_order,
            color='black',
            alpha=0.2,
            size=2,
            ax=ax
        )
        
        # Add target line if applicable
        if metric in metric_targets:
            target_value = metric_targets[metric]
            ax.axhline(y=target_value, color='red', linestyle='--', linewidth=4, 
                      label=f'Target: {target_value}', zorder=10)
        
        # Formatting
        ax.set_title(metric.replace('_', ' ').title(), fontsize=32, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=24)
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample size annotation
        n_per_config = combined_df.groupby('config_label').size()
        ax.text(0.02, 0.98, f"n={n_per_config[config_order[0]]}/config", 
                transform=ax.transAxes, fontsize=20, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Hide unused subplots
    for idx in range(len(available_metrics), n_rows * n_cols):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory
    output_dir = results_base_dir / 'aggregate' / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    timestamp = pd.Timestamp.now().strftime('%m%d_%H%M')
    output_path = output_dir / f'aggregate_all_models_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved aggregate visualization: {output_path.name}")
    print(f"   Location: {output_dir}/")
    
    return output_path


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description='Visualize factorial experiment results with boxplots'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        help='Path to CSV file or model name (if not provided, will process all CSVs in results/)'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        help='Specific metrics to plot (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        default='plots',
        help='Output directory name for plots (default: plots)'
    )
    parser.add_argument(
        '--model',
        choices=['Qwen2', 'Qwen3', 'SmolLM', 'TinyLlama', 'Phi3', 'all'],
        help='Process all CSVs for specific model'
    )
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Generate aggregate visualization combining all models'
    )
    
    args = parser.parse_args()
    
    results_base_dir = Path(__file__).parent
    
    # Handle aggregate flag
    if args.aggregate:
        plot_aggregate_across_models(results_base_dir, args.output_dir, args.metrics)
        return
    
    if args.csv_file:
        # Check if it's a direct file path
        csv_path = Path(args.csv_file)
        if csv_path.exists():
            plot_all_metrics(str(csv_path), args.output_dir, args.metrics)
        # Check if it's a model name
        elif (results_base_dir / args.csv_file).exists():
            model_dir = results_base_dir / args.csv_file
            csv_files = list(model_dir.glob('*_specification_*.csv'))
            if not csv_files:
                print(f"❌ No specification CSV files found in {args.csv_file}/")
                return
            print(f"📊 Processing {len(csv_files)} files for {args.csv_file}")
            for csv_file in csv_files:
                plot_all_metrics(str(csv_file), args.output_dir, args.metrics)
        else:
            print(f"❌ File or model directory not found: {args.csv_file}")
            return
    elif args.model:
        # Process specific model or all models
        if args.model == 'all':
            model_dirs = [d for d in results_base_dir.iterdir() 
                         if d.is_dir() and d.name in ['Qwen2', 'Qwen3', 'SmolLM', 'TinyLlama', 'Phi3']]
        else:
            model_dirs = [results_base_dir / args.model]
        
        if not model_dirs:
            print("❌ No model directories found")
            return
        
        total_processed = 0
        for model_dir in model_dirs:
            csv_files = list(model_dir.glob('*_specification_*.csv'))
            if csv_files:
                print(f"\n📊 Processing {model_dir.name}: {len(csv_files)} files")
                for csv_file in csv_files:
                    try:
                        plot_all_metrics(str(csv_file), args.output_dir, args.metrics)
                        total_processed += 1
                    except Exception as e:
                        print(f"❌ Error processing {csv_file.name}: {e}")
        
        print(f"\n✅ Processed {total_processed} visualizations!")
    else:
        # Process all specification CSVs in all model directories
        model_dirs = [d for d in results_base_dir.iterdir() 
                     if d.is_dir() and d.name in ['Qwen2', 'Qwen3', 'SmolLM', 'TinyLlama', 'Phi3']]
        
        if not model_dirs:
            print("❌ No model directories found in results/")
            return
        
        total_processed = 0
        for model_dir in model_dirs:
            csv_files = list(model_dir.glob('*_specification_*.csv'))
            if csv_files:
                print(f"\n📊 Processing {model_dir.name}: {len(csv_files)} files")
                for csv_file in csv_files:
                    try:
                        plot_all_metrics(str(csv_file), args.output_dir, args.metrics)
                        total_processed += 1
                    except Exception as e:
                        print(f"❌ Error processing {csv_file.name}: {e}")
                        import traceback
                        traceback.print_exc()
        
        print(f"\n✅ All visualizations complete! Processed {total_processed} files.")


if __name__ == "__main__":
    main()

