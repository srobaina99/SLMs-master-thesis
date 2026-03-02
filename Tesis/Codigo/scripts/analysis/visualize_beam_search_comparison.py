"""
Create boxplot comparison of beam search methods vs baseline "Both" intervention.

Compares three conditions:
1. Baseline: Prompting + 1.5x weighting (greedy decoding)
2. Beam Search (A1 Ratio): Highest A1 word ratio selection
3. Beam Search (Max Prob): Highest cumulative log probability selection

Plots Flesch-Kincaid Grade, Gunning Fog, SMOG, and Spache metrics with A1 target thresholds.

A1 Target Thresholds:
- Flesch-Kincaid Grade: ≤5.0
- Gunning Fog Index: ≤6.0
- SMOG Index: ≤7.0
- Spache Readability: ≤4.0
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)


def load_results(beam_search_csv: str, baseline_csv: str = None) -> pd.DataFrame:
    """
    Load beam search results and optionally baseline results.
    
    Args:
        beam_search_csv: Path to beam search results CSV
        baseline_csv: Path to baseline "Both" intervention results CSV (optional)
        
    Returns:
        Combined DataFrame with results
    """
    # Load beam search results (try different decimal separators)
    try:
        df_beam = pd.read_csv(beam_search_csv, decimal=',')
    except:
        df_beam = pd.read_csv(beam_search_csv)
    
    # Add method label
    df_beam['comparison_method'] = df_beam['beam_selection_method'].apply(
        lambda x: 'Beam (A1 Ratio)' if x == 'a1_ratio' else 'Beam (Max Prob)'
    )
    
    results = [df_beam]
    
    # Load baseline if provided
    if baseline_csv and os.path.exists(baseline_csv):
        try:
            df_baseline = pd.read_csv(baseline_csv, decimal=',')
        except:
            df_baseline = pd.read_csv(baseline_csv)
        
        # Filter to "Both" intervention only and first 5 prompts (P1-P5)
        df_baseline_both = df_baseline[
            (df_baseline['config_weighting'].astype(bool)) & 
            (df_baseline['config_prompting'].astype(bool)) &
            (df_baseline['prompt_id'].isin(['P1', 'P2', 'P3', 'P4', 'P5']))
        ].copy()
        df_baseline_both['comparison_method'] = 'Baseline (Both)'
        
        # Select same columns for consistency
        results.append(df_baseline_both)
    
    # Combine results
    if len(results) > 1:
        df_combined = pd.concat(results, ignore_index=True)
    else:
        df_combined = results[0]
    
    return df_combined


def create_comparison_plots(df: pd.DataFrame, output_dir: str = None) -> None:
    """
    Create boxplot comparisons for readability metrics.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots (default: results dir)
    """
    if output_dir is None:
        output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Readability metrics to plot with A1 target thresholds
    metrics = {
        'flesch_kincaid_grade': ('Flesch-Kincaid Grade Level', 5.0),
        'gunning_fog': ('Gunning Fog Index', 6.0),
        'smog_index': ('SMOG Index', 7.0),
        'spache_readability': ('Spache Readability', 4.0)
    }
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beam Search vs Baseline Comparison\n(Qwen3 Model, First 5 Prompts)', 
                 fontsize=16, fontweight='bold')
    
    # Define method order and colors
    method_order = ['Baseline (Both)', 'Beam (A1 Ratio)', 'Beam (Max Prob)']
    colors = ['#95A5A6', '#2ECC71', '#3498DB']  # Gray, Green, Blue
    
    for idx, (metric_col, (metric_title, target_threshold)) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Create boxplot (suppress FutureWarning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sns.boxplot(
                data=df,
                x='comparison_method',
                y=metric_col,
                order=method_order,
                ax=ax,
                palette=colors,
                width=0.6
            )
        
        ax.set_title(metric_title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Method', fontsize=11)
        ax.set_ylabel('Score (lower is better)', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Reset y-axis formatting to automatic
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        
        # Add A1 target threshold line
        ax.axhline(y=target_threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'A1 Target (≤{target_threshold})', alpha=0.7)
        ax.legend(loc='upper right', fontsize=9)
        
        # Add sample size info at top
        for i, method in enumerate(method_order):
            count = len(df[df['comparison_method'] == method])
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                   ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_path = os.path.join(output_dir, f"beam_search_comparison_{timestamp}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved: {output_path}")
    
    plt.close()


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics for each method.
    
    Args:
        df: DataFrame with results
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY METHOD")
    print("="*80)
    
    metrics = ['flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'spache_readability', 
               'word_count']
    metric_names = {
        'flesch_kincaid_grade': 'FK Grade',
        'gunning_fog': 'Gunning Fog',
        'smog_index': 'SMOG',
        'spache_readability': 'Spache',
        'word_count': 'Words'
    }
    
    for method in sorted(df['comparison_method'].unique()):
        print(f"\n{method}:")
        subset = df[df['comparison_method'] == method]
        print(f"  Samples: {len(subset)}")
        
        for metric in metrics:
            if metric in subset.columns:
                values = pd.to_numeric(subset[metric], errors='coerce').dropna()
                if len(values) > 0:
                    mean = values.mean()
                    std = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    print(f"  {metric_names.get(metric, metric):12s}: {mean:6.2f} ± {std:5.2f} "
                          f"(range: {min_val:5.2f} - {max_val:5.2f})")


def main():
    """Run visualization script."""
    print("\n" + "="*80)
    print("BEAM SEARCH COMPARISON VISUALIZATION")
    print("="*80)
    
    # Find latest beam search results
    results_dir = "results/Qwen3"
    
    # Find the most recent beam search CSV (use full_data, not specification)
    import glob
    beam_csvs = glob.glob(os.path.join(results_dir, "full_data", "Qwen3_beam_search_full_*.csv"))
    
    if not beam_csvs:
        print("❌ No beam search results found in", results_dir)
        return
    
    # Use most recent
    beam_csv = sorted(beam_csvs)[-1]
    print(f"\nLoading beam search results: {beam_csv}")
    
    # Try to find baseline results (full experiment with "Both" intervention)
    baseline_csv = None
    full_experiment_results = glob.glob(os.path.join(results_dir, "*full_experiment_specification*.csv"))
    if full_experiment_results:
        baseline_csv = sorted(full_experiment_results)[-1]
        print(f"Loading baseline results: {baseline_csv}")
    
    # Load and combine results
    df = load_results(beam_csv, baseline_csv)
    
    print(f"\nTotal results loaded: {len(df)}")
    print(f"Methods: {df['comparison_method'].unique().tolist()}")
    
    # Create visualizations
    create_comparison_plots(df, results_dir)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

