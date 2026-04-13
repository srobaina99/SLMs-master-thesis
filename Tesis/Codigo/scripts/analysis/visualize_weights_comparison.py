#!/usr/bin/env python3
"""
Visualize multi-weight experiment results side-by-side.
Creates single plot showing weight factor comparison across all models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")

def load_data(csv_path):
    """Load CSV with proper decimal handling."""
    df = pd.read_csv(csv_path, decimal=',')
    # Convert weight_factor from European to standard format
    df['weight_factor'] = df['weight_factor'].astype(str).str.replace(',', '.').astype(float)
    return df

def plot_weights_comparison(df, output_dir):
    """Create comparison plot showing all weights side-by-side."""
    
    # Filter out empty responses and TinyStories
    df = df[(df['word_count'] > 0) & (df['model'] != 'TinyStories')].copy()
    
    # Convert weight to string for better labeling
    df['weight_label'] = df['weight_factor'].apply(lambda x: f'Weight {x:.1f}')
    
    # Primary metrics from docs/text_metrics.md
    metrics = [
        ('flesch_kincaid_grade', 'Flesch-Kincaid Grade Level', 5.0),
        ('gunning_fog', 'Gunning Fog Index', 6.0),
        ('spache_readability', 'Spache Readability', 4.0),
        ('word_count', 'Word Count', 60),
        ('difficult_words', 'Difficult Words', None)
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Sort weights for consistent ordering
    weight_order = sorted(df['weight_label'].unique())
    
    for idx, (metric, title, target) in enumerate(metrics):
        ax = axes[idx]
        
        # Create boxplot with weights side by side
        sns.boxplot(
            data=df,
            x='weight_label',
            y=metric,
            order=weight_order,
            hue='weight_label',
            palette='Set2',
            legend=False,
            ax=ax
        )
        
        # Add individual points colored by model
        sns.stripplot(
            data=df,
            x='weight_label',
            y=metric,
            order=weight_order,
            hue='model',
            dodge=False,
            alpha=0.6,
            size=5,
            ax=ax,
            legend=False
        )
        
        # Add target line if applicable
        if target is not None:
            ax.axhline(y=target, color='red', linestyle='--', linewidth=2, 
                      label=f'A1 Target (≤{target})', alpha=0.7)
            ax.legend(loc='upper right', fontsize=10)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Weight Factor', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend for models
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[-5:], labels[-5:], loc='upper center', 
              bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=11, 
              title='Models', title_fontsize=12)
    
    plt.suptitle('Weight Factor Comparison Across All Models\n(Lower is Simpler)', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = output_dir / 'weight_comparison_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path.name}")
    
    # Print summary statistics
    print(f"\nComparative Summary by Weight:")
    for weight in sorted(df['weight_factor'].unique()):
        weight_data = df[df['weight_factor'] == weight]
        print(f"\n  Weight {weight}:")
        print(f"    Mean FK Grade: {weight_data['flesch_kincaid_grade'].mean():.2f} ± {weight_data['flesch_kincaid_grade'].std():.2f}")
        print(f"    Mean Gunning Fog: {weight_data['gunning_fog'].mean():.2f} ± {weight_data['gunning_fog'].std():.2f}")
        print(f"    Best model (FK): {weight_data.groupby('model')['flesch_kincaid_grade'].mean().idxmin()}")
        print(f"    Valid responses: {len(weight_data)}")


def main():
    """Main entry point."""
    
    # Default path
    results_dir = Path(__file__).parent.parent.parent / "results/multi"
    csv_path = results_dir / "multi_weight_experiment_specification_1007_0329.csv"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return 1
    
    print("=" * 70)
    print("WEIGHT FACTOR COMPARISON VISUALIZATION")
    print("=" * 70)
    print(f"\n📊 Loading data from: {csv_path.name}")
    
    # Load data
    df = load_data(csv_path)
    print(f"   Total rows: {len(df)}")
    
    # Create output directory
    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Generate comparison plot
    print(f"\n{'='*70}")
    plot_weights_comparison(df, output_dir)
    
    print(f"\n{'='*70}")
    print("✅ Visualization complete!")
    print(f"📁 Saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())








