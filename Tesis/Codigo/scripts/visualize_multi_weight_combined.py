#!/usr/bin/env python3
"""
Visualize multi-weight experiment results with models grouped by weight.
Creates boxplots showing all models for each weight factor.
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

def plot_by_weight(df, output_dir):
    """Create plots grouped by weight factor showing all models."""
    
    # Filter out empty responses and TinyStories
    df = df[(df['word_count'] > 0) & (df['model'] != 'TinyStories')].copy()
    
    # Primary metrics from docs/text_metrics.md
    metrics = [
        ('flesch_kincaid_grade', 'Flesch-Kincaid Grade Level', 5.0),
        ('gunning_fog', 'Gunning Fog Index', 6.0),
        ('smog_index', 'SMOG Index', 7.0),
        ('spache_readability', 'Spache Readability', 4.0),
        ('word_count', 'Word Count', 60),
        ('difficult_words', 'Difficult Words', None)
    ]
    
    weights = sorted(df['weight_factor'].unique())
    
    for weight in weights:
        weight_data = df[df['weight_factor'] == weight]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (metric, title, target) in enumerate(metrics):
            ax = axes[idx]
            
            # Create boxplot
            sns.boxplot(
                data=weight_data,
                x='model',
                y=metric,
                hue='model',
                palette='Set2',
                legend=False,
                ax=ax
            )
            
            # Add individual points
            sns.stripplot(
                data=weight_data,
                x='model',
                y=metric,
                color='black',
                alpha=0.4,
                size=4,
                ax=ax
            )
            
            # Add target line if applicable
            if target is not None:
                ax.axhline(y=target, color='red', linestyle='--', linewidth=2, 
                          label=f'A1 Target (≤{target})', alpha=0.7)
                ax.legend(loc='upper right', fontsize=9)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        weight_str = str(weight).replace('.', '_')
        plt.suptitle(f'All Models - Weight Factor {weight}\n(Lower is Simpler)', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save
        output_path = output_dir / f'weight_{weight_str}_all_models.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path.name}")
        
        # Print summary statistics
        print(f"\nWeight {weight} Summary:")
        for model in sorted(weight_data['model'].unique()):
            model_data = weight_data[weight_data['model'] == model]
            print(f"  {model}:")
            print(f"    FK Grade: {model_data['flesch_kincaid_grade'].mean():.2f} ± {model_data['flesch_kincaid_grade'].std():.2f}")
            print(f"    Gunning Fog: {model_data['gunning_fog'].mean():.2f} ± {model_data['gunning_fog'].std():.2f}")
            print(f"    Valid responses: {len(model_data)}")


def main():
    """Main entry point."""
    
    # Default path
    results_dir = Path(__file__).parent.parent / "src/evaluation/experiment_framework/results/multi"
    csv_path = results_dir / "multi_weight_experiment_specification_1007_0329.csv"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return 1
    
    print("=" * 70)
    print("MULTI-WEIGHT EXPERIMENT VISUALIZATION (BY WEIGHT)")
    print("=" * 70)
    print(f"\n📊 Loading data from: {csv_path.name}")
    
    # Load data
    df = load_data(csv_path)
    print(f"   Total rows: {len(df)}")
    
    # Create output directory
    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots grouped by weight
    print(f"\n{'='*70}")
    plot_by_weight(df, output_dir)
    
    print(f"\n{'='*70}")
    print("✅ All visualizations complete!")
    print(f"📁 Saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())







