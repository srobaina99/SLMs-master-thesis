#!/usr/bin/env python3
"""
Visualize multi-weight experiment results.
Creates boxplots showing complexity metrics by weight factor for each model.
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

def plot_model_metrics(df, model_name, output_dir):
    """Create boxplots for one model showing metrics by weight."""
    
    model_data = df[df['model'] == model_name].copy()
    
    if len(model_data) == 0:
        print(f"⚠️  No data for {model_name}")
        return
    
    # Filter out empty responses (zero metrics)
    model_data = model_data[model_data['word_count'] > 0]
    
    if len(model_data) == 0:
        print(f"⚠️  No valid responses for {model_name}")
        return
    
    # Primary metrics from docs/text_metrics.md
    metrics = [
        ('flesch_kincaid_grade', 'Flesch-Kincaid Grade Level', 5.0),
        ('gunning_fog', 'Gunning Fog Index', 6.0),
        ('smog_index', 'SMOG Index', 7.0),
        ('spache_readability', 'Spache Readability', 4.0),
        ('word_count', 'Word Count', 60),
        ('difficult_words', 'Difficult Words', None)
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Sort by weight factor for consistent ordering
    weight_order = sorted(model_data['weight_factor'].unique())
    
    for idx, (metric, title, target) in enumerate(metrics):
        ax = axes[idx]
        
        # Create boxplot
        sns.boxplot(
            data=model_data,
            x='weight_factor',
            y=metric,
            order=weight_order,
            palette='Set2',
            ax=ax
        )
        
        # Add individual points
        sns.stripplot(
            data=model_data,
            x='weight_factor',
            y=metric,
            order=weight_order,
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
        ax.set_xlabel('Weight Factor', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{model_name} - Text Complexity by Weight Factor\n(Lower is Simpler)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_path = output_dir / f'{model_name}_by_weight.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path.name}")
    
    # Print summary statistics
    print(f"\n{model_name} Summary:")
    for weight in weight_order:
        weight_data = model_data[model_data['weight_factor'] == weight]
        print(f"  Weight {weight}:")
        print(f"    FK Grade: {weight_data['flesch_kincaid_grade'].mean():.2f} ± {weight_data['flesch_kincaid_grade'].std():.2f}")
        print(f"    Gunning Fog: {weight_data['gunning_fog'].mean():.2f} ± {weight_data['gunning_fog'].std():.2f}")
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
    print("MULTI-WEIGHT EXPERIMENT VISUALIZATION")
    print("=" * 70)
    print(f"\n📊 Loading data from: {csv_path.name}")
    
    # Load data
    df = load_data(csv_path)
    print(f"   Total rows: {len(df)}")
    
    # Create output directory
    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Get models (excluding TinyStories which has no valid responses)
    models = [m for m in df['model'].unique() if m != 'TinyStories']
    print(f"\n🤖 Models to visualize: {', '.join(models)}")
    
    # Generate plots for each model
    for model in models:
        print(f"\n{'='*70}")
        plot_model_metrics(df, model, output_dir)
    
    print(f"\n{'='*70}")
    print("✅ All visualizations complete!")
    print(f"📁 Saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())








