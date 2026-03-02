"""
Script to run beam search experiment on Qwen3 model.

Tests two beam selection methods:
1. Highest A1 word ratio (with 1.5x weighting)
2. Highest cumulative log probability

Uses first 5 prompts and contextual prompting intervention.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.framework.experiments.factorial_experiment import FactorialExperiment
from src.framework.experiments.experiment_configs import STANDARD_PROMPTS
from datetime import datetime


def main():
    """Run beam search experiment."""
    
    print("\n" + "="*80)
    print("BEAM SEARCH EXPERIMENT - Qwen3 Model (Beam Width = 8)")
    print("="*80)
    
    # Create experiment runner
    experiment = FactorialExperiment(
        results_dir="results"
    )
    
    # Run beam search experiment with first 5 prompts
    test_prompts = STANDARD_PROMPTS[:5]
    
    print("\nTest prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  P{i}: {prompt}")
    
    df_beam_results = experiment.run_beam_search_experiment(
        prompts=test_prompts,
        beam_width=8,
        experiment_name=f"beam_search_qwen3_w8_{datetime.now().strftime('%m%d_%H%M')}"
    )
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    saved_files = experiment.save_results("Qwen3_beam_search")
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    if not df_beam_results.empty:
        print(f"\n✅ Generated {len(df_beam_results)} beam search results")
        
        # Group by selection method
        if 'beam_selection_method' in df_beam_results.columns:
            print("\nResults by selection method:")
            for method in df_beam_results['beam_selection_method'].unique():
                count = len(df_beam_results[df_beam_results['beam_selection_method'] == method])
                print(f"  • {method}: {count} results")
        
        # Show sample metrics
        if 'flesch_kincaid_grade' in df_beam_results.columns:
            print("\nFlesch-Kincaid Grade (mean ± std):")
            if 'beam_selection_method' in df_beam_results.columns:
                for method in df_beam_results['beam_selection_method'].unique():
                    subset = df_beam_results[df_beam_results['beam_selection_method'] == method]
                    fk_mean = subset['flesch_kincaid_grade'].mean()
                    fk_std = subset['flesch_kincaid_grade'].std()
                    print(f"  • {method}: {fk_mean:.2f} ± {fk_std:.2f}")
    
    print("\n" + "="*80)
    print("FILES SAVED")
    print("="*80)
    for file_type, path in saved_files.items():
        print(f"✅ {file_type}: {path}")
    
    print("\n" + "="*80)
    print("✅ BEAM SEARCH EXPERIMENT COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

