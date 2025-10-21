#!/usr/bin/env python3
"""
Recover weight_factor column in specification CSV from full_data CSV.
Regenerates specification CSV with weight_factor included.
"""

import pandas as pd
import sys
from pathlib import Path

def recover_specification_csv(full_csv_path: str, spec_csv_path: str, output_path: str = None):
    """
    Recover specification CSV with weight_factor column from full_data CSV.
    
    Args:
        full_csv_path: Path to full_data CSV (contains weight_factor)
        spec_csv_path: Path to specification CSV (missing weight_factor)
        output_path: Path to save recovered CSV (defaults to overwriting spec_csv_path)
    """
    # Read both CSVs
    print(f"📖 Reading full data from: {full_csv_path}")
    df_full = pd.read_csv(full_csv_path)
    
    print(f"📖 Reading specification from: {spec_csv_path}")
    df_spec = pd.read_csv(spec_csv_path, decimal=',')
    
    # Check that weight_factor exists in full data
    if 'weight_factor' not in df_full.columns:
        print("❌ ERROR: weight_factor not found in full_data CSV")
        return False
    
    print(f"✅ Full data has {len(df_full)} rows")
    print(f"✅ Specification has {len(df_spec)} rows")
    
    if len(df_full) != len(df_spec):
        print("⚠️  WARNING: Row counts differ between files")
    
    # Extract weight_factor from full data
    weight_factors = df_full['weight_factor'].values
    
    # Insert weight_factor column after config_prompting
    insert_position = list(df_spec.columns).index('prompt_id')
    df_spec.insert(insert_position, 'weight_factor', weight_factors)
    
    # Save recovered CSV
    if output_path is None:
        output_path = spec_csv_path
    
    print(f"💾 Saving recovered specification to: {output_path}")
    df_spec.to_csv(output_path, index=False, decimal=',')
    
    print(f"✅ SUCCESS: Added weight_factor column with {len(weight_factors)} values")
    print(f"   Unique weights: {sorted(df_spec['weight_factor'].unique())}")
    
    return True


def main():
    """Recover weight_factor for existing multi_weight experiment results."""
    
    # Default paths for the multi_weight experiment
    results_dir = Path(__file__).parent.parent / "src/evaluation/experiment_framework/results/multi"
    
    full_csv = results_dir / "full_data/multi_weight_experiment_full_1007_0329.csv"
    spec_csv = results_dir / "multi_weight_experiment_specification_1007_0329.csv"
    
    if not full_csv.exists():
        print(f"❌ ERROR: Full data CSV not found: {full_csv}")
        print("\nUsage: python recover_weight_factors.py [full_csv_path] [spec_csv_path] [output_path]")
        return 1
    
    if not spec_csv.exists():
        print(f"❌ ERROR: Specification CSV not found: {spec_csv}")
        return 1
    
    # Allow custom paths from command line
    if len(sys.argv) >= 3:
        full_csv = Path(sys.argv[1])
        spec_csv = Path(sys.argv[2])
        output_path = sys.argv[3] if len(sys.argv) >= 4 else None
    else:
        output_path = None
    
    print("=" * 70)
    print("WEIGHT FACTOR RECOVERY FOR SPECIFICATION CSV")
    print("=" * 70)
    
    success = recover_specification_csv(str(full_csv), str(spec_csv), output_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())







