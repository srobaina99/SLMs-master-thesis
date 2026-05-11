#!/usr/bin/env python3
"""
Verify all numerical values in slm_complexity_control.tex against 
experimental data from October 23, 2024 (date 1023).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class PaperVerifier:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.models = ['Phi3', 'Qwen2', 'Qwen3', 'TinyLlama']
        self.data = {}
        self.discrepancies = []
        
    def load_data(self):
        """Load all experimental data from JSON files."""
        print(f"{BLUE}Loading experimental data from 1023...{RESET}\n")
        
        for model in self.models:
            json_file = self.results_dir / model / f"{model}_full_experiment_summary_1023_0034.json" if model == "Phi3" else \
                       self.results_dir / model / f"{model}_full_experiment_summary_1023_0025.json" if model == "Qwen2" else \
                       self.results_dir / model / f"{model}_full_experiment_summary_1023_0027.json" if model == "Qwen3" else \
                       self.results_dir / model / f"{model}_full_experiment_summary_1023_0029.json"
            
            with open(json_file, 'r') as f:
                self.data[model] = json.load(f)
            print(f"  ✓ Loaded {model}")
        
        print()
    
    def calculate_aggregate_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics across all 4 models."""
        configs = ['control', 'weighting_only', 'prompting_only', 'both']
        metrics = ['flesch_kincaid_grade', 'gunning_fog', 'smog_index', 
                   'spache_readability', 'word_count', 'difficult_words']
        
        aggregate = {}
        
        for config in configs:
            aggregate[config] = {}
            for metric in metrics:
                values = []
                for model in self.models:
                    val = self.data[model]['by_config'][config][metric]['mean']
                    values.append(val)
                
                # Calculate mean across all 4 models (n=60 per config: 15 prompts × 4 models)
                aggregate[config][metric] = sum(values) / len(values)
        
        return aggregate
    
    def report_discrepancy(self, section: str, item: str, paper_value: Any, 
                          actual_value: Any, tolerance: float = 0.1):
        """Report a discrepancy between paper and actual values."""
        try:
            diff = abs(float(paper_value) - float(actual_value))
            if diff > tolerance:
                self.discrepancies.append({
                    'section': section,
                    'item': item,
                    'paper': paper_value,
                    'actual': actual_value,
                    'diff': diff
                })
                print(f"  {RED}✗ MISMATCH{RESET}: {item}")
                print(f"    Paper:  {paper_value}")
                print(f"    Actual: {actual_value:.2f}")
                print(f"    Diff:   {diff:.2f}\n")
            else:
                print(f"  {GREEN}✓{RESET} {item}: {paper_value} (actual: {actual_value:.2f})")
        except (ValueError, TypeError):
            print(f"  {YELLOW}⚠{RESET} Could not compare {item}: {paper_value} vs {actual_value}")
    
    def verify_aggregate_tables(self):
        """Verify Tables 1a and 1b (aggregate statistics)."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}SECTION 1: Aggregate Tables (Lines 266-298){RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
        agg = self.calculate_aggregate_stats()
        
        # Table 1a: Primary readability metrics
        print(f"{YELLOW}Table 1a - Primary Readability Metrics:{RESET}\n")
        
        paper_values = {
            'control': {'FK': 8.95, 'GF': 11.06, 'SMOG': 11.37, 'Spa': 5.08},
            'weighting_only': {'FK': 9.11, 'GF': 11.05, 'SMOG': 11.04, 'Spa': 5.13},
            'prompting_only': {'FK': 5.60, 'GF': 6.99, 'SMOG': 8.34, 'Spa': 3.60},
            'both': {'FK': 5.59, 'GF': 7.29, 'SMOG': 8.00, 'Spa': 3.68}
        }
        
        config_labels = {
            'control': 'Control',
            'weighting_only': 'Weighting',
            'prompting_only': 'Prompting',
            'both': 'Both'
        }
        
        for config, label in config_labels.items():
            print(f"  {label}:")
            self.report_discrepancy('Table 1a', f'{label} FK', 
                                   paper_values[config]['FK'],
                                   agg[config]['flesch_kincaid_grade'])
            self.report_discrepancy('Table 1a', f'{label} GF', 
                                   paper_values[config]['GF'],
                                   agg[config]['gunning_fog'])
            self.report_discrepancy('Table 1a', f'{label} SMOG', 
                                   paper_values[config]['SMOG'],
                                   agg[config]['smog_index'])
            self.report_discrepancy('Table 1a', f'{label} Spa', 
                                   paper_values[config]['Spa'],
                                   agg[config]['spache_readability'])
            print()
        
        # Table 1b: Secondary metrics
        print(f"{YELLOW}Table 1b - Secondary Metrics:{RESET}\n")
        
        paper_values_1b = {
            'control': {'Words': 82.3, 'Diff': 15.7},
            'weighting_only': {'Words': 90.6, 'Diff': 15.5},
            'prompting_only': {'Words': 69.2, 'Diff': 6.2},
            'both': {'Words': 73.1, 'Diff': 5.8}
        }
        
        for config, label in config_labels.items():
            print(f"  {label}:")
            self.report_discrepancy('Table 1b', f'{label} Words', 
                                   paper_values_1b[config]['Words'],
                                   agg[config]['word_count'], tolerance=1.0)
            self.report_discrepancy('Table 1b', f'{label} Diff', 
                                   paper_values_1b[config]['Diff'],
                                   agg[config]['difficult_words'])
            print()
    
    def verify_model_specific_table(self):
        """Verify Table 2 (model-specific marginal effects)."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}SECTION 2: Model-Specific Table 2 (Lines 316-354){RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
        paper_deltas = {
            'Phi3': {
                'FK': (-0.35, -8.3),
                'GF': (0.09, 1.6),
                'SMOG': (-0.64, -8.5),
                'Spa': (-0.02, -0.7),
                'Words': (-1.5, -2.2),
                'Diff': (-0.3, -9.2)
            },
            'Qwen2': {
                'FK': (2.01, 33.5),
                'GF': (2.34, 30.2),
                'SMOG': (0.08, 0.9),
                'Spa': (0.73, 19.2),
                'Words': (7.5, 10.9),
                'Diff': (-1.0, -17.2)
            },
            'Qwen3': {
                'FK': (-0.29, -7.7),
                'GF': (-0.10, -2.1),
                'SMOG': (-0.61, -8.4),
                'Spa': (0.11, 3.9),
                'Words': (-8.7, -14.7),
                'Diff': (-0.5, -9.4)
            },
            'TinyLlama': {
                'FK': (-1.44, -17.2),
                'GF': (-1.13, -11.4),
                'SMOG': (-0.20, -2.0),
                'Spa': (-0.50, -10.5),
                'Words': (18.4, 22.7),
                'Diff': (0.0, 0.0)
            }
        }
        
        metric_map = {
            'FK': 'flesch_kincaid_grade',
            'GF': 'gunning_fog',
            'SMOG': 'smog_index',
            'Spa': 'spache_readability',
            'Words': 'word_count',
            'Diff': 'difficult_words'
        }
        
        for model in self.models:
            print(f"{YELLOW}{model} Marginal Effects (Both - Prompting Only):{RESET}\n")
            
            prompting = self.data[model]['by_config']['prompting_only']
            both = self.data[model]['by_config']['both']
            
            for metric_short, metric_long in metric_map.items():
                prompting_val = prompting[metric_long]['mean']
                both_val = both[metric_long]['mean']
                
                actual_delta = both_val - prompting_val
                actual_pct = (actual_delta / prompting_val * 100) if prompting_val != 0 else 0
                
                paper_delta, paper_pct = paper_deltas[model][metric_short]
                
                print(f"  {metric_short}:")
                self.report_discrepancy(f'Table 2 {model}', f'{metric_short} delta', 
                                       paper_delta, actual_delta, tolerance=0.15)
                self.report_discrepancy(f'Table 2 {model}', f'{metric_short} %', 
                                       paper_pct, actual_pct, tolerance=1.5)
            print()
    
    def verify_abstract_claims(self):
        """Verify claims made in the abstract."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}SECTION 3: Abstract Claims (Lines 24-25){RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
        # Check FK Grade targets for Phi3 and Qwen3
        print(f"{YELLOW}A1 Target Achievement (FK Grade ≤ 5.0):{RESET}\n")
        
        for model in ['Phi3', 'Qwen3']:
            prompting_fk = self.data[model]['by_config']['prompting_only']['flesch_kincaid_grade']['mean']
            both_fk = self.data[model]['by_config']['both']['flesch_kincaid_grade']['mean']
            
            print(f"  {model}:")
            print(f"    Prompting Only: {prompting_fk:.2f} {'✓' if prompting_fk <= 5.0 else '✗'}")
            print(f"    Both: {both_fk:.2f} {'✓' if both_fk <= 5.0 else '✗'}")
        
        print()
        
        # Model size comparison
        print(f"{YELLOW}Model Size Comparison:{RESET}\n")
        print(f"  Paper claim: Qwen3 (0.6B) is 6.3× smaller than Phi3 (3.8B)")
        actual_ratio = 3.8 / 0.6
        print(f"  Actual ratio: {actual_ratio:.1f}×")
        if abs(actual_ratio - 6.3) > 0.1:
            print(f"  {RED}✗ MISMATCH{RESET}: Difference of {abs(actual_ratio - 6.3):.1f}")
        else:
            print(f"  {GREEN}✓ CORRECT{RESET}")
        
        print()
        
        # Complexity reduction ranges
        print(f"{YELLOW}Complexity Reduction Claims:{RESET}\n")
        print(f"  Paper claim: '28-63%' reductions under prompting interventions")
        print(f"  Checking difficult words reduction (60-63% claim):")
        
        agg = self.calculate_aggregate_stats()
        control_diff = agg['control']['difficult_words']
        prompting_diff = agg['prompting_only']['difficult_words']
        both_diff = agg['both']['difficult_words']
        
        prompting_reduction = (control_diff - prompting_diff) / control_diff * 100
        both_reduction = (control_diff - both_diff) / control_diff * 100
        
        print(f"    Prompting Only: {prompting_reduction:.1f}% reduction")
        print(f"    Both: {both_reduction:.1f}% reduction")
        
        print()
    
    def verify_discussion_percentages(self):
        """Verify percentage claims in the discussion section."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}SECTION 4: Discussion Percentages{RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
        agg = self.calculate_aggregate_stats()
        
        # 10% word count increase under weighting
        print(f"{YELLOW}Word Count Changes:{RESET}\n")
        control_words = agg['control']['word_count']
        weighting_words = agg['weighting_only']['word_count']
        weighting_increase = (weighting_words - control_words) / control_words * 100
        
        print(f"  Paper claim: '10% aggregate word count increase' under weighting")
        print(f"  Actual: {weighting_increase:.1f}%")
        self.report_discrepancy('Discussion', 'Weighting word increase %', 
                               10.0, weighting_increase, tolerance=1.0)
        
        print()
        
        # Both vs Prompting comparison
        print(f"{YELLOW}Both vs Prompting Only:{RESET}\n")
        prompting_words = agg['prompting_only']['word_count']
        both_words = agg['both']['word_count']
        both_word_increase = (both_words - prompting_words) / prompting_words * 100
        
        print(f"  Paper claim: '5.7% more words' for Both vs Prompting")
        print(f"  Actual: {both_word_increase:.1f}%")
        self.report_discrepancy('Discussion', 'Both vs Prompting word increase %', 
                               5.7, both_word_increase, tolerance=1.0)
        
        prompting_diff = agg['prompting_only']['difficult_words']
        both_diff = agg['both']['difficult_words']
        both_diff_decrease = (prompting_diff - both_diff) / prompting_diff * 100
        
        print(f"  Paper claim: '6.7% fewer difficult words' for Both vs Prompting")
        print(f"  Actual: {both_diff_decrease:.1f}%")
        self.report_discrepancy('Discussion', 'Both vs Prompting diff word decrease %', 
                               6.7, both_diff_decrease, tolerance=1.0)
        
        print()
    
    def print_summary(self):
        """Print summary of verification results."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}VERIFICATION SUMMARY{RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
        if not self.discrepancies:
            print(f"{GREEN}✓ ALL VALUES VERIFIED SUCCESSFULLY!{RESET}")
            print(f"  All numerical values in the paper match the experimental data from 1023.\n")
        else:
            print(f"{RED}✗ FOUND {len(self.discrepancies)} DISCREPANCIES{RESET}\n")
            
            for i, disc in enumerate(self.discrepancies, 1):
                print(f"{i}. {disc['section']} - {disc['item']}")
                print(f"   Paper: {disc['paper']}, Actual: {disc['actual']:.2f}, Diff: {disc['diff']:.2f}")
            
            print()


def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "src/evaluation/experiment_framework/results"
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}PAPER VERIFICATION TOOL{RESET}")
    print(f"{BLUE}Comparing slm_complexity_control.tex against 1023 experimental data{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    verifier = PaperVerifier(results_dir)
    verifier.load_data()
    
    # Run all verification sections
    verifier.verify_aggregate_tables()
    verifier.verify_model_specific_table()
    verifier.verify_abstract_claims()
    verifier.verify_discussion_percentages()
    
    # Print summary
    verifier.print_summary()


if __name__ == "__main__":
    main()

