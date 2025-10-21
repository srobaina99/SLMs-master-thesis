import textstat
from typing import Dict, Any
import json


class TextEvaluator:
    """
    A comprehensive text readability and complexity evaluator using textstat library.
    
    This class provides methods to calculate various readability metrics and grade levels
    for English text analysis, including all major readability indices.
    """
    
    def __init__(self):
        """Initialize the TextEvaluator."""
        pass
    
    def get_grade_level_indices(self, text: str) -> Dict[str, float]:
        """
        Calculate primary grade level indices for the given text.
        
        Includes only non-redundant metrics selected for A1 learner assessment:
        - Flesch-Kincaid Grade Level: Sentence structure & syllabic complexity
        - Gunning Fog Index: Polysyllabic word emphasis
        - SMOG Index: Polysyllable density, reliable for short texts
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing primary grade level metrics
        """
        if not text or not text.strip():
            return {
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0,
                'smog_index': 0.0
            }
        
        return {
            'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(text), 2),
            'gunning_fog': round(textstat.gunning_fog(text), 2),
            'smog_index': round(textstat.smog_index(text), 2)
        }
    
    def get_readability_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate primary readability scores for the given text.
        
        Includes only Spache Readability, which is specifically designed for
        primary-grade materials (grades 1-4) and provides superior discrimination
        at A1 proficiency level compared to broader word-list metrics.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing primary readability scores
        """
        if not text or not text.strip():
            return {
                'spache_readability': 0.0
            }
        
        return {
            'spache_readability': round(textstat.spache_readability(text), 2)
        }

    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Get secondary descriptive statistics for the given text.
        
        Includes only non-redundant statistics selected for A1 learner assessment:
        - Word Count: Reveals verbosity/conciseness differences
        - Difficult Words: Direct, interpretable vocabulary accessibility metric
        
        Note: Other statistics (sentence count, syllable counts, etc.) are calculated
        internally by primary metrics and don't provide independent insights.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, int]: Dictionary containing secondary descriptive statistics
        """
        if not text or not text.strip():
            return {
                'word_count': 0,
                'difficult_words': 0
            }
        
        return {
            'word_count': textstat.lexicon_count(text),
            'difficult_words': textstat.difficult_words(text)
        }
    
    
    def evaluate_text_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive text evaluation with selected metrics for A1 learner assessment.
        
        Returns 4 primary metrics + 2 secondary statistics, eliminating redundancy:
        
        Primary Metrics (4):
        - Flesch-Kincaid Grade Level
        - Gunning Fog Index
        - SMOG Index
        - Spache Readability
        
        Secondary Statistics (2):
        - Word Count
        - Difficult Words Count
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Complete analysis results with primary and secondary metrics
        """
        if not text or not text.strip():
            return {
                'text_statistics': self.get_text_statistics(text),
                'grade_level_indices': self.get_grade_level_indices(text),
                'readability_scores': self.get_readability_scores(text)
            }
        
        analysis = {
            'text_statistics': self.get_text_statistics(text),
            'grade_level_indices': self.get_grade_level_indices(text),
            'readability_scores': self.get_readability_scores(text)
        }
        
        return analysis
    
    def print_analysis(self, text: str) -> None:
        """
        Print a formatted analysis of the text.
        
        Args:
            text (str): Text to analyze
        """
        analysis = self.evaluate_text_comprehensive(text)
        
        print("=" * 60)
        print("TEXT READABILITY ANALYSIS (A1 Learner Focus)")
        print("=" * 60)
        
        # Primary Metrics - Grade Level Indices
        print("\n🎓 PRIMARY METRICS - GRADE LEVEL INDICES:")
        grades = analysis['grade_level_indices']
        print(f"  • Flesch-Kincaid Grade Level: {grades['flesch_kincaid_grade']} (target: ≤5.0)")
        print(f"  • Gunning Fog Index: {grades['gunning_fog']} (target: ≤6.0)")
        print(f"  • SMOG Index: {grades['smog_index']} (target: ≤7.0)")
        
        # Primary Metrics - Readability Scores
        print("\n📖 PRIMARY METRICS - READABILITY SCORES:")
        scores = analysis['readability_scores']
        print(f"  • Spache Readability: {scores['spache_readability']} (target: ≤4.0, primary grades)")
        
        # Secondary Statistics
        print("\n📊 SECONDARY DESCRIPTIVE STATISTICS:")
        stats = analysis['text_statistics']
        print(f"  • Word Count: {stats['word_count']} (target: 30-60 words)")
        print(f"  • Difficult Words: {stats['difficult_words']} (target: minimize)")
        
        print("\n" + "=" * 60)
        print("Note: Analysis includes only non-redundant metrics selected")
        print("for A1 English learner assessment. See docs/text_metrics.md")
        print("for detailed rationale on metric selection.")
        print("=" * 60)
    
    def save_analysis_to_json(self, text: str, filename: str) -> None:
        """
        Save comprehensive analysis to a JSON file.
        
        Args:
            text (str): Text to analyze
            filename (str): Output filename
        """
        analysis = self.evaluate_text_comprehensive(text)
        
        # Add metadata
        analysis['metadata'] = {
            'text_preview': text[:100] + "..." if len(text) > 100 else text,
            'analysis_type': 'comprehensive_readability'
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis saved to {filename}")


# Example usage and testing
if __name__ == "__main__":
    # Create evaluator instance
    evaluator = TextEvaluator()
    
    # Sample English text
    english_text = """
    The quick brown fox jumps over the lazy dog. This is a simple sentence that contains 
    every letter of the alphabet. Artificial intelligence is revolutionizing multiple 
    industries through sophisticated algorithms and machine learning techniques. 
    These computational tools enable automation of complex processes and decision-making 
    systems that were previously impossible to implement efficiently.
    """
    
    print("ENGLISH TEXT ANALYSIS:")
    evaluator.print_analysis(english_text)
    
    # Save analysis to JSON
    evaluator.save_analysis_to_json(english_text, "english_analysis.json")
