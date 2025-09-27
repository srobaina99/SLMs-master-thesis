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
        Calculate all grade level indices for the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing all grade level metrics
        """
        if not text or not text.strip():
            return {
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0,
                'smog_index': 0.0,
                'automated_readability_index': 0.0,
                'coleman_liau_index': 0.0,
                'dale_chall_readability_score': 0.0
            }
        
        return {
            'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(text), 2),
            'gunning_fog': round(textstat.gunning_fog(text), 2),
            'smog_index': round(textstat.smog_index(text), 2),
            'automated_readability_index': round(textstat.automated_readability_index(text), 2),
            'coleman_liau_index': round(textstat.coleman_liau_index(text), 2),
            'dale_chall_readability_score': round(textstat.dale_chall_readability_score(text), 2)
        }
    
    def get_readability_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate all readability scores for the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing all readability scores
        """
        if not text or not text.strip():
            return {
                'flesch_reading_ease': 0.0,
                'linsear_write_formula': 0.0,
                'spache_readability': 0.0,
                'mcalpine_eflaw': 0.0
            }
        
        return {
            'flesch_reading_ease': round(textstat.flesch_reading_ease(text), 2),
            'linsear_write_formula': round(textstat.linsear_write_formula(text), 2),
            'spache_readability': round(textstat.spache_readability(text), 2),
            'mcalpine_eflaw': round(textstat.mcalpine_eflaw(text), 2)
        }

    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Get basic text statistics.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, int]: Dictionary containing text statistics
        """
        if not text or not text.strip():
            return {
                'sentence_count': 0,
                'word_count': 0,
                'character_count': 0,
                'syllable_count': 0,
                'polysyllable_count': 0,
                'monosyllable_count': 0,
                'difficult_words': 0
            }
        
        return {
            'sentence_count': textstat.sentence_count(text),
            'word_count': textstat.lexicon_count(text),
            'character_count': textstat.char_count(text),
            'syllable_count': textstat.syllable_count(text),
            'polysyllable_count': textstat.polysyllabcount(text),
            'monosyllable_count': textstat.monosyllabcount(text),
            'difficult_words': textstat.difficult_words(text)
        }
    
    def get_reading_time(self, text: str) -> Dict[str, float]:
        """
        Estimate reading time for the text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing reading time estimates
        """
        if not text or not text.strip():
            return {
                'reading_time_seconds': 0.0,
                'reading_time_minutes': 0.0
            }
        
        reading_time_sec = textstat.reading_time(text)
        
        return {
            'reading_time_seconds': round(reading_time_sec, 2),
            'reading_time_minutes': round(reading_time_sec / 60, 2)
        }
    
    def evaluate_text_comprehensive(self, text: str, include_spanish: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive text evaluation with all available metrics.
        
        Args:
            text (str): Text to analyze
            include_spanish (bool): Whether to include Spanish-specific metrics
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        if not text or not text.strip():
            return {
                'text_statistics': self.get_text_statistics(text),
                'grade_level_indices': self.get_grade_level_indices(text),
                'readability_scores': self.get_readability_scores(text),
                'reading_time': self.get_reading_time(text),
            }
        
        analysis = {
            'text_statistics': self.get_text_statistics(text),
            'grade_level_indices': self.get_grade_level_indices(text),
            'readability_scores': self.get_readability_scores(text),
            'reading_time': self.get_reading_time(text)
        }
        
        return analysis
    
    def print_analysis(self, text: str, include_spanish: bool = False) -> None:
        """
        Print a formatted analysis of the text.
        
        Args:
            text (str): Text to analyze
            include_spanish (bool): Whether to include Spanish-specific metrics
        """
        analysis = self.evaluate_text_comprehensive(text, include_spanish)
        
        print("=" * 60)
        print("COMPREHENSIVE TEXT READABILITY ANALYSIS")
        print("=" * 60)
        
        # Text Statistics
        print("\n📊 TEXT STATISTICS:")
        stats = analysis['text_statistics']
        print(f"  • Sentences: {stats['sentence_count']}")
        print(f"  • Words: {stats['word_count']}")
        print(f"  • Characters: {stats['character_count']}")
        print(f"  • Syllables: {stats['syllable_count']}")
        print(f"  • Polysyllable words (3+ syllables): {stats['polysyllable_count']}")
        print(f"  • Monosyllable words: {stats['monosyllable_count']}")
        print(f"  • Difficult words: {stats['difficult_words']}")
        
        # Grade Level Indices
        print("\n🎓 GRADE LEVEL INDICES:")
        grades = analysis['grade_level_indices']
        print(f"  • Flesch-Kincaid Grade Level: {grades['flesch_kincaid_grade']}")
        print(f"  • Gunning Fog Index: {grades['gunning_fog']}")
        print(f"  • SMOG Index: {grades['smog_index']}")
        print(f"  • Automated Readability Index: {grades['automated_readability_index']}")
        print(f"  • Coleman-Liau Index: {grades['coleman_liau_index']}")
        print(f"  • Dale-Chall Readability Score: {grades['dale_chall_readability_score']}")
        
        # Readability Scores
        print("\n📖 READABILITY SCORES:")
        scores = analysis['readability_scores']
        print(f"  • Flesch Reading Ease: {scores['flesch_reading_ease']} (0-100, higher = easier)")
        print(f"  • Linsear Write Formula: {scores['linsear_write_formula']}")
        print(f"  • Spache Readability: {scores['spache_readability']}")
        print(f"  • McAlpine EFLAW: {scores['mcalpine_eflaw']}")
        
        # Spanish Scores (if requested)
        if include_spanish and analysis.get('spanish_scores'):
            print("\n🇪🇸 SPANISH READABILITY SCORES:")
            spanish = analysis['spanish_scores']
            print(f"  • Fernández-Huerta: {spanish['fernandez_huerta']}")
            print(f"  • Szigriszt-Pazos: {spanish['szigriszt_pazos']}")
            print(f"  • Gutiérrez de Polini: {spanish['gutierrez_polini']}")
            print(f"  • Crawford: {spanish['crawford']}")
        
        # Reading Time
        print("\n⏱️  READING TIME:")
        time_info = analysis['reading_time']
        print(f"  • Estimated reading time: {time_info['reading_time_minutes']} minutes")
        print(f"  • ({time_info['reading_time_seconds']} seconds)")
        
        print("\n" + "=" * 60)
    
    def save_analysis_to_json(self, text: str, filename: str, include_spanish: bool = False) -> None:
        """
        Save comprehensive analysis to a JSON file.
        
        Args:
            text (str): Text to analyze
            filename (str): Output filename
            include_spanish (bool): Whether to include Spanish-specific metrics
        """
        analysis = self.evaluate_text_comprehensive(text, include_spanish)
        
        # Add metadata
        analysis['metadata'] = {
            'text_preview': text[:100] + "..." if len(text) > 100 else text,
            'analysis_type': 'comprehensive_readability',
            'spanish_included': include_spanish
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
    
    # Sample Spanish text
    spanish_text = """
    La inteligencia artificial está transformando múltiples industrias a través de 
    algoritmos sofisticados y técnicas de aprendizaje automático. Estas herramientas 
    computacionales permiten la automatización de procesos complejos y sistemas de 
    toma de decisiones que anteriormente eran imposibles de implementar eficientemente.
    """
    
    print("ENGLISH TEXT ANALYSIS:")
    evaluator.print_analysis(english_text)
    
    print("\n\nSPANISH TEXT ANALYSIS:")
    evaluator.print_analysis(spanish_text, include_spanish=True)
    
    # Save analysis to JSON
    evaluator.save_analysis_to_json(english_text, "english_analysis.json")
    evaluator.save_analysis_to_json(spanish_text, "spanish_analysis.json", include_spanish=True)
