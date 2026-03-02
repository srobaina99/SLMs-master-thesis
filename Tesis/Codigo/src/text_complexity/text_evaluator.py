import textstat
from typing import Dict, Any, Optional, Callable, Set, Tuple
import json

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Download required NLTK data on first import
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)


class TextEvaluator:
    """
    A comprehensive text readability and complexity evaluator using textstat library.
    
    This class provides methods to calculate various readability metrics and grade levels
    for English text analysis, including all major readability indices.
    """
    
    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None):
        """
        Initialize the TextEvaluator.
        
        Args:
            tokenizer: Optional tokenizer function that takes text and returns list of tokens.
                      If None, token counting will not be available.
        """
        self.tokenizer = tokenizer
        # Cache for POS tags to avoid redundant computation
        self._pos_cache = {}
    
    def extract_content_words(self, text: str) -> Set[str]:
        """
        Extract content words (nouns, verbs, adjectives, adverbs) from text.
        
        Uses NLTK POS tagging to identify content words and filters out function words.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of lowercase content words (without punctuation)
        """
        if not NLTK_AVAILABLE:
            # Fallback: use simple heuristic if NLTK unavailable
            return self._extract_content_words_fallback(text)
        
        try:
            # Check cache
            if text in self._pos_cache:
                return self._pos_cache[text]
            
            # Tokenize and tag
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Content POS tags: NN, NNS, NNP, NNPS (nouns), VB, VBD, VBG, VBN, VBP, VBZ (verbs),
            #                   JJ, JJR, JJS (adjectives), RB, RBR, RBS (adverbs)
            content_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                          'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
            
            content_words = set()
            for token, pos in pos_tags:
                if pos in content_pos:
                    # Remove punctuation
                    clean_token = ''.join(c for c in token if c.isalnum())
                    if clean_token:
                        content_words.add(clean_token)
            
            # Cache result
            self._pos_cache[text] = content_words
            return content_words
            
        except Exception as e:
            print(f"⚠️ POS tagging failed: {e}. Using fallback method.")
            return self._extract_content_words_fallback(text)
    
    def _extract_content_words_fallback(self, text: str) -> Set[str]:
        """
        Fallback method to extract content words without NLTK.
        
        Uses simple heuristic: words longer than 2 characters that aren't common function words.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of lowercase content words
        """
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'may', 'might', 'can', 'must', 'shall', 'it', 'its', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'our', 'their', 'if', 'then', 'so', 'what', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'no', 'not', 'only', 'just', 'very'
        }
        
        tokens = text.lower().split()
        content_words = set()
        
        for token in tokens:
            # Remove punctuation
            clean_token = ''.join(c for c in token if c.isalnum())
            if clean_token and len(clean_token) > 2 and clean_token not in function_words:
                content_words.add(clean_token)
        
        return content_words
    
    def calculate_a1_word_ratio(self, text: str, a1_vocab: Set[str]) -> Tuple[float, int, int]:
        """
        Calculate ratio of A1 words to content words.
        
        Args:
            text: Generated text
            a1_vocab: Set of A1 vocabulary words (lowercase)
            
        Returns:
            Tuple of (ratio, a1_count, content_count)
        """
        content_words = self.extract_content_words(text)
        
        if not content_words:
            return 0.0, 0, 0
        
        # Count A1 words in content words
        a1_count = sum(1 for word in content_words if word in a1_vocab)
        
        # Calculate ratio
        ratio = a1_count / len(content_words) if content_words else 0.0
        
        return ratio, a1_count, len(content_words)
    
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
        - Token Count: Actual tokens generated by the model (if tokenizer available)
        - Difficult Words: Direct, interpretable vocabulary accessibility metric
        
        Note: Other statistics (sentence count, syllable counts, etc.) are calculated
        internally by primary metrics and don't provide independent insights.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, int]: Dictionary containing secondary descriptive statistics
        """
        if not text or not text.strip():
            stats = {
                'word_count': 0,
                'difficult_words': 0
            }
            if self.tokenizer is not None:
                stats['token_count'] = 0
            return stats
        
        stats = {
            'word_count': textstat.lexicon_count(text),
            'difficult_words': textstat.difficult_words(text)
        }
        
        # Add token count if tokenizer is available
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer(text)
                stats['token_count'] = len(tokens)
            except Exception as e:
                print(f"⚠️ Token counting failed: {e}")
                stats['token_count'] = 0
        
        return stats
    
    
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
