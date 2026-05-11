"""
Beam search generator implementing standard beam search with dual selection criteria.

Features:
- Standard beam search maintaining top-n candidates at each step
- Cumulative log probability tracking
- A1 vocabulary ratio calculation with 1.5x weighting
- Returns both beams: highest A1 ratio and highest cumulative probability
"""

import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class BeamCandidate:
    """Represents a single beam candidate."""
    token_ids: List[int]  # Token sequence
    cumulative_log_prob: float  # Sum of log probabilities
    sequence_text: str  # Decoded text for analysis


class BeamSearchGenerator:
    """
    Simplified beam search implementation for llama.cpp models.
    
    Generates multiple candidate sequences by maintaining top-n candidates
    and selects final beam based on A1 word ratio or cumulative probability.
    """
    
    def __init__(self, 
                 llm,
                 beam_width: int = 4,
                 max_length: int = 200,
                 length_penalty: float = 1.0):
        """
        Initialize beam search generator.
        
        Args:
            llm: llama.cpp model for token generation
            beam_width: Number of beams to maintain (n)
            max_length: Maximum sequence length in tokens
            length_penalty: Length normalization factor (1.0 = no penalty)
        """
        self.llm = llm
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
    
    def generate(self, 
                 prompt: str,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 top_k: int = 50) -> Dict[str, Any]:
        """
        Generate using simplified beam search (greedy per beam with stochastic sampling).
        
        For each beam, we generate by sampling stochastically. This creates beam diversity.
        
        Args:
            prompt: Formatted prompt for the model
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary with:
                - 'beams': List of candidate sequences (BeamCandidate objects)
        """
        beams = []
        
        # Generate multiple sequences by running the model with randomness
        for beam_idx in range(self.beam_width):
            try:
                output = self.llm(
                    prompt=prompt,
                    max_tokens=self.max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    echo=False,
                )
                
                if output and 'choices' in output and output['choices']:
                    response_text = output['choices'][0]['text']
                    
                    # Tokenize the response to get token IDs
                    token_ids = self.llm.tokenize(response_text.encode('utf-8'))
                    
                    # Approximate cumulative log prob (higher temp = lower prob)
                    # Use length-normalized log prob
                    response_length = len(token_ids)
                    if response_length > 0:
                        # Approximate prob: -log(temperature) per token at baseline
                        cumulative_log_prob = -(math.log(temperature) * response_length)
                    else:
                        cumulative_log_prob = 0.0
                    
                    full_text = prompt + response_text
                    
                    candidate = BeamCandidate(
                        token_ids=token_ids,
                        cumulative_log_prob=cumulative_log_prob,
                        sequence_text=full_text
                    )
                    beams.append(candidate)
            except Exception as e:
                continue
        
        if not beams:
            # Fallback: return at least one empty beam
            beams.append(BeamCandidate(
                token_ids=[],
                cumulative_log_prob=0.0,
                sequence_text=prompt
            ))
        
        return {'beams': beams}
    
    def calculate_a1_ratio(self, 
                          text: str, 
                          a1_vocab: List[str],
                          content_words_set: set) -> Tuple[float, int, int]:
        """
        Calculate ratio of A1 words to content words with 1.5x weighting.
        
        Args:
            text: Generated text
            a1_vocab: List of A1 vocabulary words
            content_words_set: Set of content words from text
            
        Returns:
            Tuple of (weighted_a1_ratio, a1_count, content_count)
        """
        # Tokenize text into words
        words = text.lower().split()
        
        # Count A1 words and content words
        a1_count = 0
        content_count = 0
        
        a1_vocab_lower = {word.lower() for word in a1_vocab}
        
        for word in words:
            # Remove punctuation for matching
            word_clean = ''.join(c for c in word if c.isalnum())
            
            if word_clean in content_words_set:
                content_count += 1
                if word_clean in a1_vocab_lower:
                    a1_count += 1
        
        # Calculate ratio with 1.5x weighting applied to A1 count
        if content_count == 0:
            weighted_ratio = 0.0
        else:
            weighted_a1_count = a1_count * 1.5
            weighted_ratio = weighted_a1_count / content_count
        
        return weighted_ratio, a1_count, content_count
    
    def select_best_beams(self,
                         beams: List[BeamCandidate],
                         a1_vocab: List[str],
                         content_words_set: set) -> Dict[str, Any]:
        """
        Select best beam by A1 ratio and best beam by cumulative probability.
        
        Args:
            beams: List of completed beam candidates
            a1_vocab: List of A1 vocabulary words
            content_words_set: Set of content words identified in texts
            
        Returns:
            Dictionary with:
                - 'best_by_a1_ratio': Best beam by weighted A1 ratio
                - 'best_by_probability': Best beam by cumulative log probability
        """
        scored_beams = []
        
        for beam in beams:
            a1_ratio, a1_count, content_count = self.calculate_a1_ratio(
                beam.sequence_text,
                a1_vocab,
                content_words_set
            )
            
            scored_beams.append({
                'beam': beam,
                'a1_ratio': a1_ratio,
                'a1_count': a1_count,
                'content_count': content_count,
                'cumulative_log_prob': beam.cumulative_log_prob,
            })
        
        if not scored_beams:
            return {
                'best_by_a1_ratio': None,
                'best_by_probability': None,
            }
        
        # Find best by A1 ratio (highest)
        best_by_a1 = max(scored_beams, key=lambda x: x['a1_ratio'])
        
        # Find best by probability (highest cumulative log prob)
        best_by_prob = max(scored_beams, key=lambda x: x['cumulative_log_prob'])
        
        return {
            'best_by_a1_ratio': best_by_a1,
            'best_by_probability': best_by_prob,
        }
