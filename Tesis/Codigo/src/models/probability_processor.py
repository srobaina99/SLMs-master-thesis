from transformers.generation import LogitsProcessor, LogitsProcessorList
import torch

class ProbabilityWeightingLogitsProcessor(LogitsProcessor):
    """Custom logits processor that properly modifies the probability of specific tokens."""
    
    def __init__(self, tokenizer, words_to_weight, weight_factor, verbose):
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.weight_factor = weight_factor
        
        # Pre-compute all target token IDs for vectorized operations
        all_token_ids = set()
        self.word_token_mapping = {}  # For verbose output only
        
        if self.verbose:
            print("\nWeighted words token mapping:")
        
        # Collect all unique token IDs from all words (with and without space prefix)
        for word in words_to_weight:
            # Get token IDs for the word without space prefix
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            
            # Get token IDs for the word with space prefix  
            word_with_space_ids = tokenizer.encode(" " + word, add_special_tokens=False)
            
            # Add to global set for vectorized processing
            all_token_ids.update(word_ids)
            all_token_ids.update(word_with_space_ids)
            
            # Always store mapping for mathematical equivalence (not just verbose)
            self.word_token_mapping[word] = {
                'without_space': word_ids,
                'with_space': word_with_space_ids
            }
        
        # Convert to sorted tensor for consistent indexing
        self.target_token_ids = torch.tensor(sorted(all_token_ids), dtype=torch.long)
        
        if self.verbose:
            print(f"Applying weight factor: {weight_factor}")
            print(f"Total unique target tokens: {len(self.target_token_ids)}")
            print()
    
    def __call__(self, input_ids, scores):
        # Convert to probability space once
        probs = torch.nn.functional.softmax(scores, dim=-1)
        
        if self.verbose:
            current_text = self.tokenizer.decode(input_ids[0])
            print(f"Current sequence: '{current_text}'")
            
            # Get top tokens before weighting
            top_probs, top_indices = torch.topk(probs[0], 5)
            print("Top 5 next tokens before weighting:")
            for i, (token_id, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                token = self.tokenizer.decode([token_id])
                print(f"  {i+1}. Token ID {token_id} → '{token}' (prob: {prob:.6f})")
        
        # VECTORIZED WEIGHTING: Replicate original behavior exactly
        # The original applies weights for each word->token occurrence, potentially multiple times per token
        vocab_size = scores.shape[-1]
        weights_applied = False
        
        # Apply weighting exactly as original: for each word, for each prefix type, for each token
        if self.verbose:
            # Use original structure for verbose output
            for word, token_ids_dict in self.word_token_mapping.items():
                for prefix_type, token_ids in token_ids_dict.items():
                    for token_id in token_ids:
                        if token_id < vocab_size:
                            original_prob = probs[0, token_id].item()
                            weighted_prob = original_prob * self.weight_factor
                            weighted_prob = max(0, min(1, weighted_prob))
                            probs[0, token_id] = weighted_prob
                            
                            if abs(weighted_prob - original_prob) > 1e-6:
                                weights_applied = True
                            
                            if abs(original_prob) > 1e-5:
                                token = self.tokenizer.decode([token_id])
                                print(f"Weighted: Token ID {token_id} → '{token}' (prob: {original_prob:.6f} → {weighted_prob:.6f})")
        else:
            # Optimized path: collect all token applications with their multiplicities
            token_applications = {}
            
            # Count how many times each token should be weighted (replicating original behavior)
            for word, token_ids_dict in self.word_token_mapping.items():
                for prefix_type, token_ids in token_ids_dict.items():
                    for token_id in token_ids:
                        if token_id < vocab_size:
                            token_applications[token_id] = token_applications.get(token_id, 0) + 1
            
            # Apply weights with correct multiplicity (matching original sequential application)
            for token_id, count in token_applications.items():
                current_prob = probs[0, token_id].item()
                original_prob = current_prob
                
                # Apply weight factor 'count' times sequentially (as original does)
                for _ in range(count):
                    weighted_prob = current_prob * self.weight_factor
                    weighted_prob = max(0, min(1, weighted_prob))
                    current_prob = weighted_prob
                
                probs[0, token_id] = current_prob
                
                if abs(current_prob - original_prob) > 1e-6:
                    weights_applied = True
        
        # Renormalize probabilities to sum to 1 (single operation)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Convert back to logits (numerically stable)
        scores = torch.log(probs + 1e-10)
        
        if self.verbose:
            # Get top tokens after weighting
            top_probs, top_indices = torch.topk(probs[0], 5)
            print("Top 5 next tokens after weighting:")
            for i, (token_id, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                token = self.tokenizer.decode([token_id])
                print(f"  {i+1}. Token ID {token_id} → '{token}' (prob: {prob:.6f})")
            
            if not weights_applied:
                print("WARNING: No significant weight changes were applied. Check token IDs.")
            print()
        
        return scores