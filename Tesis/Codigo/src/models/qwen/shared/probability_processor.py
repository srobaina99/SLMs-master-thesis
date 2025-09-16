from transformers.generation import LogitsProcessor, LogitsProcessorList
import torch

class ProbabilityWeightingLogitsProcessor(LogitsProcessor):
    """Custom logits processor that properly modifies the probability of specific tokens."""
    
    def __init__(self, tokenizer, words_to_weight, weight_factor, verbose):
        self.tokenizer = tokenizer
        self.word_token_ids = {}  # Dictionary to store word -> token_ids mapping
        self.verbose = verbose
        
        if self.verbose:
            print("\nWeighted words token mapping:")
        
        # For each word, get token IDs both with and without space prefix
        for word in words_to_weight:
            # Get token IDs for the word without space prefix
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            
            # Get token IDs for the word with space prefix
            word_with_space_ids = tokenizer.encode(" " + word, add_special_tokens=False)
            
            # Store both versions
            self.word_token_ids[word] = {
                'without_space': word_ids,
                'with_space': word_with_space_ids
            }
                    
        self.weight_factor = weight_factor
        if self.verbose:
            print(f"Applying weight factor: {weight_factor}")
            print()
    
    def __call__(self, input_ids, scores):
        # Get the current sequence
        if self.verbose:
            current_text = self.tokenizer.decode(input_ids[0])
            print(f"Current sequence: '{current_text}'")
            
            # Get top tokens before weighting
            probs = torch.nn.functional.softmax(scores, dim=-1)
            top_probs, top_indices = torch.topk(probs[0], 5)
            print("Top 5 next tokens before weighting:")
            for i, (token_id, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                token = self.tokenizer.decode([token_id])
                print(f"  {i+1}. Token ID {token_id} → '{token}' (prob: {prob:.6f})")
        else:
            probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # Track if any weights were applied
        weights_applied = False
        
        # Apply weighting to probabilities for all token variations
        for word, token_ids_dict in self.word_token_ids.items():
            # Check both with and without space versions
            for prefix_type, token_ids in token_ids_dict.items():
                for token_id in token_ids:
                    if token_id < scores.shape[-1]:  # Make sure token_id is within vocabulary
                        # Convert to probability space
                        original_prob = probs[0, token_id].item()
                        
                        # Apply weight to probability
                        weighted_prob = original_prob * self.weight_factor
                        
                        # Ensure probability is valid (between 0 and 1)
                        weighted_prob = max(0, min(1, weighted_prob))
                        
                        # Update the probability
                        probs[0, token_id] = weighted_prob
                        
                        # Check if this weight actually made a difference
                        if abs(weighted_prob - original_prob) > 1e-6:
                            weights_applied = True
                        
                        if self.verbose and abs(original_prob) > 1e-5:
                            token = self.tokenizer.decode([token_id])
                            print(f"Weighted: Token ID {token_id} → '{token}' (prob: {original_prob:.6f} → {weighted_prob:.6f})")
        
        # Renormalize probabilities to sum to 1
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Convert back to logits
        # We use a numerically stable approach to avoid overflow/underflow
        scores = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
        
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