from probability_processor import ProbabilityWeightingLogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
from IPython.display import display, Markdown

def generate_story_with_weighted_words(model_name, prompt, weighted_words=None, weight_factor=2.0, verbose=True):
    """
    Generate a story using the specified model and prompt, with proper probability weighting.
    
    Args:
        model_name (str): The name of the model to use for generation
        prompt (str): The prompt to generate the story from
        weighted_words (list): List of words to give higher/lower probability
        weight_factor (float): Factor to multiply the probabilities by (>1 increases probability, <1 decreases)
        verbose (bool): Whether to print detailed token mapping and probability information
    
    Returns:
        str: The generated story
    """

    # Determine device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    
    # Move model to appropriate device
    model = model.to(device)
    print(f"Model moved to device: {next(model.parameters()).device}")
    
    print(f"\nPrompt: {prompt}")
    
    # Process on selected device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Create a custom logits processor if weighted words are provided
    if weighted_words and len(weighted_words) > 0:        
        # Create the logits processor
        logits_processor = LogitsProcessorList([
            ProbabilityWeightingLogitsProcessor(tokenizer, weighted_words, weight_factor, verbose)
        ])
        
        # Generate text with the logits processor
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=200,
            num_beams=1,
            do_sample=True,  # Enable sampling to see the effect of weighting
            temperature=0.9,
            logits_processor=logits_processor
        )
    else:
        # Generate text without logits processor
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=200,
            num_beams=1
        )
    
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("\nGenerated story:")
    display(Markdown(story))
    
    return story
