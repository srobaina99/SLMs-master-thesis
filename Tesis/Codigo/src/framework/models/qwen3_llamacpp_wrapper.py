"""
Qwen3 model wrapper using llama.cpp GGUF backend.

This replaces the Transformers-based Qwen3Wrapper with a faster,
more memory-efficient llama.cpp implementation.

Benchmark results (vs Transformers+MPS):
- 4.4x faster inference (98 vs 22 tok/s)
- 57% less memory (491MB vs 1137MB)
- Supports probability weighting via logit_bias
"""

import os
import sys
from typing import Dict, Any, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from .llamacpp_base import LlamaCppBaseWrapper
from .beam_search_generator import BeamSearchGenerator
from src.framework.core.data_models import ExperimentConfig


class Qwen3LlamaCppWrapper(LlamaCppBaseWrapper):
    """
    Qwen3-0.6B model wrapper using llama.cpp GGUF backend.
    
    Model: ggml-org/Qwen3-0.6B-GGUF (Q4_0 quantization)
    Template: ChatML format
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize Qwen3 llama.cpp wrapper.
        
        Args:
            model_path: Path to GGUF file. If None, uses default location.
        """
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                project_root,
                "models",
                "gguf",
                "Qwen3-0.6B-Q4_0.gguf"
            )
        
        super().__init__(
            model_name="Qwen3",
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # Metal auto-detection
            timeout_seconds=300
        )
    
    def _format_prompt(self, user_input: str, system_prompt: str, enable_thinking: bool = False) -> str:
        """
        Format prompt using Qwen's ChatML template.

        When enable_thinking is False, appends /nothink to the user message
        to disable Qwen3's thinking mode. When True, appends /think.

        Args:
            user_input: User's message
            system_prompt: System instruction
            enable_thinking: Whether to enable Qwen3's thinking mode

        Returns:
            Formatted ChatML prompt
        """
        thinking_tag = "/think" if enable_thinking else "/nothink"
        return (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_input} {thinking_tag}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def _get_stop_tokens(self) -> List[str]:
        """
        Get stop tokens for Qwen3.
        
        Returns:
            List of ChatML stop tokens
        """
        return ["<|im_end|>", "<|endoftext|>"]
    
    def _extract_response(self, raw_output: str) -> str:
        """
        Extract clean response from Qwen3 output.
        
        Handles:
        - Thinking tags (<think>...</think>)
        - ChatML markers
        - Whitespace cleanup
        
        Args:
            raw_output: Raw text from llama.cpp
            
        Returns:
            Cleaned response text
        """
        response = raw_output.strip()
        
        # Remove thinking tags if present
        if "<think>" in response and "</think>" in response:
            # Extract content after thinking
            think_end = response.find("</think>")
            response = response[think_end + len("</think>"):].strip()
        
        # Remove any remaining ChatML markers
        response = response.replace("<|im_start|>", "")
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|endoftext|>", "")
        
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Qwen3-specific model information.
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info.update({
            'model_id': 'ggml-org/Qwen3-0.6B-GGUF',
            'quantization': 'Q4_0',
            'template_format': 'ChatML',
            'parameters': '0.6B',
            'benchmark_tokens_per_sec': 98.1,
            'benchmark_memory_mb': 491
        })
        return info
    
    def generate_with_beam_search(self, 
                                 prompt: str, 
                                 config: ExperimentConfig,
                                 beam_width: int = 4,
                                 selection_method: str = "a1_ratio") -> Dict[str, Any]:
        """
        Generate response using beam search with specified selection criterion.
        
        Args:
            prompt: Input prompt for the model
            config: Experiment configuration
            beam_width: Number of beams to maintain
            selection_method: "a1_ratio" for A1 word ratio or "max_probability" for cumulative log prob
            
        Returns:
            Dictionary with:
                - response: Selected beam response text
                - beam_selection_method: Selection method used
                - beam_a1_ratio: A1 word ratio of selected beam
                - beam_a1_count: Count of A1 words
                - beam_content_word_count: Count of content words
                - beam_cumulative_logprob: Cumulative log probability
                - beam_width: Number of beams used
                - time_spent: Generation time
                - generation_successful: Boolean success flag
        """
        import time
        start_time = time.time()
        
        if not self.model_loaded or not self.llm:
            return {
                'response': '',
                'beam_selection_method': selection_method,
                'beam_a1_ratio': 0.0,
                'beam_a1_count': 0,
                'beam_content_word_count': 0,
                'beam_cumulative_logprob': 0.0,
                'beam_width': beam_width,
                'time_spent': 0.0,
                'generation_successful': False,
                'error_message': f'{self.model_name} model not loaded'
            }
        
        try:
            # Apply context prompting if enabled
            final_prompt = prompt
            if config.config_prompting:
                final_prompt = self._add_simplification_context(prompt)
            
            # Format prompt with model-specific template
            formatted_prompt = self._format_prompt(final_prompt, config.system_prompt, config.enable_thinking)
            
            # Initialize beam search generator
            beam_generator = BeamSearchGenerator(
                llm=self.llm,
                beam_width=beam_width,
                max_length=config.max_new_tokens,
                length_penalty=1.0
            )
            
            # Run beam search using formatted prompt string
            beam_results = beam_generator.generate(
                prompt=formatted_prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k
            )
            
            # Prepare A1 vocabulary set
            a1_vocab_set = {word.lower() for word in self.target_vocabulary}
            
            # Get content words from all beams for consistent set
            content_words_union = set()
            for beam in beam_results['beams']:
                content_words = self.text_evaluator.extract_content_words(beam.sequence_text)
                content_words_union.update(content_words)
            
            # Select best beams
            selection_results = beam_generator.select_best_beams(
                beam_results['beams'],
                a1_vocab=self.target_vocabulary,
                content_words_set=content_words_union
            )
            
            # Choose beam based on selection method
            if selection_method == "a1_ratio":
                selected_beam_data = selection_results['best_by_a1_ratio']
            else:  # max_probability
                selected_beam_data = selection_results['best_by_probability']
            
            if not selected_beam_data:
                return {
                    'response': '',
                    'beam_selection_method': selection_method,
                    'beam_a1_ratio': 0.0,
                    'beam_a1_count': 0,
                    'beam_content_word_count': 0,
                    'beam_cumulative_logprob': 0.0,
                    'beam_width': beam_width,
                    'time_spent': 0.0,
                    'generation_successful': False,
                    'error_message': 'No valid beams generated'
                }
            
            # Extract response text
            response = self._extract_response(selected_beam_data['beam'].sequence_text)
            
            end_time = time.time()
            time_spent = end_time - start_time
            
            return {
                'response': response,
                'beam_selection_method': selection_method,
                'beam_a1_ratio': selected_beam_data['a1_ratio'],
                'beam_a1_count': selected_beam_data['a1_count'],
                'beam_content_word_count': selected_beam_data['content_count'],
                'beam_cumulative_logprob': selected_beam_data['cumulative_log_prob'],
                'beam_width': beam_width,
                'time_spent': time_spent,
                'generation_successful': True,
                'error_message': ''
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'response': '',
                'beam_selection_method': selection_method,
                'beam_a1_ratio': 0.0,
                'beam_a1_count': 0,
                'beam_content_word_count': 0,
                'beam_cumulative_logprob': 0.0,
                'beam_width': beam_width,
                'time_spent': end_time - start_time,
                'generation_successful': False,
                'error_message': str(e)
            }
