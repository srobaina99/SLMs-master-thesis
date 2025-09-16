#!/usr/bin/env python3
"""
Entry point script for Qwen3 chat interface.
Run this script to start an interactive chat session with Qwen3.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.qwen.qwen3.qwen3_weighted import start_chat
from src.evaluation.text_complexity.text_evaluator import TextEvaluator

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Qwen3 Chat Interface')
    parser.add_argument('--system', type=str, 
                       default="You are a helpful English teacher for beginner students.",
                       help='System prompt for the chat')
    parser.add_argument('--words', type=str, default="",
                       help='Comma-separated list of words to weight')
    parser.add_argument('--factor', type=float, default=2.0,
                       help='Weight factor for specified words')
    parser.add_argument('--thinking', action='store_true',
                       help='Enable thinking mode')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose mode')
    
    args = parser.parse_args()
    
    # Process weighted words
    weighted_words = [word.strip() for word in args.words.split(',')] if args.words else []
    if '' in weighted_words:
        weighted_words.remove('')
    
    print("🚀 Starting Qwen3 Chat Interface")
    print(f"System prompt: {args.system}")
    print(f"Weighted words: {weighted_words}")
    print(f"Weight factor: {args.factor}")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    start_chat(
        system_prompt=args.system,
        weighted_words=weighted_words,
        weight_factor=args.factor,
        enable_thinking=args.thinking,
        verbose=args.verbose
    )
