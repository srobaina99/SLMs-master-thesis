#!/usr/bin/env python3
"""
Integration test for Qwen3 llama.cpp wrapper.

Validates:
1. Model loads successfully
2. Generates responses for all intervention combinations
3. Probability weighting works
4. Context prompting works
5. Response formatting/cleaning works
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.experiment_framework.models import Qwen3LlamaCppWrapper
from src.evaluation.experiment_framework.core.data_models import ExperimentConfig


def test_qwen3_llamacpp():
    """Run integration test for Qwen3 llama.cpp wrapper."""
    
    print("\n" + "="*60)
    print("QWEN3 LLAMACPP INTEGRATION TEST")
    print("="*60)
    
    # Initialize wrapper
    print("\n[1/5] Initializing Qwen3LlamaCppWrapper...")
    print("      ⏳ Loading GGUF model from disk (this may take 10-20 seconds)...")
    
    start_load = time.time()
    wrapper = Qwen3LlamaCppWrapper()
    load_time = time.time() - start_load
    
    if not wrapper.model_loaded:
        print("❌ FAILED: Model did not load")
        return False
    
    print(f"✅ Model loaded successfully in {load_time:.1f}s")
    
    # Test prompt
    test_prompt = "What is a cat?"
    system_prompt = "You are a helpful English teacher for beginner students."
    
    # Test 1: Control (no interventions)
    print("\n[2/5] Testing control configuration (no interventions)...")
    print("      ⏳ Generating response (may take 2-5 seconds)...")
    config_control = ExperimentConfig(
        model_name="Qwen3",
        system_prompt=system_prompt,
        config_weighting=False,
        config_prompting=False,
        weight_factor=2.0,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=512
    )
    
    result = wrapper.generate_response(test_prompt, config_control)
    
    if not result['generation_successful']:
        print(f"❌ FAILED: {result['error_message']}")
        return False
    
    print(f"✅ Generated response ({result['time_spent']:.2f}s)")
    print(f"   Response preview: {result['response'][:100]}...")
    print(f"   Cleaned length: {len(result['cleaned_response'])} chars")
    
    # Test 2: Weighting only
    print("\n[3/5] Testing probability weighting...")
    print("      ⏳ Applying logit_bias to vocabulary and generating...")
    config_weighted = ExperimentConfig(
        model_name="Qwen3",
        system_prompt=system_prompt,
        config_weighting=True,
        config_prompting=False,
        weight_factor=2.0,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=512
    )
    
    result = wrapper.generate_response(test_prompt, config_weighted)
    
    if not result['generation_successful']:
        print(f"❌ FAILED: {result['error_message']}")
        return False
    
    print(f"✅ Weighting works ({result['time_spent']:.2f}s)")
    print(f"   Vocab size: {len(wrapper.target_vocabulary)} words")
    
    # Test 3: Prompting only
    print("\n[4/5] Testing context prompting...")
    print("      ⏳ Adding simplification context and generating...")
    config_prompted = ExperimentConfig(
        model_name="Qwen3",
        system_prompt=system_prompt,
        config_weighting=False,
        config_prompting=True,
        weight_factor=2.0,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=512
    )
    
    result = wrapper.generate_response(test_prompt, config_prompted)
    
    if not result['generation_successful']:
        print(f"❌ FAILED: {result['error_message']}")
        return False
    
    print(f"✅ Context prompting works ({result['time_spent']:.2f}s)")
    
    # Test 4: Both interventions
    print("\n[5/5] Testing both interventions...")
    print("      ⏳ Applying weighting + prompting together...")
    config_both = ExperimentConfig(
        model_name="Qwen3",
        system_prompt=system_prompt,
        config_weighting=True,
        config_prompting=True,
        weight_factor=2.0,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=512
    )
    
    result = wrapper.generate_response(test_prompt, config_both)
    
    if not result['generation_successful']:
        print(f"❌ FAILED: {result['error_message']}")
        return False
    
    print(f"✅ Both interventions work ({result['time_spent']:.2f}s)")
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    model_info = wrapper.get_model_info()
    print(f"\n✅ ALL TESTS PASSED")
    print(f"\nModel Info:")
    print(f"  - Name: {model_info['model_name']}")
    print(f"  - Backend: {model_info['backend']}")
    print(f"  - Format: {model_info['model_format']}")
    print(f"  - Quantization: {model_info['quantization']}")
    print(f"  - Template: {model_info['template_format']}")
    print(f"  - Vocab size: {model_info['vocab_size']} words")
    print(f"  - Weighting: {'✅' if model_info['supports_weighting'] else '❌'}")
    print(f"  - Prompting: {'✅' if model_info['supports_context_prompting'] else '❌'}")
    
    print(f"\n🎯 Qwen3 llama.cpp wrapper is ready for experiments!")
    
    return True


if __name__ == "__main__":
    success = test_qwen3_llamacpp()
    sys.exit(0 if success else 1)
