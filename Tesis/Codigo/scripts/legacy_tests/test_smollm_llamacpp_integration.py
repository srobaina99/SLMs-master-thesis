"""
Integration test for SmolLM llama.cpp wrapper.

Tests:
1. Model loading
2. Basic response generation
3. All intervention combinations (baseline, prompting, weighting, both)
4. Response quality and timing

Run from project root:
    python scripts/test_smollm_llamacpp_integration.py
"""

import os
import sys
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.evaluation.experiment_framework.models.smollm_llamacpp_wrapper import SmolLMLlamaCppWrapper
from src.evaluation.experiment_framework.core.data_models import ExperimentConfig


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_progress(message: str):
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def test_model_loading():
    """Test 1: Model loading and initialization."""
    print_section("TEST 1: Model Loading")
    
    print_progress("Initializing SmolLM wrapper...")
    start_time = time.time()
    
    wrapper = SmolLMLlamaCppWrapper()
    
    load_time = time.time() - start_time
    
    print_progress(f"Model loaded in {load_time:.2f}s")
    print(f"✅ Model loaded: {wrapper.model_loaded}")
    print(f"✅ Model name: {wrapper.model_name}")
    
    # Print model info
    info = wrapper.get_model_info()
    print("\n📊 Model Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    return wrapper


def test_basic_generation(wrapper: SmolLMLlamaCppWrapper):
    """Test 2: Basic response generation (baseline - no interventions)."""
    print_section("TEST 2: Basic Generation (Baseline)")
    
    test_prompt = "What is your favorite color?"
    
    # Create baseline config (no interventions)
    config = ExperimentConfig(
        model_name="SmolLM",
        config_prompting=False,
        config_weighting=False,
        weight_factor=0.0,
        system_prompt="You are a helpful assistant.",
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    print_progress(f"Generating response for: '{test_prompt}'")
    print_progress("⏳ Generating... (this may take 5-15 seconds)")
    
    start_time = time.time()
    result = wrapper.generate_response(test_prompt, config)
    gen_time = time.time() - start_time
    
    print_progress(f"Generation complete in {gen_time:.2f}s")
    
    print(f"\n✅ Generation successful: {result['generation_successful']}")
    print(f"✅ Time spent: {result['time_spent']:.2f}s")
    print(f"\n📝 Response:\n{result['response']}")
    print(f"\n🧹 Cleaned response:\n{result['cleaned_response']}")
    
    # Calculate tokens per second (approximate)
    response_length = len(result['response'].split())
    tokens_per_sec = response_length / result['time_spent'] if result['time_spent'] > 0 else 0
    print(f"\n⚡ Approximate speed: {tokens_per_sec:.1f} words/sec")
    
    return result['generation_successful']


def test_intervention_combinations(wrapper: SmolLMLlamaCppWrapper):
    """Test 3: All intervention combinations."""
    print_section("TEST 3: Intervention Combinations")
    
    test_prompt = "Tell me about dogs."
    
    interventions = [
        ("Baseline", False, False, 0.0),
        ("Prompting Only", True, False, 0.0),
        ("Weighting Only", False, True, 2.0),
        ("Both Interventions", True, True, 2.0)
    ]
    
    results = []
    
    for idx, (name, prompting, weighting, weight) in enumerate(interventions, 1):
        print(f"\n--- Test 3.{idx}: {name} ---")
        
        config = ExperimentConfig(
            model_name="SmolLM",
            config_prompting=prompting,
            config_weighting=weighting,
            weight_factor=weight,
            system_prompt="You are a helpful assistant.",
            max_new_tokens=80,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        print_progress(f"Config: prompting={prompting}, weighting={weighting}, weight={weight}")
        print_progress("⏳ Generating...")
        
        start_time = time.time()
        result = wrapper.generate_response(test_prompt, config)
        gen_time = time.time() - start_time
        
        print_progress(f"Complete in {gen_time:.2f}s")
        
        success = result['generation_successful']
        response_preview = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
        
        print(f"✅ Success: {success}")
        print(f"📝 Response preview: {response_preview}")
        
        results.append({
            'name': name,
            'success': success,
            'time': result['time_spent'],
            'response_length': len(result['response'])
        })
    
    # Summary
    print("\n" + "="*70)
    print("INTERVENTION TEST SUMMARY")
    print("="*70)
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        print(f"{status} | {r['name']:20s} | {r['time']:.2f}s | {r['response_length']} chars")
    
    all_passed = all(r['success'] for r in results)
    return all_passed


def test_response_quality(wrapper: SmolLMLlamaCppWrapper):
    """Test 4: Response quality checks."""
    print_section("TEST 4: Response Quality")
    
    test_cases = [
        ("Simple question", "What is the sun?"),
        ("Creative prompt", "Write a short poem about rain."),
        ("Instructional", "Explain how to make a sandwich.")
    ]
    
    all_good = True
    
    for idx, (case_name, prompt) in enumerate(test_cases, 1):
        print(f"\n--- Test 4.{idx}: {case_name} ---")
        print(f"Prompt: '{prompt}'")
        
        config = ExperimentConfig(
            model_name="SmolLM",
            config_prompting=False,
            config_weighting=False,
            weight_factor=0.0,
            system_prompt="You are a helpful assistant.",
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        print_progress("⏳ Generating...")
        result = wrapper.generate_response(prompt, config)
        
        response = result['response']
        
        # Quality checks
        checks = {
            'Non-empty': len(response) > 0,
            'Reasonable length': 10 < len(response) < 1000,
            'No error message': 'error' not in response.lower() and 'failed' not in response.lower(),
            'Contains words': len(response.split()) > 3
        }
        
        print(f"\n📊 Quality Checks:")
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
        
        print(f"\n📝 Response ({len(response)} chars):\n{response[:200]}...")
        
        if not all(checks.values()):
            all_good = False
    
    return all_good


def main():
    """Run all integration tests."""
    print_section("SmolLM llama.cpp Integration Test Suite")
    print("Testing SmolLM-1.7B-Instruct with llama.cpp GGUF backend")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    try:
        # Test 1: Model Loading
        wrapper = test_model_loading()
        if not wrapper.model_loaded:
            print("\n❌ CRITICAL: Model failed to load. Aborting tests.")
            return False
        
        # Test 2: Basic Generation
        basic_success = test_basic_generation(wrapper)
        if not basic_success:
            print("\n❌ CRITICAL: Basic generation failed. Aborting tests.")
            return False
        
        # Test 3: Intervention Combinations
        interventions_success = test_intervention_combinations(wrapper)
        
        # Test 4: Response Quality
        quality_success = test_response_quality(wrapper)
        
        # Final Summary
        total_time = time.time() - overall_start
        
        print_section("FINAL TEST SUMMARY")
        print(f"✅ Model Loading: PASS")
        print(f"{'✅' if basic_success else '❌'} Basic Generation: {'PASS' if basic_success else 'FAIL'}")
        print(f"{'✅' if interventions_success else '❌'} Intervention Combinations: {'PASS' if interventions_success else 'FAIL'}")
        print(f"{'✅' if quality_success else '❌'} Response Quality: {'PASS' if quality_success else 'FAIL'}")
        print(f"\n⏱️  Total test time: {total_time:.2f}s")
        
        all_passed = basic_success and interventions_success and quality_success
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! SmolLM integration is functional.")
        else:
            print("\n⚠️  SOME TESTS FAILED. Review output above.")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
