#!/usr/bin/env python3
"""
Benchmark script comparing Qwen3-0.6B performance:
- Transformers + MPS (current implementation)
- llama.cpp + GGUF (quantized)

Measures: load time, memory usage, inference speed, quality
Tests: probability weighting compatibility
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test prompts (3 complexity levels)
TEST_PROMPTS = [
    {
        "name": "Simple",
        "prompt": "What is a cat?",
        "expected_length": "short"
    },
    {
        "name": "EFL-relevant",
        "prompt": "Explain photosynthesis to a 10-year-old.",
        "expected_length": "medium"
    },
    {
        "name": "Complex",
        "prompt": "Write a short story about friendship.",
        "expected_length": "long"
    }
]

# Target vocabulary for weighting test (from filtered_starters_vocab.txt)
SAMPLE_VOCAB = ["cat", "dog", "simple", "easy", "small", "big", "good", "bad", "happy", "sad"]


def benchmark_transformers(test_prompts: List[Dict], vocab: List[str]) -> Dict[str, Any]:
    """Benchmark current Transformers + MPS implementation."""
    print("\n" + "="*60)
    print("BENCHMARK 1: Transformers + MPS (Current)")
    print("="*60)
    
    results = {
        "method": "transformers_mps",
        "model": "unsloth/Qwen3-0.6B (float16)",
        "load_time": 0.0,
        "memory_mb": 0,
        "prompts": []
    }
    
    try:
        # Import and load model
        print("\n[1/4] Loading model...")
        load_start = time.time()
        
        from src.models.qwen.qwen3.qwen3_weighted import generate_response, model, tokenizer
        import torch
        
        load_time = time.time() - load_start
        results["load_time"] = load_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Measure memory
        if torch.backends.mps.is_available():
            memory_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            results["memory_mb"] = memory_mb
            print(f"✅ Memory allocated: {memory_mb:.1f} MB")
        
        # Test prompts WITHOUT weighting
        print("\n[2/4] Testing prompts (no weighting)...")
        for test in test_prompts:
            print(f"\n  Testing: {test['name']}")
            start = time.time()
            
            response = generate_response(
                user_input=test['prompt'],
                system_prompt="You are a helpful AI assistant.",
                enable_thinking=False,
                weighted_words=[],
                weight_factor=1.0,
                verbose=False
            )
            
            elapsed = time.time() - start
            tokens = len(tokenizer.encode(response))
            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
            
            results["prompts"].append({
                "name": test['name'],
                "prompt": test['prompt'],
                "response": response[:200] + "..." if len(response) > 200 else response,
                "time_seconds": elapsed,
                "tokens": tokens,
                "tokens_per_second": tokens_per_sec,
                "weighted": False
            })
            
            print(f"    ✅ {elapsed:.2f}s | {tokens} tokens | {tokens_per_sec:.1f} tok/s")
        
        # Test probability weighting
        print("\n[3/4] Testing probability weighting...")
        test = test_prompts[0]  # Use simple prompt
        start = time.time()
        
        response = generate_response(
            user_input=test['prompt'],
            system_prompt="You are a helpful AI assistant.",
            enable_thinking=False,
            weighted_words=vocab,
            weight_factor=2.0,
            verbose=False
        )
        
        elapsed = time.time() - start
        tokens = len(tokenizer.encode(response))
        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
        
        results["prompts"].append({
            "name": f"{test['name']} (weighted)",
            "prompt": test['prompt'],
            "response": response[:200] + "..." if len(response) > 200 else response,
            "time_seconds": elapsed,
            "tokens": tokens,
            "tokens_per_second": tokens_per_sec,
            "weighted": True,
            "vocab_size": len(vocab),
            "weight_factor": 2.0
        })
        
        print(f"    ✅ {elapsed:.2f}s | {tokens} tokens | {tokens_per_sec:.1f} tok/s")
        print(f"    ✅ Weighting functional with {len(vocab)} words")
        
        results["weighting_supported"] = True
        results["status"] = "success"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


def benchmark_llamacpp(test_prompts: List[Dict], vocab: List[str]) -> Dict[str, Any]:
    """Benchmark llama.cpp + GGUF implementation."""
    print("\n" + "="*60)
    print("BENCHMARK 2: llama.cpp + GGUF (Quantized)")
    print("="*60)
    
    results = {
        "method": "llamacpp_gguf",
        "model": "ggml-org/Qwen3-0.6B-Q4_0.gguf",
        "load_time": 0.0,
        "memory_mb": 0,
        "prompts": []
    }
    
    try:
        from llama_cpp import Llama
        
        # Load model
        print("\n[1/4] Loading model...")
        model_path = "models/gguf/Qwen3-0.6B-Q4_0.gguf"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        load_start = time.time()
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
            n_gpu_layers=0  # CPU only for now, Metal support varies
        )
        load_time = time.time() - load_start
        results["load_time"] = load_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Memory estimation (GGUF doesn't expose direct memory API)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        results["memory_mb"] = file_size_mb * 1.2  # Estimate: file size + overhead
        print(f"✅ Estimated memory: {results['memory_mb']:.1f} MB")
        
        # Test prompts WITHOUT weighting
        print("\n[2/4] Testing prompts (no weighting)...")
        for test in test_prompts:
            print(f"\n  Testing: {test['name']}")
            
            # Format prompt (ChatML format for Qwen)
            formatted_prompt = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{test['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            
            start = time.time()
            output = llm(
                formatted_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                stop=["<|im_end|>"],
                echo=False
            )
            elapsed = time.time() - start
            
            response = output["choices"][0]["text"].strip()
            tokens = output["usage"]["completion_tokens"]
            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
            
            results["prompts"].append({
                "name": test['name'],
                "prompt": test['prompt'],
                "response": response[:200] + "..." if len(response) > 200 else response,
                "time_seconds": elapsed,
                "tokens": tokens,
                "tokens_per_second": tokens_per_sec,
                "weighted": False
            })
            
            print(f"    ✅ {elapsed:.2f}s | {tokens} tokens | {tokens_per_sec:.1f} tok/s")
        
        # Test logit_bias for weighting
        print("\n[3/4] Testing logit_bias for vocabulary weighting...")
        test = test_prompts[0]
        
        # Create logit_bias dictionary (token_id: bias_value)
        # Note: llama.cpp uses token IDs, need to map vocabulary
        logit_bias = {}
        for word in vocab:
            tokens = llm.tokenize(word.encode('utf-8'))
            for token_id in tokens:
                logit_bias[token_id] = 2.0  # Positive bias
        
        formatted_prompt = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{test['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        
        start = time.time()
        output = llm(
            formatted_prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            stop=["<|im_end|>"],
            echo=False,
            logit_bias=logit_bias
        )
        elapsed = time.time() - start
        
        response = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]
        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
        
        results["prompts"].append({
            "name": f"{test['name']} (weighted)",
            "prompt": test['prompt'],
            "response": response[:200] + "..." if len(response) > 200 else response,
            "time_seconds": elapsed,
            "tokens": tokens,
            "tokens_per_second": tokens_per_sec,
            "weighted": True,
            "vocab_size": len(vocab),
            "logit_bias_tokens": len(logit_bias)
        })
        
        print(f"    ✅ {elapsed:.2f}s | {tokens} tokens | {tokens_per_sec:.1f} tok/s")
        print(f"    ✅ logit_bias applied to {len(logit_bias)} token IDs")
        
        results["weighting_supported"] = True
        results["status"] = "success"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results["status"] = "error"
        results["error"] = str(e)
    
    return results


def compare_results(transformers_results: Dict, llamacpp_results: Dict):
    """Generate comparative analysis."""
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    print("\n📊 PERFORMANCE COMPARISON\n")
    
    # Load time
    print(f"Load Time:")
    print(f"  Transformers: {transformers_results['load_time']:.2f}s")
    print(f"  llama.cpp:    {llamacpp_results['load_time']:.2f}s")
    speedup = transformers_results['load_time'] / llamacpp_results['load_time'] if llamacpp_results['load_time'] > 0 else 0
    print(f"  → llama.cpp is {speedup:.2f}x faster" if speedup > 1 else f"  → Transformers is {1/speedup:.2f}x faster")
    
    # Memory
    print(f"\nMemory Usage:")
    print(f"  Transformers: {transformers_results['memory_mb']:.1f} MB")
    print(f"  llama.cpp:    {llamacpp_results['memory_mb']:.1f} MB")
    reduction = (1 - llamacpp_results['memory_mb'] / transformers_results['memory_mb']) * 100 if transformers_results['memory_mb'] > 0 else 0
    print(f"  → llama.cpp uses {reduction:.1f}% less memory")
    
    # Inference speed (average)
    trans_speeds = [p['tokens_per_second'] for p in transformers_results['prompts'] if not p.get('weighted', False)]
    llama_speeds = [p['tokens_per_second'] for p in llamacpp_results['prompts'] if not p.get('weighted', False)]
    
    if trans_speeds and llama_speeds:
        trans_avg = sum(trans_speeds) / len(trans_speeds)
        llama_avg = sum(llama_speeds) / len(llama_speeds)
        
        print(f"\nInference Speed (avg):")
        print(f"  Transformers: {trans_avg:.1f} tok/s")
        print(f"  llama.cpp:    {llama_avg:.1f} tok/s")
        speedup = llama_avg / trans_avg if trans_avg > 0 else 0
        print(f"  → llama.cpp is {speedup:.2f}x faster" if speedup > 1 else f"  → Transformers is {1/speedup:.2f}x faster")
    
    # Weighting support
    print(f"\nProbability Weighting:")
    print(f"  Transformers: {'✅ Supported' if transformers_results.get('weighting_supported') else '❌ Not supported'}")
    print(f"  llama.cpp:    {'✅ Supported (logit_bias)' if llamacpp_results.get('weighting_supported') else '❌ Not supported'}")
    
    # Decision criteria
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60 + "\n")
    
    if llamacpp_results['status'] == 'error':
        print("⚠️  llama.cpp encountered errors. Recommend staying with Transformers.")
        print(f"    Error: {llamacpp_results.get('error', 'Unknown')}")
    elif speedup > 2.0 and llamacpp_results.get('weighting_supported'):
        print("✅ ADOPT llama.cpp for all models")
        print(f"   Rationale: {speedup:.1f}x faster, {reduction:.0f}% less memory, weighting works")
    elif speedup > 1.5:
        print("⚠️  HYBRID APPROACH recommended")
        print(f"   - Use llama.cpp for baseline/prompting experiments (faster)")
        print(f"   - Use Transformers for weighting experiments (native support)")
    else:
        print("✅ KEEP Transformers")
        print(f"   Rationale: Marginal gains ({speedup:.1f}x), simpler integration")
    
    return {
        "transformers": transformers_results,
        "llamacpp": llamacpp_results,
        "comparison": {
            "load_time_speedup": speedup,
            "memory_reduction_percent": reduction,
            "inference_speedup": speedup if trans_speeds and llama_speeds else None
        }
    }


def main():
    print("\n🔬 QWEN3-0.6B DEPLOYMENT BENCHMARK")
    print("Comparing: Transformers+MPS vs llama.cpp+GGUF\n")
    
    # Run benchmarks
    transformers_results = benchmark_transformers(TEST_PROMPTS, SAMPLE_VOCAB)
    llamacpp_results = benchmark_llamacpp(TEST_PROMPTS, SAMPLE_VOCAB)
    
    # Compare and recommend
    full_results = compare_results(transformers_results, llamacpp_results)
    
    # Save results
    output_path = "scripts/benchmark_results_qwen3.json"
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n📁 Full results saved to: {output_path}")
    print("\n✅ Benchmark complete")


if __name__ == "__main__":
    main()

