# Legacy Tests and Benchmarks

This directory contains historical integration tests and benchmarks used during the llama.cpp migration process.

## Contents

### Integration Tests
- `test_qwen2_llamacpp_integration.py` - Qwen2 wrapper verification
- `test_qwen3_llamacpp_integration.py` - Qwen3 wrapper verification
- `test_smollm_llamacpp_integration.py` - SmolLM wrapper verification
- `test_tinyllama_llamacpp_integration.py` - TinyLlama wrapper verification

### Benchmarks
- `benchmark_qwen3_llamacpp.py` - Performance comparison (Transformers vs llama.cpp)

## Purpose

These tests were used to:
1. Verify correct llama.cpp integration
2. Test all intervention combinations (baseline, prompting, weighting, combined)
3. Benchmark performance improvements
4. Document migration success

## Status

✅ All models successfully migrated to llama.cpp
✅ All tests passed
✅ Benchmarks confirmed 4-4.4x speedup

These files are kept for historical reference but are no longer needed for regular development.

## Active Testing

For current testing, use:
```bash
# Run full factorial experiment for any model
python scripts/run_experiment.py --experiment [Qwen2|Qwen3|SmolLM|TinyLlama]
```
