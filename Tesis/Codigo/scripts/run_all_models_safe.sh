#!/bin/bash
# Safe batch runner for all models - runs each model sequentially with cleanup between
# This is a fallback option if the integrated cleanup doesn't work as expected

set -e  # Exit on error

# Activate virtual environment
source venv/bin/activate

# Array of models to run
MODELS=("Phi3" "Qwen2" "Qwen3" "TinyLlama")

# Number of prompts (default: all)
PROMPTS="${1:-all}"

echo "🚀 Running safe batch experiment for all models"
echo "📝 Prompts: $PROMPTS"
echo "🤖 Models: ${MODELS[@]}"
echo ""

# Run each model sequentially
for model in "${MODELS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🤖 Running experiment for: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python scripts/run_experiment.py --experiment "$model" --prompts "$PROMPTS" --no-plots
    
    echo ""
    echo "✅ $model completed"
    echo "⏸️  Pausing 5 seconds to let system stabilize..."
    sleep 5
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All models completed successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


