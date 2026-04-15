#!/bin/bash
# ============================================================
# ClusterUY setup — pull Singularity image
#
# Run this on the LOGIN NODE (it's just a file download).
# See CLUSTERUY_GUIDE.md for the full workflow.
#
# Prerequisites:
#   - Docker image already pushed to Docker Hub
#     (see build_and_push.sh for local build instructions)
#
# Ref: https://www.cluster.uy/ayuda/singularity/
# ============================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash setup_env.sh <dockerhub_username>"
    echo "Example: bash setup_env.sh srobaina99"
    exit 1
fi

DOCKER_USER="$1"
IMAGE_NAME="$DOCKER_USER/slm-thesis:latest"
SIF_PATH="$HOME/slm-thesis.sif"

if [ -f "$SIF_PATH" ]; then
    echo "Singularity image already exists at $SIF_PATH"
    echo "To re-pull, delete it first: rm $SIF_PATH"
    exit 0
fi

echo "Pulling Singularity image from Docker Hub: $IMAGE_NAME"
echo "This may take a few minutes..."
singularity pull --name "$SIF_PATH" "docker://$IMAGE_NAME"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo "Image saved to: $SIF_PATH"
echo ""
echo "Test with:"
echo "  interactivo -gpun"
echo "  cd ~/SLMs-master-thesis/Tesis/Codigo"
echo "  singularity exec --nv --bind \$(pwd):/workspace ~/slm-thesis.sif python /workspace/scripts/run_experiment.py --experiment multi_weight --weights '1.5,2.0' --prompts 2 --model Qwen3 --no-plots"
