#!/bin/bash
# ============================================================
# ClusterUY setup — pull Singularity image
#
# Prefer sbatch pull_image.sh for long pulls (login node kills them).
# This script is a quick helper for small images only.
#
# Current workflow: SLMs-experiments/docs/clusteruy.md
# Legacy guide:     CLUSTERUY_GUIDE.md (deprecated)
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
echo "Smoke test (current framework):"
echo "  cd ~/SLMs-experiments && sbatch scripts/clusteruy/smoke_test.sh"
echo ""
echo "See: https://github.com/srobaina99/SLMs-experiments/blob/main/docs/clusteruy.md"
