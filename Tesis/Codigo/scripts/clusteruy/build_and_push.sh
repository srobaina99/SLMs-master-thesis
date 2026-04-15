#!/bin/bash
# ============================================================
# Build the Docker image and push to Docker Hub.
# Run this on your LOCAL machine (not on ClusterUY).
#
# Prerequisites:
#   - Docker installed and running
#   - Docker Hub account (https://hub.docker.com)
#   - Logged in: docker login
#
# Usage:
#   bash scripts/clusteruy/build_and_push.sh <dockerhub_username>
# ============================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash build_and_push.sh <dockerhub_username>"
    echo "Example: bash build_and_push.sh srobaina99"
    exit 1
fi

DOCKER_USER="$1"
IMAGE_NAME="$DOCKER_USER/slm-thesis:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building Docker image: $IMAGE_NAME"
echo "This may take 10-20 minutes (compiling llama-cpp-python with CUDA)..."
echo ""

docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo ""
echo "Pushing to Docker Hub..."
docker push "$IMAGE_NAME"

echo ""
echo "============================================"
echo "Done! Image pushed to: $IMAGE_NAME"
echo "============================================"
echo ""
echo "On ClusterUY, pull with:"
echo "  singularity pull --name slm-thesis.sif docker://$IMAGE_NAME"
