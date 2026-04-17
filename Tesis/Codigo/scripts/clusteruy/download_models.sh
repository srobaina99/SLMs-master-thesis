#!/bin/bash
# ============================================================
# Download GGUF model files from HuggingFace.
# Run on the ClusterUY login node (file management, ~3.8 GB total).
#
# Usage:
#   cd ~/SLMs-master-thesis/Tesis/Codigo
#   bash scripts/clusteruy/download_models.sh
# ============================================================

set -e

MODELS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/models/gguf"
mkdir -p "$MODELS_DIR"

download() {
    local repo="$1"
    local filename="$2"
    local dest="$MODELS_DIR/$filename"
    local url="https://huggingface.co/${repo}/resolve/main/${filename}"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "[skip] $filename already exists ($(du -h "$dest" | cut -f1))"
        return
    fi

    echo "[dl]   $filename  <-  $repo"
    wget -q --show-progress -O "$dest" "$url"
}

download "microsoft/Phi-3-mini-4k-instruct-gguf"  "Phi-3-mini-4k-instruct-q4.gguf"
download "Qwen/Qwen2.5-0.5B-Instruct-GGUF"        "qwen2.5-0.5b-instruct-q4_0.gguf"
download "ggml-org/Qwen3-0.6B-GGUF"               "Qwen3-0.6B-Q4_0.gguf"
download "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

echo ""
echo "Done. Contents of $MODELS_DIR:"
ls -lh "$MODELS_DIR"
