#!/bin/bash
# ============================================================
# One-time setup script for ClusterUY
# Run this ONCE after cloning the repo to set up the environment
# ============================================================

set -e

echo "Setting up thesis environment on ClusterUY..."

# ---- 1. Install Miniconda (if not already installed) ----
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh

    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash
    source "$HOME/.bashrc"
    echo "Miniconda installed. You may need to log out and back in."
else
    echo "Conda already installed."
fi

# ---- 2. Create conda environment ----
echo "Creating 'thesis' conda environment..."
conda create -n thesis python=3.10 -y
conda activate thesis

# ---- 3. Install dependencies ----
echo "Installing Python dependencies..."

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install llama-cpp-python with CUDA support (critical for GPU inference)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install remaining requirements
PROJECT_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo"
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt" --ignore-installed llama-cpp-python torch
else
    echo "WARNING: requirements.txt not found at $PROJECT_DIR"
    echo "Install manually: pip install -r /path/to/requirements.txt"
fi

# ---- 4. Verify installation ----
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from llama_cpp import Llama; print('llama-cpp-python: OK')"
python -c "import textstat; print('textstat: OK')"
python -c "import pandas; print('pandas: OK')"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Clone/upload your repo:  git clone <your-repo-url> ~/SLMs-master-thesis"
echo "  2. GGUF models will be auto-downloaded by llama-cpp-python on first run"
echo "     OR manually download them to save time:"
echo "       pip install huggingface-hub"
echo "       huggingface-cli download ggml-org/Qwen3-0.6B-GGUF --local-dir ~/models/qwen3"
echo "  3. Submit the job:  sbatch scripts/clusteruy/run_multiweight.sh"
echo "  4. Monitor:  squeue -u \$USER --long"
