#!/bin/bash
# ============================================================
# One-time environment setup for ClusterUY
#
# IMPORTANT: Run this inside an interactive session (1 hour),
# NOT on the login node. See CLUSTERUY_GUIDE.md for details.
#
# Safe to re-run: each step checks if it was already completed.
#
# Ref: https://www.cluster.uy/ayuda/primeros_pasos/
# Ref: https://www.cluster.uy/ayuda/lista_software/
# ============================================================

set -e

PROJECT_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo"

echo "Setting up thesis environment on ClusterUY..."

# ---- 1. Install Miniconda ----
# ClusterUY runs CentOS 7 (glibc 2.17). The latest Miniconda requires
# glibc >= 2.28, so we use the py310_23.1.0-1 release which is the last
# version compatible with CentOS 7.
if [ -d "$HOME/miniconda3" ]; then
    echo "[1/6] Miniconda already installed. Skipping."
else
    echo "[1/6] Installing Miniconda (py310_23.1.0-1 for CentOS 7 compatibility)..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
fi

# Ensure conda is available in this shell
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ---- 2. Create conda environment ----
if conda env list | grep -q "^thesis "; then
    echo "[2/6] Conda environment 'thesis' already exists. Skipping."
else
    echo "[2/6] Creating 'thesis' conda environment (Python 3.10)..."
    conda create -n thesis python=3.10 -y
fi

conda activate thesis

# ---- 3. Install PyTorch with CUDA 12.1 ----
# Plain pip install torch installs CPU-only.
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[3/6] PyTorch with CUDA already installed. Skipping."
else
    echo "[3/6] Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# ---- 4. Install llama-cpp-python with CUDA ----
# Prebuilt wheel avoids compiling from source (requires C++17 / GCC 8+,
# takes >30 min, and often times out in interactive sessions).
if python -c "from llama_cpp import Llama" 2>/dev/null; then
    echo "[4/6] llama-cpp-python already installed. Skipping."
else
    echo "[4/6] Installing llama-cpp-python (prebuilt CUDA 12.1 wheel)..."
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
fi

# ---- 5. Install compiled packages via conda ----
# Avoids CentOS 7 build toolchain issues (old GCC, missing Cython, etc.)
if python -c "import pandas; import matplotlib; import seaborn" 2>/dev/null; then
    echo "[5/6] Conda packages already installed. Skipping."
else
    echo "[5/6] Installing conda packages (pandas, numpy, matplotlib, seaborn)..."
    conda install -y pandas numpy matplotlib seaborn pyarrow ipython -c conda-forge
fi

# ---- 6. Install remaining pip packages ----
if python -c "import textstat; import transformers; import accelerate" 2>/dev/null; then
    echo "[6/6] Pip packages already installed. Skipping."
else
    echo "[6/6] Installing pip packages..."
    pip install textstat transformers accelerate sentencepiece safetensors Markdown
fi

# ---- Verify installation ----
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from llama_cpp import Llama; print('  llama-cpp-python: OK')"
python -c "import textstat; print('  textstat: OK')"
python -c "import pandas; print(f'  pandas: {pandas.__version__}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"

echo ""
echo "============================================"
echo "Setup complete! All steps verified."
echo "============================================"
