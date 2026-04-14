#!/bin/bash
#SBATCH --job-name=multiweight_exp
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --tmp=50G
#SBATCH --output=multiweight_%j.out
#SBATCH --error=multiweight_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=CHANGE_ME@example.com

# ============================================================
# Multi-weight experiment on ClusterUY
# Tests weight factors [1.5, 2.0, 3.0, 4.0, 5.0] across all 4 models
#
# Ref: https://www.cluster.uy/ayuda/como_ejecutar/
# Ref: https://www.cluster.uy/ayuda/tips/
# ============================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# ---- 1. Activate conda environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# ---- 2. Set project paths ----
PROJECT_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo"
cd "$PROJECT_DIR" || { echo "ERROR: Project dir not found"; exit 1; }
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ---- 3. Verify GPU is available ----
echo ""
echo "GPU info:"
nvidia-smi
echo ""

# ---- 4. Run the multi-weight experiment ----
echo "Starting multi-weight experiment..."
echo "Weight factors: 1.5, 2.0, 3.0, 4.0, 5.0"
echo "Using all 25 prompts"
echo ""

python scripts/run_experiment.py \
    --experiment multi_weight \
    --weights "1.5,2.0,3.0,4.0,5.0" \
    --prompts all \
    --no-plots

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "========================================"
