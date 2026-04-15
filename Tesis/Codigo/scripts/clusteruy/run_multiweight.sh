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
# Multi-weight experiment on ClusterUY (Singularity)
# Tests weight factors [1.5, 2.0, 3.0, 4.0, 5.0] across all 4 models
#
# Ref: https://www.cluster.uy/ayuda/como_ejecutar/
# Ref: https://www.cluster.uy/ayuda/singularity/
# ============================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# ---- 1. Set paths ----
PROJECT_DIR="$HOME/SLMs-master-thesis/Tesis/Codigo"
SIF_IMAGE="$HOME/slm-thesis.sif"

cd "$PROJECT_DIR" || { echo "ERROR: Project dir not found"; exit 1; }

# ---- 2. Verify GPU is available ----
echo ""
echo "GPU info:"
nvidia-smi
echo ""

# ---- 3. Run the multi-weight experiment ----
# --nv: expose NVIDIA GPU inside the container
# --bind: mount project directory into /workspace
echo "Starting multi-weight experiment..."
echo "Weight factors: 1.5, 2.0, 3.0, 4.0, 5.0"
echo "Using all 25 prompts"
echo ""

singularity exec --nv \
    --bind "$PROJECT_DIR":/workspace \
    "$SIF_IMAGE" \
    python /workspace/scripts/run_experiment.py \
        --experiment multi_weight \
        --weights "1.5,2.0,3.0,4.0,5.0" \
        --prompts all \
        --no-plots

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "========================================"
