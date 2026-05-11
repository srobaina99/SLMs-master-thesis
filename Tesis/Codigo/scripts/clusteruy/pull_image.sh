#!/bin/bash
#SBATCH --job-name=pull_image
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --time=06:00:00
#SBATCH --output=pull_image_%j.out
#SBATCH --error=pull_image_%j.err

# ============================================================
# Pull the Singularity image on a compute node (login node kills
# long-running pulls, and interactive sessions die with SSH drops).
# Submit with: sbatch pull_image.sh <dockerhub_user/image:tag>
# Default image: srobaina99/experiment-cuda:v1
# ============================================================

IMAGE_REF="${1:-srobaina99/experiment-cuda:v1}"
SIF_PATH="$HOME/slm-thesis.sif"

mkdir -p "$HOME/singularity_tmp"
export SINGULARITY_TMPDIR="$HOME/singularity_tmp"
export TMPDIR="$HOME/singularity_tmp"

echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST  start: $(date)"
echo "Pulling docker://$IMAGE_REF -> $SIF_PATH"

rm -f "$SIF_PATH"
singularity pull --name "$SIF_PATH" "docker://$IMAGE_REF"

echo "Done: $(date)"
ls -lh "$SIF_PATH"
