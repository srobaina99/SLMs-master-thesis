# Running Experiments on ClusterUY

> **Deprecated workflow.** This guide documents the legacy `Tesis/Codigo/` experiment
> runner (`scripts/run_experiment.py`). For current Phase 1/2 experiments, use
> **[SLMs-experiments/docs/clusteruy.md](https://github.com/srobaina99/SLMs-experiments/blob/main/docs/clusteruy.md)**
> and the sbatch scripts in `SLMs-experiments/scripts/clusteruy/`.
>
> **Still maintained here:** `Dockerfile`, `pull_image.sh`, `download_models.sh`
> (container build and one-time cluster setup).

Step-by-step guide to run the **legacy** multi-weight experiment on ClusterUY using
Singularity containers.

**Official documentation**: https://www.cluster.uy/ayuda/

## Why Singularity?

ClusterUY runs CentOS 7 (glibc 2.17), which is too old for most modern Python
packages (PyTorch, llama-cpp-python, pandas). Singularity containers solve this
by running a modern Linux (Ubuntu 22.04) inside the container while using the
cluster's GPU drivers.

Ref: [Contenedores de Linux (Singularity)](https://www.cluster.uy/ayuda/singularity/)

## Prerequisites

- An active ClusterUY account ([register here](https://www.cluster.uy/registro/))
- Your SSH key pair submitted during registration
- UdelaR students need written endorsement from a faculty supervisor
- Docker installed on your **local machine** (for building the image)
- A Docker Hub account (https://hub.docker.com)

Ref: [Política y costo de uso](https://www.cluster.uy/ayuda/politica_uso/)

---

## Step 1: Build and push the Docker image (local machine)

This builds a lean container with Ubuntu 22.04, CUDA 12.1, llama-cpp-python
(compiled for P100 / sm_60), and minimal runtime deps (pandas, tqdm, textstat,
nltk). Run this on your local machine, not on ClusterUY.

```bash
cd Tesis/Codigo

# Log in to Docker Hub
docker login

# Build and push (takes 10-20 min, compiles llama-cpp-python with CUDA)
bash scripts/clusteruy/build_and_push.sh <your_dockerhub_username>
```

This creates and pushes `<your_dockerhub_username>/slm-thesis:latest`.

The Dockerfile is at `scripts/clusteruy/Dockerfile` — it installs only what the
experiment needs: llama-cpp-python (CUDA), pandas, tqdm, textstat, nltk.

---

## Step 2: Connect to ClusterUY

No VPN required. Authentication is via SSH key pair only.

```bash
ssh santiago.robaina@login.cluster.uy
```

Ref: [Cómo conectarse](https://www.cluster.uy/ayuda/como_conectarse/)

---

## Step 3: Clone the repository

The login node can be used for file management tasks like cloning.

```bash
git clone -b feature/refactor https://github.com/srobaina99/SLMs-master-thesis.git
```

Use HTTPS for public repos. SSH (`git@github.com:...`) requires your cluster SSH
key to be added to your GitHub account.

Ref: [Utilización de repositorios GIT](https://www.cluster.uy/ayuda/git/)

---

## Step 4: Pull the Singularity image

Submit as a batch job (the login node kills long-running pulls):

```bash
cd ~/SLMs-master-thesis
sbatch Tesis/Codigo/scripts/clusteruy/pull_image.sh <your_dockerhub_username>/slm-thesis:latest
```

This converts the Docker image to a Singularity `.sif` file in your home
directory. It only needs to be done once (~1.5 GB for the lean image).

Ref: [Contenedores de Linux](https://www.cluster.uy/ayuda/singularity/)

---

## Step 5: Interactive test

Before submitting a batch job, verify everything works with a quick interactive
GPU session.

```bash
interactivo -gpun
```

This gives a 30-minute session on the normal partition with GPU access.

```bash
cd ~/SLMs-master-thesis/Tesis/Codigo

singularity exec --nv \
    --bind $(pwd):/workspace \
    ~/slm-thesis.sif \
    python /workspace/scripts/run_experiment.py \
        --experiment multi_weight \
        --weights "1.5,2.0" \
        --prompts 2 \
        --model Qwen3 \
        --no-plots
```

- `--nv`: exposes the host NVIDIA GPU drivers inside the container
- `--bind`: mounts the project directory at `/workspace` inside the container

If this completes without errors, the full experiment is ready.

Ref: [Cómo ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)

---

## Step 6: Submit the full experiment

First, edit `scripts/clusteruy/run_multiweight.sh` and set your email:

```bash
#SBATCH --mail-user=your_actual_email@example.com
```

Then submit from the login node:

```bash
cd ~/SLMs-master-thesis/Tesis/Codigo
sbatch scripts/clusteruy/run_multiweight.sh
```

### Job configuration

| Parameter         | Value          | Reason                                    |
|-------------------|----------------|-------------------------------------------|
| `--partition`     | normal         | Guaranteed resources (not preemptible)     |
| `--qos`           | gpu            | Required for GPU jobs; max 4 GPUs, 3 days |
| `--gres`          | gpu:1          | One GPU for inference                      |
| `--cpus-per-task` | 8              | Enough for data loading and text eval      |
| `--mem`           | 32768 (32 GB)  | Headroom for model loading                 |
| `--time`          | 12:00:00       | 4 models x 5 weights x 25 prompts         |
| `--no-plots`      | (flag)         | No display on cluster; generate locally    |

### What it runs

- **4 models**: Phi3, Qwen2, Qwen3, TinyLlama
- **5 weight factors**: 1.5, 2.0, 3.0, 4.0, 5.0
- **25 prompts**: Full STANDARD_PROMPTS set
- **Total**: 500 experiment runs

Ref: [Recursos disponibles](https://www.cluster.uy/ayuda/recursos_disponibles/)

---

## Step 7: Monitor the job

```bash
squeue -u $USER --long           # check job status
tail -f multiweight_<jobid>.out  # follow output (from login node)
scancel <jobid>                  # cancel if needed
seff <jobid>                     # check resource usage after completion
```

Ref: [Comandos útiles](https://www.cluster.uy/ayuda/comandos_utiles/)

---

## Step 8: Download results

Results are saved to `~/SLMs-master-thesis/Tesis/Codigo/results/`. From your
local machine, use port 10022 to avoid consuming login node bandwidth:

```bash
scp -P 10022 santiago.robaina@cluster.uy:~/SLMs-master-thesis/Tesis/Codigo/results/*.csv ./results/
```

Or with rsync:

```bash
rsync -arvz -e "ssh -p 10022" \
    santiago.robaina@cluster.uy:~/SLMs-master-thesis/Tesis/Codigo/results/ \
    ./Tesis/Codigo/results/
```

Ref: [Tips y buenas prácticas](https://www.cluster.uy/ayuda/tips/)

---

## Important notes

- **No backups**: The cluster does not back up any user data.
- **Inactivity policy**: After 4 months without running a SLURM job, your
  account is deactivated. After 6 months total, data is permanently deleted.
- **Home directory**: 300 GB quota, persistent NFS storage. Check with `quota -gvs`.
- **Containers are read-only**: You cannot modify a pulled image. If you need to
  change dependencies, rebuild the Docker image locally and re-push/re-pull.
- **Login node**: Only for file management and job submission. All computation
  must go through interactive or batch jobs.

Ref: [Política de uso](https://www.cluster.uy/ayuda/politica_uso/)
