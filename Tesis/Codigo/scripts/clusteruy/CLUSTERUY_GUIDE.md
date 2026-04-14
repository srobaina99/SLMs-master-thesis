# Running Experiments on ClusterUY

Step-by-step guide to run the multi-weight experiment on ClusterUY.

**Official documentation**: https://www.cluster.uy/ayuda/

## Prerequisites

- An active ClusterUY account ([register here](https://www.cluster.uy/registro/))
- Your SSH key pair submitted during registration
- UdelaR students need written endorsement from a faculty supervisor

Ref: [Política y costo de uso](https://www.cluster.uy/ayuda/politica_uso/)

---

## Step 1: Connect to ClusterUY

No VPN required. Authentication is via SSH key pair only.

```bash
ssh usuario@login.cluster.uy
```

Ref: [Cómo conectarse](https://www.cluster.uy/ayuda/como_conectarse/)

---

## Step 2: Clone the repository

The login node can be used for file management tasks like cloning.

```bash
git clone -b feature/refactor https://github.com/srobaina99/SLMs-master-thesis.git
```

> **Note**: Use HTTPS for public repos. SSH (`git@github.com:...`) requires your
> cluster SSH key to be added to your GitHub account.

Ref: [Utilización de repositorios GIT](https://www.cluster.uy/ayuda/git/)

---

## Step 3: Set up the environment

Running installations on the login node is **prohibited**. Request an interactive
session first. The default `interactivo` gives 30 minutes, which is not enough
for the full setup. Request 1 hour instead.

Ref: [Primeros pasos](https://www.cluster.uy/ayuda/primeros_pasos/),
[Cómo ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)

```bash
srun --time=1:00:00 --partition=normal --qos=normal --pty bash -l
```

Then run the setup script:

```bash
cd ~/SLMs-master-thesis/Tesis/Codigo
bash scripts/clusteruy/setup_env.sh
```

This script:

1. **Installs Miniconda** (py310_23.1.0-1 — the last version compatible with
   CentOS 7's glibc 2.17)
2. **Creates a conda environment** `thesis` with Python 3.10
3. **Installs PyTorch** with CUDA 12.1 support (plain `pip install torch`
   installs CPU-only)
4. **Installs llama-cpp-python** from a prebuilt CUDA wheel (compiling from
   source requires C++17 / GCC 8+ and takes over 30 minutes)
5. **Installs compiled packages** (pandas, numpy, matplotlib) via conda to avoid
   CentOS 7 build toolchain issues
6. **Installs remaining packages** (textstat, transformers, etc.) via pip
7. **Verifies** all imports work

### If the session times out

Everything installed to `$HOME` persists between sessions (home directory is
persistent NFS storage with 300 GB quota). Just get a new interactive session
and pick up where you left off.

Check quota with: `quota -gvs`

Ref: [Tips y buenas prácticas](https://www.cluster.uy/ayuda/tips/)

---

## Step 4: Interactive test

Before submitting a batch job, verify everything works with a quick interactive
GPU session.

```bash
interactivo -gpun
```

This gives a 30-minute session on the normal partition with GPU access.

```bash
source ~/.bashrc
conda activate thesis
cd ~/SLMs-master-thesis/Tesis/Codigo
python scripts/run_experiment.py \
    --experiment multi_weight \
    --weights "1.5,2.0,3.0,4.0,5.0" \
    --prompts 3 \
    --model Qwen3 \
    --no-plots
```

This runs a minimal test: 1 model, 3 prompts, 5 weights (15 runs). If it
completes without errors, the full experiment is ready.

Ref: [Cómo ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)

---

## Step 5: Submit the full experiment

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

| Parameter        | Value          | Reason                                    |
|-----------------|----------------|-------------------------------------------|
| `--partition`   | normal         | Guaranteed resources (not preemptible)     |
| `--qos`         | gpu            | Required for GPU jobs; max 4 GPUs, 3 days |
| `--gres`        | gpu:1          | One GPU for inference                      |
| `--cpus-per-task` | 8            | Enough for data loading and text eval      |
| `--mem`         | 32768 (32 GB)  | Headroom for model loading                 |
| `--time`        | 12:00:00       | 4 models x 5 weights x 25 prompts         |
| `--no-plots`    | (flag)         | No display on cluster; generate locally    |

### What it runs

- **4 models**: Phi3, Qwen2, Qwen3, TinyLlama
- **5 weight factors**: 1.5, 2.0, 3.0, 4.0, 5.0
- **25 prompts**: Full STANDARD_PROMPTS set
- **Total**: 500 experiment runs

Ref: [Recursos disponibles](https://www.cluster.uy/ayuda/recursos_disponibles/)

---

## Step 6: Monitor the job

```bash
squeue -u $USER --long           # check job status
tail -f multiweight_<jobid>.out  # follow output (from login node)
scancel <jobid>                  # cancel if needed
seff <jobid>                     # check resource usage after completion
```

Ref: [Comandos útiles](https://www.cluster.uy/ayuda/comandos_utiles/)

---

## Step 7: Download results

Results are saved to `~/SLMs-master-thesis/Tesis/Codigo/results/`. From your
local machine:

```bash
scp -P 10022 usuario@cluster.uy:~/SLMs-master-thesis/Tesis/Codigo/results/*.csv ./results/
```

Or with rsync:

```bash
rsync -arvz -e "ssh -p 10022" \
    usuario@cluster.uy:~/SLMs-master-thesis/Tesis/Codigo/results/ \
    ./Tesis/Codigo/results/
```

Use port 10022 to avoid consuming login node bandwidth.

Ref: [Tips y buenas prácticas](https://www.cluster.uy/ayuda/tips/)

---

## Important notes

- **No backups**: The cluster does not back up any user data.
  Ref: [Política de uso](https://www.cluster.uy/ayuda/politica_uso/)
- **Inactivity policy**: After 4 months without running a SLURM job, your
  account is deactivated. After 6 months total, data is permanently deleted.
- **Home directory**: 300 GB quota, persistent NFS storage. Check with `quota -gvs`.
- **Scratch** (`/scratch/$USER`): 300 GB SSD per node, fast but node-local.
  Clean up after jobs. Reserve with `--tmp=xxxG`.
- **Login node**: Only for file management and job submission. All computation
  (including `pip install`, compilation) must go through interactive or batch jobs.
