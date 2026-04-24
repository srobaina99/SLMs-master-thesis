---
date: 2026-04-17T18:02:10+0000
researcher: Santiago Robaina
git_commit: a030a47842dba69f834f11b714a974c1100a452d
branch: feature/refactor
repository: SLMs-master-thesis
topic: "ClusterUY Container Setup — P100 GPU Rebuild"
tags: [clusteruy, singularity, docker, slurm, llama-cpp-python, cuda, p100]
status: work_in_progress
last_updated: 2026-04-17
last_updated_by: Santiago Robaina
type: implementation_strategy
---

# Handoff: ClusterUY container setup — llama-cpp-python rebuild for P100 GPU

## Task(s)

Setting up the experiment container on ClusterUY so the multi-weight experiment
(`run_multiweight.sh`) can run on P100 GPU nodes.

Status of each subtask:

- ✅ **Docker image initially built & pushed** (`srobaina99/experiment-cuda:v1`) — had
  prebuilt llama-cpp-python wheel from `abetlen.github.io/whl/cu121` (targets sm_70+ only).
- ✅ **SLURM batch scripts created** for pulling image (`pull_image.sh`) and downloading
  GGUF models (`download_models.sh`). Both working.
- ✅ **Singularity image pulled** to cluster (`~/slm-thesis.sif`, ~6 GB). Required running
  the pull as a SLURM batch job because the login node kills long-running processes and
  interactive sessions die with SSH drops.
- ✅ **GGUF models downloaded** to `~/SLMs-master-thesis/Tesis/Codigo/models/gguf/` (all 4
  files, total ~3.7 GB) via `download_models.sh` as a batch job.
- ✅ **Smoke test identified the P100 incompatibility** — the prebuilt llama-cpp-python
  wheel crashes with `CUDA error` at inference time. Tesla P100 is sm_60, wheels are sm_70+.
- 🚧 **Dockerfile rewritten** to build llama-cpp-python from source with
  `CMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89` on the `devel` base image. Not yet built
  or pushed — **this is the next action the user must take on their local Mac**.
- ⏳ **Re-pull image on cluster** (new tag `v2-p100`) — pending.
- ⏳ **Re-run smoke test** — pending.
- ⏳ **Submit full multi-weight experiment** via `run_multiweight.sh` — pending.

## Critical References

- `Tesis/Codigo/scripts/clusteruy/CLUSTERUY_GUIDE.md` — step-by-step cluster workflow (slightly stale; interactive-session notes no longer fully accurate given what we learned about login-node process killing).
- `Tesis/Codigo/scripts/clusteruy/Dockerfile` — **updated** to compile llama-cpp-python from source for P100. **Must be rebuilt locally and re-pushed.**
- `Tesis/CLAUDE.md` — project overview, model registry, experiment design.
- Official ClusterUY docs: https://cluster.uy/ayuda/singularity/, https://cluster.uy/ayuda/como_ejecutar/

## Recent changes

All on `feature/refactor` branch, pushed to origin:

- `6011c3b`: Added `Tesis/Codigo/scripts/clusteruy/pull_image.sh` — SLURM batch wrapper that
  runs `singularity pull` on a compute node (login node was killing the process).
- `a0c2a76`: Added `Tesis/Codigo/scripts/clusteruy/download_models.sh` — downloads all 4
  GGUF models from HuggingFace directly on the cluster (avoids scp from local).
- `e51d2d1`: Made `download_models.sh` submittable via `sbatch` (added SBATCH headers).
- `667e73e`: Fixed `download_models.sh` path resolution. Under sbatch `$0` points to SLURM's
  spool copy, so relative paths broke. Now uses `$SLURM_SUBMIT_DIR` when set, else the
  script's own dir.
- `e478726`: Replaced `wget --show-progress` with `wget -nv` for CentOS 7 compatibility.
- `a030a47`: **Dockerfile rewrite** — switched to `nvidia/cuda:12.1.1-devel-ubuntu22.04`
  (provides nvcc), added cmake, set `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89"` and `FORCE_CMAKE=1`, installed
  llama-cpp-python with `--no-binary=llama-cpp-python` to force source compile. See
  `Tesis/Codigo/scripts/clusteruy/Dockerfile:1-38`.

## Learnings

- **Login node kills long-running processes.** `singularity pull` of a 6GB image unpacks
  ~13GB on NFS and the login node's cgroup reaper kills it silently. `nohup` + `disown`
  does not help. Must run on compute node via `srun --pty` or as a `sbatch` batch job.
- **SSH disconnects kill interactive srun sessions.** A long pull via `srun --pty bash`
  dies when SSH drops. `tmux`/`screen` are not installed on ClusterUY login. The robust
  answer is **always submit as an sbatch batch job**.
- **NFS is extremely slow for small-file unpacks.** Layer unpacks of the Singularity image
  took ~2 hours on the login node vs minutes on a compute node. Compute node is dramatically
  faster — another reason to batch-submit pulls.
- **ClusterUY GPU is Tesla P100 (sm_60).** Verified on node19 via `nvidia-smi` and
  `torch.cuda.get_device_capability(0)` → `(6, 0)`. Prebuilt `llama-cpp-python` wheels from
  `abetlen.github.io/whl/cu121` target sm_70+, causing `ggml_cuda` to abort at inference.
  PyTorch is fine because its wheels cover sm_60.
- **Don't use relative `$0` paths in sbatch scripts.** Under sbatch, `$0` is the spooled
  copy in `/var/spool/slurm/`, not the submit dir. Use `$SLURM_SUBMIT_DIR` (the dir where
  sbatch was run from) for path resolution.
- **CentOS 7's wget lacks `--show-progress`.** Use `-nv` instead.
- **Docker Hub image naming was inconsistent.** Originally the scripts referenced
  `slm-thesis:latest` but the actual push was `experiment-cuda:v1`. All scripts now accept
  the image ref as an arg (default still `experiment-cuda:v1`, but next pull will use
  `experiment-cuda:v2-p100`).
- **Home directory accumulated stale singularity temp dirs** (~82 GB in
  `~/singularity_tmp/`) from failed pull attempts. Safe to delete all of them before next
  pull. User has 300GB home quota.

## Artifacts

- `Tesis/Codigo/scripts/clusteruy/Dockerfile` — **rewritten**, awaiting local build.
- `Tesis/Codigo/scripts/clusteruy/pull_image.sh` — sbatch pull script (works, accepts
  `<user/image:tag>` as arg, defaults to `srobaina99/experiment-cuda:v1`).
- `Tesis/Codigo/scripts/clusteruy/download_models.sh` — sbatch GGUF download script (done).
- `Tesis/Codigo/scripts/clusteruy/build_and_push.sh` — unchanged; still targets
  `slm-thesis:latest`. If reused, user must override or use `docker build/push` manually
  with the correct tag (see next steps).
- `Tesis/Codigo/scripts/clusteruy/run_multiweight.sh` — **unmodified**. Still has
  placeholder `CHANGE_ME@example.com` for the email. Must be updated before the full
  experiment is submitted.
- `Tesis/Codigo/scripts/clusteruy/CLUSTERUY_GUIDE.md` — has local uncommitted modifications
  from before this session (M status in git). Not updated to reflect what we learned this
  session. Consider updating to document (a) the P100 requirement, (b) the need to batch-
  submit pulls, (c) the correct image tag.

On the cluster (not in git):

- `~/slm-thesis.sif` — currently based on the **old** v1 image. Must be deleted before the
  re-pull.
- `~/singularity_tmp/` — large stale temp dirs from failed pulls. Safe to delete.
- `~/SLMs-master-thesis/` — cloned repo on cluster, on branch `feature/refactor`.
- `~/SLMs-master-thesis/Tesis/Codigo/models/gguf/*.gguf` — all 4 GGUF files present.

## Action Items & Next Steps

**Immediate (user on local Mac — the long pole):**

1. Rebuild the Docker image with the updated Dockerfile, using a new tag so the cluster
   forces a fresh pull:
   ```bash
   cd ~/Desktop/SLMs-master-thesis/Tesis/Codigo
   docker build --platform=linux/amd64 -t srobaina99/experiment-cuda:v2-p100 scripts/clusteruy/
   docker push srobaina99/experiment-cuda:v2-p100
   ```
   Expect 30-60 minutes. Final image will be ~8-9 GB because `devel` base is larger than
   `runtime`.

**Then on ClusterUY:**

2. Clean up stale state:
   ```bash
   rm -f ~/slm-thesis.sif
   rm -rf ~/singularity_tmp
   ```

3. Pull the new image as a batch job:
   ```bash
   cd ~/SLMs-master-thesis && git pull
   sbatch Tesis/Codigo/scripts/clusteruy/pull_image.sh srobaina99/experiment-cuda:v2-p100
   squeue -u $USER
   tail -f pull_image_*.out
   ```

4. Smoke test on a GPU compute node:
   ```bash
   interactivo -gpun
   cd ~/SLMs-master-thesis/Tesis/Codigo
   singularity exec --nv --bind $(pwd):/workspace ~/slm-thesis.sif \
       python /workspace/scripts/run_experiment.py \
       --experiment multi_weight --weights "1.5" --prompts 1 --model Qwen3 --no-plots
   ```
   Expected: Qwen3 loads, generates output, and completes without `CUDA error`.

5. Update email in `Tesis/Codigo/scripts/clusteruy/run_multiweight.sh`:
   ```bash
   sed -i 's/CHANGE_ME@example.com/<real-email>/' \
       Tesis/Codigo/scripts/clusteruy/run_multiweight.sh
   ```

6. Submit the full multi-weight experiment:
   ```bash
   cd ~/SLMs-master-thesis/Tesis/Codigo
   sbatch scripts/clusteruy/run_multiweight.sh
   ```

**Follow-up (not blocking):**

- Consider updating `CLUSTERUY_GUIDE.md` with the P100 requirement, the "use sbatch for
  pulls" guidance, and the correct image tag. There are also pre-existing uncommitted
  modifications to that file from earlier (unstaged `M`) — review whether to commit or
  discard.
- If the final image size is painful, consider a multi-stage Docker build that compiles in
  `devel` and copies the wheels/binaries into a `runtime` final stage. Deferred — not worth
  the complexity until we know the current approach works.

## Other Notes

- **Git remote**: https://github.com/srobaina99/SLMs-master-thesis.git on branch
  `feature/refactor`. All commits pushed.
- **Main branch for PRs**: `mac-backup`. We have not touched it.
- The multi-weight experiment tests 4 models × 5 weight factors × 25 prompts = 500 runs.
  See `Tesis/Codigo/scripts/clusteruy/run_multiweight.sh` for the exact command and SLURM
  resources (12h, 1 GPU, 8 CPUs, 32 GB RAM, normal partition + gpu QoS).
- There are pre-existing untracked directories in the repo (`.claude/agents/`,
  `.claude/commands/`, `BEA2026/predictions/closed 2/`, a `Presentación grupo fing ceibal/`
  dir, and two un-tracked CSV/JSON result files in `Tesis/Codigo/results/multi/`). None of
  these were touched this session but are visible in `git status`.
- The user's Docker Hub account has only one repo: `srobaina99/experiment-cuda`. Image
  naming has been inconsistent between docs (`slm-thesis:latest`) and reality
  (`experiment-cuda:v1`). Standardizing on `experiment-cuda:vX` going forward.
