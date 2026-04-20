---
date: 2026-04-20T11:19:02-0300
researcher: Santiago Robaina
git_commit: a030a47842dba69f834f11b714a974c1100a452d
branch: feature/refactor
repository: SLMs-master-thesis
topic: "Lean Docker image for multi-weight experiment on ClusterUY"
tags: [research, docker, clusteruy, llama-cpp-python, dependencies, image-size]
status: complete
last_updated: 2026-04-20
last_updated_by: Santiago Robaina
---

# Research: Lean Docker image for multi-weight experiment on ClusterUY

**Date**: 2026-04-20T11:19:02-0300
**Researcher**: Santiago Robaina
**Git Commit**: a030a47842dba69f834f11b714a974c1100a452d
**Branch**: feature/refactor
**Repository**: SLMs-master-thesis

## Research Question

What is the minimum set of dependencies required to run the multi-weight experiment (`python scripts/run_experiment.py --experiment multi_weight --weights "1.5,2.0,3.0,4.0,5.0" --prompts all --no-plots`) on ClusterUY, so that the Docker image for Singularity can be as small as possible?

## Summary

**The experiment needs only 5 Python packages** at runtime: `llama-cpp-python`, `pandas`, `tqdm`, `textstat`, `nltk`. Every other package in `requirements.txt` — including `torch`, `transformers`, `accelerate`, `sentencepiece`, `safetensors`, `matplotlib`, `seaborn`, `pyarrow`, `ctransformers`, `IPython`, `jupyter`, `Markdown` — is either never imported, only used by dead legacy code, or gated behind the plotting branch that `--no-plots` disables.

**PyTorch is not used anywhere in `src/`.** The current Dockerfile installs `torch torchvision torchaudio` from the CUDA 12.1 index, adding ~5 GB to the image for zero benefit in this experiment path.

**None of the three NLTK corpora currently pre-downloaded (`punkt`, `averaged_perceptron_tagger`, `wordnet`) are actually read at runtime** by the multi-weight path. The only NLTK corpus actually exercised is `cmudict`, which `textstat` self-downloads internally on first readability-metric call.

**Moving to a multi-stage build on `nvidia/cuda:12.1.1-runtime-ubuntu22.04` without PyTorch brings the image from the current ~13–15 GB down to an expected ~3–3.5 GB uncompressed** (~1.5–2 GB compressed / Singularity pull).

## Detailed Findings

### Runtime Import Graph (what's actually loaded)

For `--experiment multi_weight --no-plots`, the full module-level import graph reachable at runtime is:

```
scripts/run_experiment.py
 └── src.framework.core.experiment_runner
      ├── src.framework.core.data_models                (→ pandas)
      ├── src.framework.experiments.factorial_experiment (→ pandas, tqdm)
      │    ├── src.framework.models.__init__
      │    │    ├── src.framework.models.base_model
      │    │    ├── src.framework.models.llamacpp_base   (→ llama_cpp)
      │    │    ├── src.framework.models.phi3_llamacpp_wrapper
      │    │    ├── src.framework.models.qwen2_llamacpp_wrapper
      │    │    ├── src.framework.models.qwen3_llamacpp_wrapper
      │    │    │    └── src.framework.models.beam_search_generator
      │    │    └── src.framework.models.tinyllama_llamacpp_wrapper
      │    ├── src.text_complexity.text_evaluator        (→ textstat, nltk)
      │    └── src.framework.experiments.experiment_configs
      └── src.framework.experiments.experiment_configs
```

Every third-party import at module level across that graph:

| File | Line | Import | Package |
|------|------|--------|---------|
| `src/framework/core/data_models.py` | 9 | `import pandas as pd` | pandas |
| `src/framework/experiments/factorial_experiment.py` | 11 | `import pandas as pd` | pandas |
| `src/framework/experiments/factorial_experiment.py` | 13 | `from tqdm import tqdm` | tqdm |
| `src/framework/models/llamacpp_base.py` | 32 | `from llama_cpp import Llama` (try/except) | llama-cpp-python |
| `src/text_complexity/text_evaluator.py` | 1 | `import textstat` | textstat |
| `src/text_complexity/text_evaluator.py` | 6-9 | `import nltk` (try/except) | nltk |

That's the complete list.

### Packages NOT Needed

None of the following are imported anywhere in `src/` (confirmed by exhaustive grep):

- **`torch`, `torchvision`, `torchaudio`** — zero `import torch` statements in `src/`. Only appears in `scripts/clusteruy/test_gpu.py` (a 6-line GPU probe script — trivially replaceable) and `scripts/legacy_tests/benchmark_qwen3_llamacpp.py` (legacy).
- **`transformers`** — only in `legacy/` dir, never imported by anything in `src/`.
- **`accelerate`, `sentencepiece`, `safetensors`** — legacy Transformers-backend leftovers.
- **`ctransformers`** — requirements.txt mentions it as "Optional: For quantized models (TinyLlama)" but all models now use llama-cpp-python.
- **`matplotlib`, `seaborn`** — used only in `results/visualize_results.py`, which is dynamically loaded via `importlib.util.spec_from_file_location` inside `ExperimentRunner._generate_visualizations()` (at `src/framework/core/experiment_runner.py:267-296`). That method is gated by `if generate_plots:` checks at lines 87, 130, 257. With `--no-plots`, it is never called.
- **`pyarrow`** — `ExperimentDataManager.save_to_parquet()` at `src/framework/core/data_models.py:236-244` uses it, but `save_results()` only calls `save_to_csv` and `export_to_csv_specification_format`, never `save_to_parquet`.
- **`IPython`, `jupyter`, `Markdown`** — legacy notebook leftovers, not in any `src/` file.

### NLTK Corpus Audit

The Dockerfile line at `Tesis/Codigo/scripts/clusteruy/Dockerfile:36` runs:
```
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

**Runtime usage analysis** of the only NLTK-touching file, `src/text_complexity/text_evaluator.py`:

| Corpus | Guard check at import | Used at runtime in multi-weight path? |
|--------|----------------------|----------------------------------------|
| `punkt` | Yes, line 16-18 (`nltk.data.find('tokenizers/punkt')`) | No — `word_tokenize` is only called from `extract_content_words()`, which is only reached by beam-search (Qwen3), not multi-weight |
| `averaged_perceptron_tagger` | Yes, line 21-23 | No — `pos_tag` only called from `extract_content_words()` |
| `wordnet` | Yes, line 26-28 | No — `wordnet` is imported at line 8 but never used anywhere in the file |
| `cmudict` | No — but **textstat downloads it itself** internally on first use | **Yes** — every `textstat.flesch_kincaid_grade`, `gunning_fog`, `spache_readability`, `difficult_words` call reads `cmudict` via `get_cmudict()` |

**NLTK 3.9.1 caveat**: In NLTK 3.9+ `word_tokenize` actually looks for `tokenizers/punkt_tab/english/` — a different corpus than `punkt`. The guard at `text_evaluator.py:16` checks `tokenizers/punkt` which is the old path, so the guard is satisfied by downloading `punkt`, but if `word_tokenize` were ever actually called it would fail with a `punkt_tab` LookupError. Since `word_tokenize` is never reached in the multi-weight path, this is not a real problem — but the guard + download is dead weight.

**The only NLTK corpus actually read from disk during a multi-weight run is `cmudict`**, used by textstat for syllable counting. Textstat downloads it itself if missing.

**Recommendation for the Dockerfile**: pre-download `cmudict` to avoid a network call on the compute node (where internet may be unavailable), and optionally keep `punkt` + `averaged_perceptron_tagger` to satisfy the import-time guards without errors. `wordnet` is pure dead weight — remove it.

### llama-cpp-python Build System Requirements

**Build-time (stage 1 only):**
- `python3.10`, `python3.10-dev` *(currently missing from Dockerfile — needed for `Python.h` when compiling the C extension)*, `python3-pip`
- `build-essential`, `cmake`
- `ninja-build` (optional, ~2× faster builds)
- `git` — **not required**; `pip install llama-cpp-python` uses the PyPI sdist (vendored `llama.cpp` included), no `git clone`

**Runtime (stage 2):**
- `python3.10`, `python3-pip`
- Dynamic libraries: `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` (provided by `nvidia/cuda:12.1.1-runtime-ubuntu22.04`)
- NOT needed: `nvcc`, CUDA headers, `cuda-cudart-dev`, any `-devel` content

**Image size comparison (from NVIDIA's GitLab Dockerfiles for `12.1.1/ubuntu2204`):**

| Tag | Uncompressed | Contains |
|-----|-------------|----------|
| `nvidia/cuda:12.1.1-base-ubuntu22.04` | ~240 MB | `cuda-cudart` only — **no cuBLAS, not enough** |
| `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | ~2.5–2.9 GB | Base + cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NPP, nccl |
| `nvidia/cuda:12.1.1-devel-ubuntu22.04` | ~6.5–7.0 GB | Runtime + nvcc, headers, static libs |

`base` is insufficient because llama-cpp-python links cuBLAS. `runtime` has everything llama-cpp-python needs at run time. `devel` is only needed for the builder stage.

### Projected Image Sizes

| Strategy | Uncompressed | Compressed (pull) | vs. current |
|----------|-------------|-------------------|-------------|
| **Current single-stage devel + PyTorch** (what `Dockerfile` will produce if rebuilt) | ~13–15 GB | ~6–7 GB | baseline |
| Multi-stage with `-runtime` base, **keeping PyTorch** | ~8–9 GB | ~3.5–4.5 GB | –40% |
| **Multi-stage with `-runtime` base, no PyTorch** (recommended) | ~3–3.5 GB | ~1.5–2 GB | **–75%** |

Breakdown of the recommended ~3 GB image:
- `nvidia/cuda:12.1.1-runtime-ubuntu22.04`: ~2.5 GB
- Python 3.10 + pip: ~100 MB
- llama-cpp-python wheel (kernels for archs 60;70;75;80;86;89): ~150–300 MB
- pandas + tqdm + textstat + nltk: ~200 MB
- NLTK corpora (cmudict + optional punkt/avp): ~30–50 MB

## Code References

### Runtime path
- `Tesis/Codigo/scripts/run_experiment.py:107-112` — `run_multi_weight_experiment` entry
- `Tesis/Codigo/src/framework/core/experiment_runner.py:257-296` — plot gating via `generate_plots`; dynamic import of `visualize_results.py` only when plots enabled
- `Tesis/Codigo/src/framework/experiments/factorial_experiment.py:11-13` — pandas + tqdm module-level imports
- `Tesis/Codigo/src/framework/models/llamacpp_base.py:32-35` — llama_cpp guarded import
- `Tesis/Codigo/src/text_complexity/text_evaluator.py:1` — textstat import
- `Tesis/Codigo/src/text_complexity/text_evaluator.py:6-28` — nltk imports + guard downloads

### Data files read at runtime
- `Tesis/Codigo/data/vocabularies/filtered_starters_vocab.txt` — A1 vocabulary, loaded by `BaseModelWrapper._load_target_vocabulary()` at `src/framework/models/base_model.py:103-115`
- `Tesis/Codigo/models/gguf/Qwen3-0.6B-Q4_0.gguf` — Qwen3 weights
- `Tesis/Codigo/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf` — Qwen2 weights
- `Tesis/Codigo/models/gguf/Phi-3-mini-4k-instruct-q4.gguf` — Phi-3 weights
- `Tesis/Codigo/models/gguf/tinyllama-1.1b-chat-v1.0.Q4_0.gguf` — TinyLlama weights

### Current Dockerfile
- `Tesis/Codigo/scripts/clusteruy/Dockerfile:1-38` — single-stage devel build, contains PyTorch (dead), missing `python3.10-dev`, installs 3 unused NLTK corpora

## Architecture Insights

- **The refactor on `feature/refactor` fully replaced the Transformers backend with llama-cpp-python for all four models.** The model wrapper hierarchy (`BaseModelWrapper → LlamaCppBaseWrapper → four concrete wrappers`) has zero torch/transformers imports. `requirements.txt` was not updated to reflect this — it still lists the full legacy Transformers stack.
- **Plotting is cleanly isolated.** `ExperimentRunner._generate_visualizations()` uses `importlib` to load `results/visualize_results.py` dynamically, so matplotlib/seaborn are never pulled into the main import graph. The `--no-plots` flag is a true kill switch for the plotting subsystem.
- **Textstat transitively uses NLTK cmudict.** This is the one NLTK dependency that's actually load-bearing at runtime, and it's currently not pre-downloaded — textstat silently downloads it on first use, which relies on the compute node having internet. If ClusterUY compute nodes are air-gapped, this is a latent failure mode. Pre-downloading `cmudict` at build time is cheap insurance.
- **The multi-weight experiment does NOT use beam search.** Beam search (and therefore `extract_content_words` / `word_tokenize` / `pos_tag`) is only invoked by `Qwen3LlamaCppWrapper.generate_with_beam_search()`, which the multi-weight path never calls.

## Recommendation: Lean Multi-Stage Dockerfile

```dockerfile
# ============================================================
# Stage 1: builder — compile llama-cpp-python for P100-aware GPU archs
# ============================================================
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89"
ENV FORCE_CMAKE=1

# Build llama-cpp-python as a reusable wheel in /wheels
RUN pip wheel --no-cache-dir --no-deps --no-binary=llama-cpp-python \
      -w /wheels llama-cpp-python

# ============================================================
# Stage 2: runtime — only what's needed to RUN the experiment
# ============================================================
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install the prebuilt CUDA-enabled llama-cpp-python wheel from stage 1
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Minimal runtime deps for the multi-weight experiment
RUN pip install --no-cache-dir \
      pandas==2.1.4 \
      tqdm==4.66.1 \
      textstat==0.7.4 \
      nltk==3.9.1

# Pre-populate NLTK data so no network call is needed on the compute node.
# cmudict is the one actually used at runtime (via textstat).
# punkt/averaged_perceptron_tagger satisfy the module-level guards in text_evaluator.py.
# wordnet is imported but never used — omit.
RUN python -m nltk.downloader -d /usr/share/nltk_data cmudict punkt averaged_perceptron_tagger
ENV NLTK_DATA=/usr/share/nltk_data

WORKDIR /workspace
```

**Changes vs. current Dockerfile:**
1. Multi-stage: builder uses `-devel`, runtime uses `-runtime` → drops nvcc/headers from shipped image.
2. **Drop PyTorch entirely** — not imported anywhere in the multi-weight path.
3. Add `python3.10-dev` to the builder stage (needed for Python.h during C extension compile).
4. Drop `git` — not needed.
5. Use `pip wheel` in stage 1 so the wheel can be copied to stage 2 without a rebuild.
6. Drop `wordnet` from NLTK downloads; add `cmudict` (the one actually used).
7. Drop `setuptools` (not needed for runtime).

## Follow-up Actions (not in scope for this research, but implied)

1. Optionally replace `scripts/clusteruy/test_gpu.py` (imports torch just to print GPU info) with `nvidia-smi` or a llama-cpp-python-based probe, so it works in the lean image.
2. Consider marking `requirements.txt` as development-only (local venv) and creating `requirements-cluster.txt` or similar with only the 5 runtime packages.

## Related Research

None yet — this is the first research doc in `thoughts/shared/research/`.

## Open Questions

- Does the ClusterUY compute node have internet access, or are compute nodes fully air-gapped? The handoff mentions "login node kills long-running processes" but doesn't specify compute-node network policy. If compute nodes are offline, pre-downloading `cmudict` is mandatory, not optional.
- Does textstat's `get_cmudict()` tolerate a pre-populated `cmudict` in `NLTK_DATA` without trying to re-download? (Likely yes, since `nltk.download` is a no-op if the corpus is already present, but worth verifying before relying on the offline assumption.)
