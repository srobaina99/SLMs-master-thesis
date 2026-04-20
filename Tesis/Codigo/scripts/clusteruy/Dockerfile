# ============================================================
# Stage 1: builder — compile llama-cpp-python with CUDA support
# for P100 (sm_60) plus common newer archs.
#
# Uses the devel image because nvcc / CUDA headers are required
# to compile from source, but nothing from this stage ships.
# ============================================================
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Make the CUDA driver stubs discoverable at link time. The devel image ships
# /usr/local/cuda/lib64/stubs/libcuda.so, but:
#   (a) libggml-cuda.so has a DT_NEEDED entry for libcuda.so.1 (the SONAME of
#       the real driver), not libcuda.so, so we need a libcuda.so.1 symlink.
#   (b) The stubs dir isn't on ld's default search path; we add it via LDFLAGS
#       (-L and -rpath-link) so the linker can satisfy transitive deps when
#       producing the auxiliary mtmd binaries.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
ENV LDFLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

# 60 = P100 — the only GPU on ClusterUY. Multi-arch builds take 6× longer.
# If we ever target a non-P100 cluster we can add archs back here.
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60"
ENV FORCE_CMAKE=1

# Build llama-cpp-python as a reusable wheel so stage 2 just installs it.
RUN pip wheel --no-cache-dir --no-deps --no-binary=llama-cpp-python \
        -w /wheels llama-cpp-python

# ============================================================
# Stage 2: runtime — only what's needed to RUN the experiment.
#
# PyTorch, transformers, matplotlib, seaborn, pyarrow, ctransformers
# and friends are intentionally omitted: none of them are imported
# on the --experiment multi_weight --no-plots code path.
# ============================================================
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install the prebuilt CUDA-enabled llama-cpp-python wheel from stage 1.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Minimal runtime deps for --experiment multi_weight --no-plots.
RUN pip install --no-cache-dir \
        pandas==2.1.4 \
        tqdm==4.66.1 \
        textstat==0.7.4 \
        nltk==3.9.1

# Pre-populate NLTK data so no network call is needed on the compute node.
#   cmudict — actually read by textstat for flesch_kincaid / gunning_fog / spache
#   punkt + averaged_perceptron_tagger — satisfy import-time guards in text_evaluator.py
#   wordnet — imported but never called on this path, omitted.
RUN python -m nltk.downloader -d /usr/share/nltk_data \
        cmudict punkt averaged_perceptron_tagger
ENV NLTK_DATA=/usr/share/nltk_data

WORKDIR /workspace
