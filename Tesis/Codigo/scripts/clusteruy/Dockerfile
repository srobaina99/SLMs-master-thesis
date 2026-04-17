# Use devel image so nvcc is available to compile llama-cpp-python from source.
# Runtime image lacks nvcc. --platform=linux/amd64 for M2 Mac build compatibility.
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch with CUDA 12.1 (prebuilt wheels cover sm_60+)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Build llama-cpp-python from source with CUDA for common GPU archs:
# 60 = P100 (ClusterUY), 70 = V100, 75 = T4, 80 = A100, 86 = A10/RTX30, 89 = RTX40
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir --no-binary=llama-cpp-python llama-cpp-python

RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    tqdm==4.66.1 \
    textstat==0.7.4 \
    nltk==3.9.1 \
    setuptools

RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

WORKDIR /workspace
