# 1. CRITICAL: Added --platform=linux/amd64 for M2 compatibility
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. Added build-essential (small but prevents pip install failures)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 3. PyTorch Install (This will take ~5-8 mins to download 2GB+)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 4. llama-cpp-python (This is now FAST because it's a prebuilt wheel)
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 5. Remaining deps
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    tqdm==4.66.1 \
    textstat==0.7.4 \
    nltk==3.9.1 \
    setuptools

RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

WORKDIR /workspace