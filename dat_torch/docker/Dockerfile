# Base image with CUDA 11.3 and cuDNN 8
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3.9-distutils \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Upgrade pip/setuptools/wheel
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Install PyTorch + torchvision (CUDA 11.3 build)
RUN python3.9 -m pip install \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN python3.9 -m pip install natten==0.14.6+torch1110cu113 \
 -f https://whl.natten.org/old

# Install remaining dependencies
RUN python3.9 -m pip install \
    numpy \
    timm \
    einops \
    PyYAML \
    yacs \
    termcolor

RUN alias python=python3.9

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]