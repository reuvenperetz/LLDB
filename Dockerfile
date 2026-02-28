# ---- Base image ----
# CUDA runtime + Ubuntu 22.04, then install Python 3.8
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git curl ca-certificates \
    ffmpeg libsm6 libxext6 \
    zip unzip \
    nano \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3.8-dev python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

# Make `python` → `python3.8`
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# ---- Virtual environment ----
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ---- Python tooling ----
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python \
    && pip install --upgrade pip setuptools wheel

# ---- Install your repo dependencies ----
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt
# Install non-torch deps first (torch/torchvision are pinned to older CUDA)
RUN grep -vE '^(torch|torchvision)==' /tmp/requirements.txt > /tmp/requirements.no_torch.txt \
    && pip install -r /tmp/requirements.no_torch.txt \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2+cu121 \
        torchvision==0.17.2+cu121


# ---- Workspace ----
WORKDIR /workspace
COPY . /workspace

CMD ["/bin/bash"]
