# FROM ubuntu:22.04
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# =========================
# Base environment
# =========================
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV PATH=/opt/conda/bin:$PATH

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    build-essential \
    patchelf \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    libglib2.0-0 \
    ffmpeg \
    megatools \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Install Miniconda
# =========================
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Disable base auto-activation
RUN conda config --system --set auto_activate_base false

# =========================
# Accept Anaconda ToS
# =========================
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# =========================
# Create Conda environment
# =========================
RUN conda create -y -n policy_switching python=3.10

# =========================
# Create required directories
# =========================
RUN mkdir -p \
    /root/.mujoco \
    /root/.d4rl \
    /root/moreldataset \
    /root/results \
    /root/moreldataset/datasets \
    /root/.d4rl/datasets

# =========================
# Install MuJoCo 2.1.0
# =========================
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    -O /tmp/mujoco210.tar.gz && \
    tar --no-same-owner -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco210.tar.gz

# =========================
# Environment variables
# =========================
ENV MUJOCO_DIR=/root/.mujoco
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV MUJOCO_PATH=/root/.mujoco/mujoco210
ENV D4RL_DATASET_DIR=/root/.d4rl
ENV MOREL_DATASET_DIR=/root/moreldataset
ENV MOREL_OUTPUT_DIR=/root/results
ENV D4RL_SUPPRESS_IMPORT_ERROR=1
ENV MUJOCO_GL=egl
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin
ENV PATH=/opt/conda/envs/policy_switching/bin:/opt/conda/bin:$PATH


# =========================
# Upgrade pip
# =========================
RUN pip install --upgrade pip

# =========================
# Check Python and pip versions
# =========================
RUN which python && python --version
RUN which pip && pip --version

# =========================
# Python dependencies
# =========================
# RUN pip install \
#     git+https://github.com/tinkoff-ai/d4rl@master#egg=d4rl \
#     tqdm==4.64.0 \
#     wandb \
#     mujoco-py==2.1.2.14 \
#     numpy==1.23.1 \
#     "gym[mujoco_py,classic_control]==0.23.0" \
#     --extra-index-url https://download.pytorch.org/whl/cu113 \
#     torch==1.11.0+cu113 \
#     pyrallis==0.3.1 \
#     "Cython<3" \
#     tabulate \
#     matplotlib \
#     pandas \
#     tensorboard \
#     tensorboardX \
#     seaborn \
#     pettingzoo \
#     stable_baselines3 \
#     patchelf

RUN pip install \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.11.0+cu113 \
    "gym[mujoco_py,classic_control]==0.23.0" \
    numpy==1.26.4 \
    tensorboardX \
    tensorflow \
    seaborn \
    pettingzoo \
    matplotlib \
    stable_baselines3 \
    git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl \
    patchelf \
    "cython<3"\
    wandb

# =========================
# Verify MuJoCo
# =========================
RUN python -c "import mujoco_py; print('mujoco_py import OK')"

# =========================
# Workspace
# =========================
WORKDIR /workspace

# =========================
# Default command
# =========================
CMD ["/bin/bash"]

