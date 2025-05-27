FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
LABEL maintainer="junghyup.lee@yonsei.ac.kr"

ARG DEBIAN_FRONTEND=noninteractive

# ---------- system packages ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git vim zsh tmux htop curl wget \
        locales language-pack-en \
        ffmpeg libsm6 libxext6 libgtk2.0-dev \
        libpng-dev libfreetype6-dev libjpeg8-dev \
        graphviz xdg-utils && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    ldconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---------- Python layer ----------
RUN pip install --no-cache-dir \
        numpy scipy pandas scikit-learn scikit-image easydict \
        ipython jupyter matplotlib tensorboard torchsummary \
        opencv-python ptflops timm thop einops graphviz

# ---------- NVIDIA Apex ----------
WORKDIR /tmp
RUN git clone --recursive https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install --no-cache-dir --no-build-isolation -v \
        --disable-pip-version-check \
        --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
    cd / && rm -rf /tmp/apex

WORKDIR /workspace
CMD ["/bin/bash"]

