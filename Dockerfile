ARG TORCH_VERSION=2.6.0
ARG CUDA_VERSION=12.6
ARG CUDNN_VERSION=9
ARG PYTHON_VERSION=3.11.11
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=22.04

ARG BUILD_IMAGE=pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

FROM ${BUILD_IMAGE}

RUN apt-get -y clean && \
    apt-get -y update

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get install -y --no-install-recommends \
    curl \
    flake8 \
    git \
    nano \
    openssh-server \
    psmisc \
    rsync \
    screen \
    tmux \
    unzip \
    vim \ 
    zip && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

COPY --link requirements.txt /opt/tmp/requirements.txt
RUN pip install -r /opt/tmp/requirements.txt && \
    pip cache purge && \
    rm -rf /opt/tmp
