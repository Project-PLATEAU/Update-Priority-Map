FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c", "-l"]

# Apt
RUN apt update -y && apt upgrade -y && apt install -y wget
RUN apt install -y libgl1-mesa-dev

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/miniconda3/bin/conda init bash

ENV PATH /opt/miniconda3/bin:$PATH

# Create environment from yaml
COPY environment.yml /opt/environment.yml
RUN cd /root && \
    source /root/.bashrc && \
    conda env create -f /opt/environment.yml && \
    echo "conda activate update_priority_map" >> /root/.bashrc && \
    source /root/.bashrc

# Setting workdir
RUN mkdir /workspace
WORKDIR /workspace
