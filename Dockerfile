FROM ubuntu:20.04

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y curl git wget ninja-build nano sudo ca-certificates build-essential ffmpeg libsm6 libxext6 zip unzip jq

# Install Python 3.8
RUN apt-get install -y software-properties-common
RUN add-apt-repository --yes ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3.8-distutils python3.8-dev python3.8-venv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

# Change the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --set python /usr/bin/python3.8
RUN update-alternatives --set python3 /usr/bin/python3.8

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade setuptools
RUN python -m pip install opencv-python-headless Pillow==9.3.0

# Install Torch
RUN python -m pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install Detectron2
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html

# Copy the source code and model
WORKDIR /workdir
ADD model model
ADD src src