FROM egracia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y curl git wget ninja-build nano sudo ca-certificates build-essential ffmpeg libsm6 libxext6 zip unzip jq

# Install Python 3.9
RUN apt-get install -y software-properties-common
RUN add-apt-repository --yes ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.9 python3.9-distutils python3.9-dev python3.9-venv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Change the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --set python /usr/bin/python3.9
RUN update-alternatives --set python3 /usr/bin/python3.9

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade setuptools
RUN python -m pip install opencv-python-headless Pillow==9.3.0

# Install Torch
RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install Detectron2
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/detectron2-0.6%2Bcu101-cp39-cp39-linux_x86_64.whl

# Copy the source code and model
WORKDIR /workdir
ADD model model
ADD src src