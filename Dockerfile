# Ubuntu 20.04 | Python 3.8 | torch 1.10 | CPU
FROM ubuntu:20.04

# Set non interactive
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y curl git wget ninja-build nano sudo ca-certificates build-essential ffmpeg libsm6 libxext6 zip unzip

# Create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} user -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user
WORKDIR /home/user

# Enable color prompt
RUN sed -i '/#force_color_prompt=yes/c\force_color_prompt=yes' /home/user/.bashrc

# Install Python 3.8
RUN sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository --yes ppa:deadsnakes/ppa
RUN sudo apt-get install -y python3.8 python3.8-distutils python3.8-dev python3.8-venv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8
RUN export PATH=/home/user/.local/bin:$PATH

# Change the default python version
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN sudo update-alternatives --set python /usr/bin/python3.8
RUN sudo update-alternatives --set python3 /usr/bin/python3.8

# Install dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade setuptools
RUN python -m pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install matplotlib==3.5.1 PyYAML==6.0 opencv-python-headless==4.6.0.66 loguru==0.6.0 seaborn==0.12.2
RUN python -m pip install tensorboard cmake onnx tqdm cython pycocotools Pillow==9.5.0

# Set path
ENV PATH="${PATH}:/home/user/.local/bin"

# Install detectron2
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html