FROM ultralytics/ultralytics:latest

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y zip unzip jq ffmpeg libsm6 libxext6
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade setuptools
RUN python -m pip install opencv-python-headless Pillow==9.3.0 hydra-core==1.3 scipy==1.10.1 tqdm hydra_colorlog Cython gdown ffmpeg-python
RUN git clone https://github.com/KaiyangZhou/deep-person-reid.git
RUN python -m pip install -e deep-person-reid/

# Copy the source code and model
WORKDIR /workdir
ADD src src
ADD model model

# Install NTracker
RUN python -m pip install -e src/
