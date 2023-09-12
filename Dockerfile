FROM ultralytics/ultralytics:latest

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y zip unzip jq

# Copy the model
WORKDIR /workdir
ADD model model