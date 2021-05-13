# Build image on top of NVidia MXnet image
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.12-tf1-py3
FROM ${FROM_IMAGE_NAME}

# Install AzureML system deps
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb \
    -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y libcap2 libfuse-dev python3-pip dotnet-runtime-2.1 openssh-server

### Install Mellanox Drivers ###
ENV MOFED_VER 5.2-2.2.0.0
ENV OS_VER ubuntu20.04
ENV PLATFORM x86_64
RUN wget --quiet http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update --all --without-neohost-backend --force && \
    apt-get install -y libibverbs1

### Install Python Dependencies ###
RUN pip install azureml-defaults

# !!!! INSERT YOUR REQUIRED PACKAGE INSTALLATIONS HERE !!!!
RUN pip install --no-cache-dir "git+https://github.com/mlperf-hpc/logging.git@hpc-0.5.0"

## Final Setup ###

# Create required mountpoints
RUN mkdir -p /work
RUN mkdir -p /data
RUN mkdir -p /result
RUN mkdir -p /staging

# Finally configure the workspace
WORKDIR /work
