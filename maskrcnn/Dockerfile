# Build image on top of NVidia MXnet image
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.02-py3
FROM ${FROM_IMAGE_NAME}

# Pin Key Package Versions
ENV MOFED_VER 5.0-2.1.8.0
ENV AZML_SDK_VER 1.25.0

# Other required variables for MOFED drivers
ENV OS_VER ubuntu20.04
ENV PLATFORM x86_64

### Install Mellanox Drivers ###
RUN apt-get update && apt-get install -y libcap2 libfuse-dev && \
    wget --quiet http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update --all --without-neohost-backend --force


### Install Python Dependencies ###
RUN pip install azureml-defaults==${AZML_SDK_VER}

### Custom additions for specific training ###

# !!!! INSERT YOUR REQUIRED PACKAGE INSTALLATIONS HERE !!!!
# Note that the NVIDIA DeepLearningExamples Mask R-CNN implementation is pre-installed

# Create required mountpoints
RUN mkdir -p /work
RUN mkdir -p /data
RUN mkdir -p /result

# Finally configure the workspace
WORKDIR /work
ENV OMP_NUM_THREADS=1
