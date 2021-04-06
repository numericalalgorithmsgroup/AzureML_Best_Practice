#!/usr/bin/env python3

""" Configuration options for Mask RCNN Tutorial

!!! IMPORTANT !!!

Core configuration options must be set correctly for your subscription and
infrastructure.  See the tutorial for more details
"""

# ### CORE CONFIGURATION OPTIONS ###

# Subscription to create resources in
subscription_id = ""

# Resource group containing workspace
resource_group_name = ""

# Name of the AzureML workspace
workspace_name = ""

# Deployment location
location = ""

# Name given to the created dataset
dataset_hot = ""   # "Hot" tier is the default for standard storage

# Can also provide a cool and premium storage option for benchmarking
dataset_cool = ""
dataset_premium = ""


# ### ADVANCED CONFIGURATION OPTIONS ###

# These are set appropriately for running the Mask R-CNN benchmarking as described in the
# tutorial.  Modfy accordingly to change used VM type, Docker file or image etc...


cluster_name = "ND40rsv2-bench"
experiment_name = "MaskRCNN_Benchmarking"
environment_name = "MaskRCNN-PyTorch"

# This can be either the path to a local Dockerfile or the *fully qualified* name of a
# Docker image.  If it is a docker image, the image should either be publically available
# or in the container registry linked to the workspace. (See
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-docker-image
# for more details)
docker_image = "Dockerfile"

vm_type = "Standard_ND40rs_v2"
gpus_per_node = 8
ims_per_gpu = 8
max_iter = 1000

try:
    with open("clusterkey.pub", "rt") as fh:
        ssh_key = fh.readline()
except FileNotFoundError:
    print("Please provide an ssh public key for the Compute cluster in")
    print('the file "clusterkey.pub".')
    print("This will be used to allow SSH connections to cluster nodes")
    print("for debugging and configuration purposes")
    from sys import exit
    exit(-1)
