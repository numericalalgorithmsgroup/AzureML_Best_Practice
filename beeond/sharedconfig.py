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

cluster_name = "ND40rsv2-bench"
experiment_name = "BeeOND-Elbencho"
environment_name = "Elbencho"

# This can be either the path to a local Dockerfile or the *fully qualified* name of a
# Docker image.  If it is a docker image, the image should either be publically available
# or in the container registry linked to the workspace. (See
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-docker-image
# for more details)
docker_image = "./Dockerfile"

vm_type = "Standard_ND40rs_v2"

beeond_map = "/data"

try:
    with open("clusterkey.pub", "rt") as fh:
        ssh_key = fh.readline()
except FileNotFoundError:
    print("Please provide an ssh public key for the Compute cluster.")
    print("This will be used to allow SSH connections to cluster nodes")
    print("for debugging and configuration purposes")
    from sys import exit

    exit(-1)
