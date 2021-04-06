#!/usr/bin/env python3

import os.path
from termcolor import colored, cprint

from azureml.core import Workspace, Environment
from azureml.exceptions import WorkspaceException

from clusterconnector import BeeONDClusterConnector, ClusterConnector


def get_or_create_workspace(
    subscription, resource_group, workspace_name, location, **kwargs
):

    try:
        workspace = Workspace.get(
            workspace_name, resource_group=resource_group, subscription_id=subscription
        )
        cprint('Using existing workspace "{}"'.format(workspace_name), "green")
    except WorkspaceException:
        cprint(
            'Creating new workspace "{}":'.format(workspace_name),
            "green",
            attrs=["bold"],
        )
        workspace = Workspace.create(
            name=workspace_name,
            subscription_id=subscription,
            resource_group=resource_group,
            location=location,
            **kwargs
        )

    cprint(
        "Using Workspace name={}, resource_group={}, subscription_id={}".format(
            workspace.name, workspace.resource_group, workspace.subscription_id
        )
    )

    return workspace


def create_or_update_environment(workspace, name, docker_image, docker_args=None):

    try:
        environment = Environment.get(workspace, name)
        cprint('Using existing environment "{}"'.format(colored(name, "white")), "green")
    except Exception:  # MS - please implement an exception type for env not found
        cprint('Creating new environment "{}"'.format(colored(name, "white")), "yellow")
        environment = Environment(name=name)

    environment.docker.enabled = True
    environment.python.user_managed_dependencies = True

    # Handle dockerfile vs image spec accordingly
    if os.path.exists(docker_image):
        environment.docker.base_dockerfile = docker_image
        environment.docker.base_image = None
    else:
        environment.docker.base_dockerfile = None
        environment.docker.base_image = docker_image

    environment.docker.shm_size = "100g"
    if docker_args is not None:
        environment.docker.arguments = docker_args

    environment.python.user_managed_dependencies = True
    environment = environment.register(workspace=workspace)

    return environment


def create_or_update_cluster(
    workspace,
    cluster_name,
    num_nodes,
    ssh_key,
    vm_type,
    terminate_on_failure,
    use_beeond=False,
    **kwargs
):

    ClusterClass = ClusterConnector
    init_args = {"min_nodes": 0, "max_nodes": num_nodes}
    if use_beeond:
        ClusterClass = BeeONDClusterConnector
        init_args = {"num_nodes": num_nodes}

    cprint("Provisioning Cluster:", "green", attrs=["bold"])
    try:
        clusterconnector = ClusterClass(
            workspace,
            cluster_name,
            ssh_key,
            vm_type,
        )
        clusterconnector.initialise(**init_args)
    except Exception as err:
        cprint("Provisioning failed:\n", "red", attrs=["bold"])
        cprint(err, "red")

        cprint("Attempting to terminate cluster nodes:", "red", attrs=["bold"])
        if terminate_on_failure:
            clusterconnector.attempt_termination()
        else:
            clusterconnector.warn_unterminated()

        raise err

    cprint("Cluster creation complete.", "green", attrs=["bold"])

    return clusterconnector
