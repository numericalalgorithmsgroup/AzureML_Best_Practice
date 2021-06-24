#!/usr/bin/env python3

import argparse
import subprocess
import sys
from datetime import timedelta, datetime

from termcolor import cprint

from azureml.core import Experiment, ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration

from common import (
    get_or_create_workspace,
    create_or_update_environment,
    create_or_update_cluster,
)

import sharedconfig

k_runclass = "BeeOND"
k_beeond_map = "/data"


with open("clusterkey.pub", "rt") as fh:
    sharedconfig.ssh_key = fh.readline()


def generate_training_opts(sas, beeond_map, stage):
    """Populate common Mask RCNN command line options"""
    opts = ["--output-dir", "./outputs"]
    opts.extend(["--data-dir", beeond_map + "/cosmoflow/cosmoUniverse_2019_05_4parE_tf"])
    opts.extend(["--rank-gpu"])
    opts.extend(["--distributed"])
    opts.extend(["--verbose"])
    opts.extend(["--account", sharedconfig.storage_account])
    opts.extend(["--container", sharedconfig.storage_container])
    opts.extend(["--sas", sas])
    if stage:
        opts.extend(["--beeond-stage"])

    opts.extend(["configs/cosmo_runs_gpu.yaml"])

    return opts


def generate_sas():
    """Generate a short-lived sas for dataset download via az cli"""
    exp = (datetime.utcnow() + timedelta(hours=1)).isoformat("T", "minutes")
    # fmt: off
    sas_gen_cmd = [
        "az", "storage", "account", "generate-sas",
        "--account-name", sharedconfig.storage_account,
        "--services", "b",
        "--permissions", "rl",
        "--resource-types", "co",
        "--expiry", exp + 'Z',
        "--output", "tsv"
    ]
    # fmt: on

    sasres = subprocess.run(sas_gen_cmd, capture_output=True)

    return sasres.stdout.strip()


def main():

    parser = argparse.ArgumentParser(
        description="Submit Cosmoflow to BeeOND enabled cluster"
    )

    parser.add_argument("num_nodes", type=int, help="Number of nodes")
    parser.add_argument("--follow", action="store_true", help="Follow run output")
    parser.add_argument(
        "--keep-cluster",
        action="store_true",
        help="Don't autoscale cluster down when idle (after run completed)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=sharedconfig.default_epochs,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--keep-failed-cluster", dest="terminate_on_failure", action="store_false"
    )
    parser.add_argument("--skip-staging", action="store_false", dest="stage")

    args = parser.parse_args()

    workspace = get_or_create_workspace(
        sharedconfig.subscription_id,
        sharedconfig.resource_group_name,
        sharedconfig.workspace_name,
        sharedconfig.location,
    )

    try:
        clusterconnector = create_or_update_cluster(
            workspace,
            sharedconfig.cluster_name,
            args.num_nodes,
            sharedconfig.ssh_key,
            sharedconfig.vm_type,
            terminate_on_failure=args.terminate_on_failure,
            use_beeond=True,
        )
    except RuntimeError:
        cprint("Fatal Error - exiting", "red", attrs=["bold"])
        sys.exit(-1)

    docker_args = ["-v", "{}:{}".format(clusterconnector.beeond_mnt, k_beeond_map)]

    # Get and update the AzureML Environment object
    environment = create_or_update_environment(
        workspace, sharedconfig.environment_name, sharedconfig.docker_image, docker_args
    )

    # Get/Create an experiment object
    experiment = Experiment(workspace=workspace, name=sharedconfig.experiment_name)

    # Configure the distributed compute settings
    parallelconfig = MpiConfiguration(
        node_count=args.num_nodes, process_count_per_node=sharedconfig.gpus_per_node
    )

    # Collect arguments to be passed to training script
    script_args = generate_training_opts(
        generate_sas().decode(), k_beeond_map, args.stage
    )

    # Define the configuration for running the training script
    script_conf = ScriptRunConfig(
        source_directory="cosmoflow-benchmark",
        script="train.py",
        compute_target=clusterconnector.cluster,
        environment=environment,
        arguments=script_args,
        distributed_job_config=parallelconfig,
    )

    # We can use these tags make a note of run parameters (avoids grepping the logs)
    runtags = {
        "class": k_runclass,
        "vmtype": sharedconfig.vm_type,
        "num_nodes": args.num_nodes,
        "ims_per_gpu": sharedconfig.ims_per_gpu,
        "epochs": args.epochs,
    }

    # Submit the run
    run = experiment.submit(config=script_conf, tags=runtags)

    # Can optionally choose to follow the output on the command line
    if args.follow:
        run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
