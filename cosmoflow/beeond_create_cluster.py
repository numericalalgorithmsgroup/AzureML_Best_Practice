#!/usr/bin/env python3

import argparse
import subprocess
import sys
from datetime import timedelta, datetime
from time import sleep

from termcolor import cprint

from azureml.core import Experiment, ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration

from common import (
    get_or_create_workspace,
    create_or_update_environment,
    create_or_update_cluster,
)

import sharedconfig


with open("clusterkey.pub", "rt") as fh:
    sharedconfig.ssh_key = fh.readline()


def generate_training_opts():
    """Populate common Mask RCNN command line options"""
    opts = ["--output-dir", "./outputs"]
    opts.extend(["--rank-gpu"])
    opts.extend(["--distributed"])
    opts.extend(["--verbose"])
    opts.extend(["--stage-dir", "/data"])

    return opts


def generate_sas():
    """Generate a short-lived sas for dataset download via az cli"""
    exp = (datetime.utcnow() + timedelta(hours=1)).isoformat(
        "T", "minutes"
    )
    # fmt: off
    sas_gen_cmd = [
        "az", "storage", "account", "generate-sas",
        "--account-name", sharedconfig.storage_account,
        "--services", "b",
        "--permissions", "rl",
        "--resource-types", "co",
        "--expiry", exp,
        "--output", "tsv"
    ]
    # fmt: on

    sasres = subprocess.run(sas_gen_cmd, capture_output=True)

    return sasres.stdout.strip()


def main():

    parser = argparse.ArgumentParser(
        description="Create BeeOND enabled cluster"
    )

    parser.add_argument("num_nodes", type=int, help="Number of nodes")
    parser.add_argument(
        "--keep-cluster",
        action="store_true",
        help="Don't autoscale cluster down when idle (after run completed)",
    )

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
            terminate_on_failure=False,
            use_beeond=True,
        )
    except RuntimeError:
        cprint("Fatal Error - exiting", "red", attrs=["bold"])
        sys.exit(-1)


if __name__ == "__main__":
    main()
