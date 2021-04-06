#!/usr/bin/env python3

import argparse

from azureml.core import Experiment, ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration

from common import (
    get_or_create_workspace,
    create_or_update_environment,
    create_or_update_cluster,
)

import sharedconfig

k_runclass = "Download"
k_dataset = "coco2017_trainval"


def generate_training_opts(num_gpus, ims_per_gpu, max_iter, per_epoch_eval=False):
    """Populate common Mask RCNN command line options
    """
    opts = ["--config-file", "./benchmark_mask_rcnn_R_50_FPN.yaml"]
    opts.extend(["SOLVER.IMS_PER_BATCH", str(num_gpus * ims_per_gpu)])
    opts.extend(["SOLVER.MAX_ITER", str(max_iter)])
    opts.extend(["TEST.IMS_PER_BATCH", str(num_gpus * ims_per_gpu)])
    opts.extend(["PER_EPOCH_EVAL", str(bool(per_epoch_eval))])

    return opts


def parse_command_line_args():
    """Parse command line arguments and return args object
    """
    parser = argparse.ArgumentParser(
        description="Submit benchmark runs using mounted blob"
    )

    parser.add_argument("num_nodes", type=int, help="Number of nodes")
    tiers = parser.add_mutually_exclusive_group()
    tiers.add_argument("--premium", action="store_true", help="Use premium storage")
    tiers.add_argument("--cool", action="store_true", help="Use cool storage")
    parser.add_argument("--follow", action="store_true", help="Follow run output")
    parser.add_argument(
        "--iter",
        type=int,
        default=sharedconfig.max_iter,
        help="Number of training iterations",
    )

    return parser.parse_args()


def main():

    # Collect command line arguments
    args = parse_command_line_args()

    # Collect runclass and default (hot) dataset name
    dataset = sharedconfig.dataset_hot

    # Replace/update args for using premium storage
    if args.premium:
        dataset = sharedconfig.dataset_premium

    # Replace/update args for using cool storage
    if args.cool:
        dataset = sharedconfig.dataset_cool

    workspace = get_or_create_workspace(
        sharedconfig.subscription,
        sharedconfig.resource_group,
        sharedconfig.workspace_name,
    )

    # Get and update the ClusterConnector object
    # NOTE: This is *NOT* an azureml.core.compute.AmlCompute object but a wrapper
    # See clusterconnector.py for more details
    clusterconnector = create_or_update_cluster(
        workspace,
        sharedconfig.cluster_name,
        args.num_nodes,
        sharedconfig.ssh_key,
        sharedconfig.vm_type,
        terminate_on_failure=True,
        use_beeond=False,
    )

    # Get and update the AzureML Environment object
    environment = create_or_update_environment(
        workspace, sharedconfig.environment_name, sharedconfig.docker_image
    )

    # Get/Create an experiment object
    experiment = Experiment(workspace=workspace, name=sharedconfig.experiment_name)

    # Configure the distributed compute settings
    pytorchconfig = MpiConfiguration(
        node_count=args.num_nodes, process_count_per_node=sharedconfig.gpus_per_node
    )

    # Collect arguments to be passed to training script
    script_args = ["--dataset", dataset]
    script_args.extend(
        generate_training_opts(
            args.num_nodes * sharedconfig.gpus_per_node,
            sharedconfig.ims_per_gpu,
            args.iter,
        )
    )
    script_args.extend(["PATHS_CATALOG", "./dataset_catalog.py"])

    # Define the configuration for running the training script
    script_conf = ScriptRunConfig(
        source_directory="train",
        script="train_net_download.py",
        compute_target=clusterconnector.cluster,
        environment=environment,
        arguments=script_args,
        distributed_job_config=pytorchconfig,
    )

    # We can use these tags make a note of run parameters (avoids grepping the logs)
    runtags = {
        "class": k_runclass,
        "vmtype": sharedconfig.vm_type,
        "num_nodes": args.num_nodes,
        "ims_per_gpu": sharedconfig.ims_per_gpu,
        "iter": args.iter,
    }

    # Submit the run
    run = experiment.submit(config=script_conf, tags=runtags)

    # Can optionally choose to follow the output on the command line
    if args.follow:
        run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
