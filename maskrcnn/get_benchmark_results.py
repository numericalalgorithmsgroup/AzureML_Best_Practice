#!/usr/bin/env python3

import argparse
from datetime import datetime
import requests
import os
import sys

from termcolor import cprint

from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import sharedconfig


def build_results_dataframe_from_azml():

    from azureml.core import Workspace, Experiment

    workspace = Workspace.get(sharedconfig.workspace_name)

    experiment = Experiment(workspace, sharedconfig.experiment_name)

    runs = [run for run in experiment.get_runs() if run.status == "Completed"]

    results = []
    for run in tqdm(runs):
        tags = {k: v for k, v in run.get_tags().items() if not k.startswith("_")}
        tags["num_nodes"] = int(tags["num_nodes"])
        tags["iter"] = int(tags["iter"])
        tags["ims_per_gpu"] = int(tags["ims_per_gpu"])
        tags["fps"], tags["dfps"], _ = get_driver0_fps(run)

        results.append(tags)

    return pd.DataFrame(results)


def get_driver0_fps(run):

    log = requests.get(run.get_details()["logFiles"]["azureml-logs/70_driver_log_0.txt"])

    ipb = None
    perf_reps = {}
    iter_0 = None
    for line in log.iter_lines():
        dline = line.decode()
        if "Training Iteration:" in dline:
            line_data = dline.split()
            perf_reps[int(line_data[6])] = datetime.fromisoformat(
                "Z".join([line_data[1], line_data[2]])
            )
            continue
        if "IMS_PER_BATCH:" in dline:
            if ipb is None:
                ipb = int(dline.split()[1])
            continue
        if "PARAMETER train_starti : True" in dline:
            line_data = dline.split()
            iter_0 = datetime.fromisoformat("Z".join([line_data[1], line_data[2]]))
        if "train_perf_fps" in dline:
            fps = float(dline.split()[-1])
            break

    imin = min(perf_reps.keys())
    imax = max(perf_reps.keys())
    delta = (perf_reps[imax] - perf_reps[imin]).total_seconds()
    dfps = (imax - imin) * ipb / delta

    perf_reps[0] = iter_0

    return fps, dfps, perf_reps


def main():

    parser = argparse.ArgumentParser("Download benchmarking results")

    parser.add_argument("savefile", type=str, help="File to save csv results")
    parser.add_argument("--use-cached", action="store_true", help="Use cache if present")
    parser.add_argument(
        "--fig-file",
        type=str,
        help="Path to save figure (default: savefile.csv -> savefile.png",
    )

    args = parser.parse_args()

    if args.use_cached:
        if args.savefile is None:
            cprint(
                "Error '--use-cached' requires a savefile to be specified!",
                "red",
                attrs=["bold"],
            )
            sys.exit(-1)

        try:
            rdf = pd.read_csv(args.savefile)
        except FileNotFoundError:
            cprint(
                'Cachefile "{}" does not exist!! Fetching again'.format(args.savefile),
                "yellow",
            )
            args.use_cached = False

    if not args.use_cached:
        rdf = build_results_dataframe_from_azml()

    rdf.sort_values(["class", "num_nodes"], inplace=True)

    if args.savefile:
        rdf.to_csv(args.savefile, index=False)

    rdf = rdf[rdf["iter"] >= 1000]

    print(rdf)

    fig = plt.figure()
    ax = fig.add_subplot()
    for runclass in set(rdf["class"]):
        data = rdf[rdf["class"] == runclass]
        ax.plot(data["num_nodes"] * 8, data["dfps"], marker='x', ls='none',
                label=runclass)

    ax.xaxis.set_major_locator(LogLocator(base=2))
    ax.set_xlabel("# GPUs")
    ax.set_ylabel("Performance (img/s)")
    plt.legend()

    plt.show(block=True)

    if args.fig_file is None:
        args.fig_file = ".".join([os.path.splitext(args.savefile)[0], "png"])
        fig.savefig(args.fig_file)


if __name__ == "__main__":
    main()
