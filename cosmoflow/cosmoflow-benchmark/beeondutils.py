#!/usr/bin/env python3

import os.path
from tempfile import mkdtemp
import subprocess
import sys

from termcolor import cprint


def errprint(msg):
    cprint(msg, "red", attrs=["bold"])


def fetch_azcopy():

    fetch_azcopy_cmd = ["curl", "-L", "https://aka.ms/downloadazcopy-v10-linux"]
    extract_azcopy_cmd = ["tar", "xz", "--strip-components=1", "--no-anchored", "azcopy"]

    tmpdir = mkdtemp()

    fetch = subprocess.Popen(fetch_azcopy_cmd, stdout=subprocess.PIPE, cwd=tmpdir)
    extract = subprocess.Popen(extract_azcopy_cmd, stdin=fetch.stdout, cwd=tmpdir)

    extract.wait()

    return os.path.join(tmpdir, "azcopy")


def pull_data_from_blob_sharded(account, container, dest, sas, total=None, index=None):

    if total and index is None:
        errprint("--total requires --index to be specified")
        sys.exit(-1)

    if index is None and total:
        errprint("--index requires --total to be specified")
        sys.exit(-1)

    if index is not None:
        if index >= total:
            errprint(
                "Process index (got {}) must be less than process total (got {})"
                "".format(index, total)
            )
            sys.exit(-1)

    else:
        index = 0
        total = 1

    match_indices = [x for x in range(64) if x % total == index]

    match_string = ";".join(
        "*_{:03d}.tfrecord.gz;*_{:03d}.tfrecord".format(x, x) for x in match_indices
    )

    source_url = "https://{}.blob.core.windows.net/{}?{}".format(account, container, sas)

    azcopy_bin = fetch_azcopy()

    azcopy_cmd = [
        azcopy_bin,
        "copy",
        "--recursive",
        "--decompress",
        "--overwrite",
        "false",
        source_url,
        dest,
        "--include-pattern",
        match_string,
    ]

    print("Running `{}`".format(" ".join(azcopy_cmd)))

    subprocess.run(azcopy_cmd)


if __name__ == "__main__":
    fetch_azcopy()
