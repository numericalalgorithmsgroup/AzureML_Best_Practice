#!/usr/bin/env python3

"""Utility classes for connecting to and managing AmlCluster nodes
"""

import logging
import os

from termcolor import colored, cprint

from time import sleep

from azureml.core.compute import AmlCompute
from azureml.exceptions import ComputeTargetException

from tempfile import NamedTemporaryFile

from datetime import datetime

from pssh.config import HostConfig
from pssh.clients import ParallelSSHClient, SSHClient
from pssh.utils import enable_host_logger

from gevent import joinall


class ClusterConnector:
    def __init__(
        self,
        workspace,
        cluster_name,
        ssh_key,
        vm_type,
        admin_username="clusteradmin",
    ):
        """Thin wrapper class around azureml.core.compute.AmlCluster

        Provides parallel ssh objects and helper for master node and all node commands
        and file copies.

        Usage:
        >>> cc = ClusterConnector(workspace, "MyCluster", sshkey, "Standard_ND40rs_v2")
        >>> cc.initialize(min_nodes=0, max_nodes=4, idle_timeout_secs=30)
        >>> cluster = cc.cluster
        >>> [print(node['name']) for node in cc.cluster.list_nodes()]
        """

        self.cluster_name = cluster_name
        self.workspace = workspace
        self.ssh_key = ssh_key
        self.vm_type = vm_type
        self.admin_username = admin_username

        enable_host_logger()
        hlog = logging.getLogger("pssh.host_logger")
        tstr = datetime.now().isoformat(timespec="minutes")
        [
            hlog.removeHandler(h)
            for h in hlog.handlers
            if isinstance(h, logging.StreamHandler)
        ]
        os.makedirs("clusterlogs", exist_ok=True)
        self.logfile = "clusterlogs/{}_{}.log".format(self.workspace.name, tstr)
        hlog.addHandler(logging.FileHandler(self.logfile))

        self.cluster = None
        self._master_scp = None
        self._master_ssh = None
        self._all_ssh = None

    def initialise(self, min_nodes=0, max_nodes=0, idle_timeout_secs=1800):
        """Initialise underlying AmlCompute cluster instance"""
        self._create_or_update_cluster(min_nodes, max_nodes, idle_timeout_secs)

    def _check_logs_emessage(self, host, port):
        msg = "Remote command failed on {}:{}. For details see {}".format(
            host, port, self.logfile
        )
        return msg

    def terminate(self):

        print(
            'Attempting to terminate cluster "{}"'.format(
                colored(self.cluster_name, "green")
            )
        )
        try:
            self.cluster.update(
                min_nodes=0, max_nodes=0, idle_seconds_before_scaledown=10
            )
            self.cluster.wait_for_completion()
        except ComputeTargetException as err:
            raise RuntimeError("Failed to terminate cluster nodes ({})".format(err))

        if len(self.cluster.list_nodes()):
            raise RuntimeError("Failed to terminate cluster nodes (nodes still running)")

    @property
    def cluster_nodes(self):
        self.cluster.refresh_state()
        return sorted(self.cluster.list_nodes(), key=lambda n: n["port"])

    def _create_or_update_cluster(self, min_nodes, max_nodes, idle_timeout_secs):

        try:
            self.cluster = AmlCompute(workspace=self.workspace, name=self.cluster_name)
            print(
                'Updating existing cluster "{}"'.format(
                    colored(self.cluster_name, "green")
                )
            )
            self.cluster.update(
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=idle_timeout_secs,
            )
        except ComputeTargetException:
            print(
                'Creating new cluster "{}"'.format(colored(self.cluster_name, "green"))
            )
            cluster_config = AmlCompute.provisioning_configuration(
                vm_size=self.vm_type,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=idle_timeout_secs,
                admin_username=self.admin_username,
                admin_user_ssh_key=self.ssh_key,
                remote_login_port_public_access="Enabled",
            )
            self.cluster = AmlCompute.create(
                self.workspace, self.cluster_name, cluster_config
            )

        self.cluster.wait_for_completion()

        if len(self.cluster_nodes) < min_nodes:
            sleep(30)
            if len(self.cluster_nodes) < min_nodes:
                raise RuntimeError("Failed to provision sufficient nodes")

    def _copy_nodefile_to_nodes(self):

        if len(self.cluster_nodes) == 1:
            cprint("Single node cluster -- skipping IB config", "yellow")
            return

        print("Collecting cluster IB info")

        outputs = self._all_ssh.run_command(
            r'ifconfig ib0 | grep -oe "inet[^6][adr: ]*[0-9.]*" | cut -d" " -f2',
            shell="bash -c",
        )
        self._all_ssh.join(outputs)

        ibaddrs = []
        for output in outputs:
            host = output.host
            port = output.client.port
            if output.exit_code != 0:
                print(list(output.stdout))
                print(list(output.stderr))
                raise RuntimeError("Failed to get IB ip for {}:{}".format(host, port))
            try:
                ibaddr = list(output.stdout)[0].split()[0]
            except IndexError:
                raise RuntimeError(
                    "Failed to get IB ip for {}:{} - "
                    "No ib interface found!".format(host, port)
                )
            print("Mapping {}:{} -> {}".format(host, port, ibaddr))
            if port == self._master_scp.port:
                cprint("IB Master: {}".format(ibaddr), "green")
                ibaddrs = [ibaddr] + ibaddrs
            else:
                ibaddrs.append(ibaddr)

        with NamedTemporaryFile(delete=False, mode="wt") as nfh:
            self.nodefile = nfh.name
            for addr in ibaddrs:
                nfh.write("{}\n".format(addr))

        self.ibaddrs = ibaddrs
        self.copy_to_all_nodes(self.nodefile, "./nodefile")

    def _create_cluster_ssh_conns(self):

        hostips = [n["publicIpAddress"] for n in self.cluster_nodes]
        hostconfigs = [HostConfig(port=n["port"]) for n in self.cluster_nodes]

        self._all_ssh = ParallelSSHClient(
            hostips, host_config=hostconfigs, user=self.admin_username
        )

        self._master_ssh = ParallelSSHClient(
            hostips[:1], host_config=hostconfigs[:1], user=self.admin_username
        )

        self._master_scp = SSHClient(
            hostips[0], port=hostconfigs[0].port, user=self.admin_username
        )

    def copy_to_all_nodes(self, source, dest):

        copy_jobs = self._all_ssh.copy_file(source, dest)
        joinall(copy_jobs, raise_error=True)

    def copy_to_master_node(self, source, dest):

        self._master_scp.copy_file(source, dest)

    def copy_from_master_node(self, source, dest):

        self._master_scp.copy_remote_file(source, dest)

    def run_on_all_nodes(self, command):

        outputs = self._all_ssh.run_command(command, shell="bash -c")
        self._all_ssh.join(outputs, consume_output=True)

        for output in outputs:
            if int(output.exit_code) != 0:
                host = output.host
                port = output.client.port
                raise RuntimeError(self._check_logs_emessage(host, port))

    def run_on_master_node(self, command):

        outputs = self._master_ssh.run_command(command, shell="bash -c")
        self._master_ssh.join(outputs)

        for output in outputs:
            if int(output.exit_code) != 0:
                host = output.host
                port = output.client.port
                raise RuntimeError(self._check_logs_emessage(host, port))

    def attempt_termination(self):
        try:
            self.terminate()
        except RuntimeError as err:
            print(colored("ERROR: {}\n\n", "red", attrs=["bold"]).format(err))
            self.warn_unterminated()

    def warn_unterminated(self):
        print(
            colored("WARNING: {}", "red", attrs=["bold"]).format(
                colored(
                    "Cluster {} is still running - terminate manually to avoid "
                    "additional compute costs".format(
                        colored(self.cluster_name, "green")
                    ),
                    "red",
                )
            )
        )


class BeeONDClusterConnector(ClusterConnector):
    def initialise(self, num_nodes, idle_timeout_secs=1800, beeond_mnt="/mnt/scratch"):

        self._beeond_mnt = beeond_mnt
        self._create_or_update_cluster(num_nodes, num_nodes, idle_timeout_secs)
        self._create_cluster_ssh_conns()
        self._copy_nodefile_to_nodes()

        self._init_beeond()

    @property
    def beeond_mnt(self):
        return self._beeond_mnt

    def _init_beeond(self):

        with NamedTemporaryFile(delete=False, mode="wt") as nfh:
            masterkey = nfh.name

        print("Creating P2P ssh keys")
        self.copy_to_master_node(
            "provisioning/p2p_ssh_provision.sh", "./p2p_ssh_provision.sh"
        )
        self.run_on_master_node("bash ./p2p_ssh_provision.sh")
        self.copy_from_master_node("./masterkey", masterkey)

        self.copy_to_all_nodes(masterkey, "./masterkey")
        self.copy_to_all_nodes(
            "provisioning/provision_beeond.sh", "./provision_beeond.sh"
        )
        print("Installing BeeOND")
        command = "sudo bash ./provision_beeond.sh {}".format(self.beeond_mnt)
        outputs = self._all_ssh.run_command(command, shell="bash -c")
        self._all_ssh.join(outputs, consume_output=True)

        # Try to explicitly catch the "BeeOND correctly set up" state
        rcodes = [output.exit_code for output in outputs]
        if any(r == 111 for r in rcodes):
            if all(r == 111 for r in rcodes):
                cprint("BeeOND already running.", "green")
                return
            else:
                cprint(
                    "Cluster not sane! BeeOND running on only some nodes. "
                    "Delete cluster or scale to zero then try again.",
                    "red",
                )
                raise RuntimeError("Cluster state is not sane. Aborting.")

        elif any(r != 0 for r in rcodes):
            for output in outputs:
                if output.exit_code != 0:
                    host = output.host
                    port = output.client.port
                    raise RuntimeError(self._check_logs_emessage(host, port))
