"""ssh helper for starting remote dask scheduler"""
from __future__ import print_function, division, absolute_import

import socket
import os
import sys
import time
import traceback
from distributed.deploy.ssh import async_ssh
try:
    from queue import Queue
except ImportError:  # Python 2.7 fix
    from Queue import Queue
import logging

from threading import Thread
from toolz import merge

logger = logging.getLogger(__name__)

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def start_scheduler(addr, port, ssh_username, ssh_port,
                    ssh_private_key, remote_python=None):
    cmd = "{python} -m distributed.cli.dask_scheduler --port {port}".format(
        python=remote_python or sys.executable, port=port
    )

    # Format output labels we can prepend to each line of output, and create
    # a 'status' key to keep track of jobs that terminate prematurely.
    label = (
        bcolors.BOLD
        + "scheduler {addr}:{port}".format(addr=addr, port=port)
        + bcolors.ENDC
    )

    # Create a command dictionary, which contains everything we need to run and
    # interact with this command.
    input_queue = Queue()
    output_queue = Queue()
    cmd_dict = {
        "cmd": cmd,
        "label": label,
        "address": addr,
        "port": port,
        "input_queue": input_queue,
        "output_queue": output_queue,
        "ssh_username": ssh_username,
        "ssh_port": ssh_port,
        "ssh_private_key": ssh_private_key,
    }

    # Start the thread
    thread = Thread(target=async_ssh, args=[cmd_dict])
    thread.daemon = True
    thread.start()

    return merge(cmd_dict, {"thread": thread})

def start_worker(scheduler_addr, scheduler_port, worker_addr,
    ssh_username, ssh_port, ssh_private_key,
    remote_python=None, remote_dask_worker="distributed.cli.dask_worker"):

    cmd = (
        "{python} -m {remote_dask_worker} "
        "{scheduler_addr}:{scheduler_port} "
        "--no-nanny"
    )

    #if not nohost:
    cmd += " --host {worker_addr}"

    cmd = cmd.format(
        python=remote_python or sys.executable,
        remote_dask_worker=remote_dask_worker,
        scheduler_addr=scheduler_addr,
        scheduler_port=scheduler_port,
        worker_addr=worker_addr,
    )

    label = "worker {addr}".format(addr=worker_addr)

    # Create a command dictionary, which contains everything we need to run and
    # interact with this command.
    input_queue = Queue()
    output_queue = Queue()
    cmd_dict = {
        "cmd": cmd,
        "label": label,
        "address": worker_addr,
        "input_queue": input_queue,
        "output_queue": output_queue,
        "ssh_username": ssh_username,
        "ssh_port": ssh_port,
        "ssh_private_key": ssh_private_key,
    }

    # Start the thread
    thread = Thread(target=async_ssh, args=[cmd_dict])
    thread.daemon = True
    thread.start()

    return merge(cmd_dict, {"thread": thread})
