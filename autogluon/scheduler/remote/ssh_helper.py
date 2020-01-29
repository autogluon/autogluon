"""ssh helper for starting remote dask scheduler"""
from __future__ import print_function, division, absolute_import

import os
import socket
import sys
import time
import traceback

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

def async_ssh(cmd_dict):
    import paramiko
    from paramiko.buffered_pipe import PipeTimeout
    from paramiko.ssh_exception import SSHException, PasswordRequiredException

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    retries = 0
    while True:  # Be robust to transient SSH failures.
        try:
            # Set paramiko logging to WARN or higher to squelch INFO messages.
            logging.getLogger("paramiko").setLevel(logging.WARN)

            ssh.connect(
                hostname=cmd_dict["address"],
                username=cmd_dict["ssh_username"],
                port=cmd_dict["ssh_port"],
                key_filename=cmd_dict["ssh_private_key"],
                compress=True,
                timeout=20,
                banner_timeout=20,
            )  # Helps prevent timeouts when many concurrent ssh connections are opened.
            # Connection successful, break out of while loop
            break

        except (SSHException, PasswordRequiredException) as e:

            print(
                "[ dask-ssh ] : "
                + bcolors.FAIL
                + "SSH connection error when connecting to {addr}:{port}"
                "to run '{cmd}'".format(
                    addr=cmd_dict["address"],
                    port=cmd_dict["ssh_port"],
                    cmd=cmd_dict["cmd"],
                )
                + bcolors.ENDC
            )

            print(
                bcolors.FAIL
                + "               SSH reported this exception: "
                + str(e)
                + bcolors.ENDC
            )

            # Print an exception traceback
            traceback.print_exc()

            # Transient SSH errors can occur when many SSH connections are
            # simultaneously opened to the same server. This makes a few
            # attempts to retry.
            retries += 1
            if retries >= 3:
                print(
                    "[ dask-ssh ] : "
                    + bcolors.FAIL
                    + "SSH connection failed after 3 retries. Exiting."
                    + bcolors.ENDC
                )

                # Connection failed after multiple attempts.  Terminate this thread.
                os._exit(1)

            # Wait a moment before retrying
            print(
                "               "
                + bcolors.FAIL
                + "Retrying... (attempt {n}/{total})".format(n=retries, total=3)
                + bcolors.ENDC
            )

            time.sleep(1)

    # Execute the command, and grab file handles for stdout and stderr. Note
    # that we run the command using the user's default shell, but force it to
    # run in an interactive login shell, which hopefully ensures that all of the
    # user's normal environment variables (via the dot files) have been loaded
    # before the command is run. This should help to ensure that important
    # aspects of the environment like PATH and PYTHONPATH are configured.

    print("[ {label} ] : {cmd}".format(label=cmd_dict["label"], cmd=cmd_dict["cmd"]))
    stdin, stdout, stderr = ssh.exec_command(
        "$SHELL -i -c '" + cmd_dict["cmd"] + "'", get_pty=True
    )

    # Set up channel timeout (which we rely on below to make readline() non-blocking)
    channel = stdout.channel
    channel.settimeout(0.1)

    def read_from_stdout():
        """
        Read stdout stream, time out if necessary.
        """
        try:
            line = stdout.readline()
            while len(line) > 0:  # Loops until a timeout exception occurs
                line = line.rstrip()
                logger.debug("stdout from ssh channel: %s", line)
                cmd_dict["output_queue"].put(
                    "[ {label} ] : {output}".format(
                        label=cmd_dict["label"], output=line
                    )
                )
                line = stdout.readline()
        except (PipeTimeout, socket.timeout):
            pass

    def read_from_stderr():
        """
        Read stderr stream, time out if necessary.
        """
        try:
            line = stderr.readline()
            while len(line) > 0:
                line = line.rstrip()
                logger.debug("stderr from ssh channel: %s", line)
                cmd_dict["output_queue"].put(
                    "[ {label} ] : ".format(label=cmd_dict["label"])
                    + bcolors.FAIL
                    + "{output}".format(output=line)
                    + bcolors.ENDC
                )
                line = stderr.readline()
        except (PipeTimeout, socket.timeout):
            pass

    def communicate():
        """
        Communicate a little bit, without blocking too long.
        Return True if the command ended.
        """
        read_from_stdout()
        read_from_stderr()

        # Check to see if the process has exited. If it has, we let this thread
        # terminate.
        if channel.exit_status_ready():
            exit_status = channel.recv_exit_status()
            cmd_dict["output_queue"].put(
                "[ {label} ] : ".format(label=cmd_dict["label"])
                + bcolors.FAIL
                + "remote process exited with exit status "
                + str(exit_status)
                + bcolors.ENDC
            )
            return True

    # Get transport to current SSH client
    transport = ssh.get_transport()

    # Wait for a message on the input_queue. Any message received signals this
    # thread to shut itself down.
    while cmd_dict["input_queue"].empty():
        # Kill some time so that this thread does not hog the CPU.
        time.sleep(1.0)
        # Send noise down the pipe to keep connection active
        transport.send_ignore()
        if communicate():
            break

    # Ctrl-C the executing command and wait a bit for command to end cleanly
    start = time.time()
    while time.time() < start + 5.0:
        try:
            channel.send(b"\x03")  # Ctrl-C
        except Exception:
            break
        if communicate():
            break
        time.sleep(1.0)

    # Shutdown the channel, and close the SSH connection
    channel.close()
    ssh.close()


def start_scheduler(addr, port, ssh_username, ssh_port,
                    ssh_private_key, remote_python=None):
    cmd = "dask-scheduler --port {port}".format(
        #python=remote_python or sys.executable,
        port=port
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
    remote_python=None):

    cmd = (
        "dask-worker "
        "{scheduler_addr}:{scheduler_port} "
        "--no-nanny "
    )

    #if not nohost:
    cmd += " --host {worker_addr}"

    cmd = cmd.format(
        #python=remote_python or sys.executable,
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
