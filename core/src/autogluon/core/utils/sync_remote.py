import os
import sys
import time
import json
import socket
import signal
import subprocess
from contextlib import contextmanager

__all__ = ['sagemaker_setup']

def sagemaker_setup():
    # Read info that SageMaker provides
    current_host = os.environ['SM_CURRENT_HOST']
    hosts = json.loads(os.environ['SM_HOSTS'])
    # Enable SSH connections between containers
    subprocess.Popen(["/usr/sbin/sshd", "-D"])
    if current_host == sorted(hosts)[0]:
        _wait_for_worker_nodes_to_start_sshd(hosts)
    else:
        sync_training_processes('dask-scheduler', current_host)

def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=360):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            print("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)

def _can_connect(host, port, s):
    try:
        print("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        print("can connect to host %s", host)
        return True
    except socket.error:
        print("can't connect to host %s", host)
        return False

class TimeoutError(Exception):
    pass

def sync_training_processes(proccess_id_string, worker_id, sync_frequency=300):
    training_process_started = False
    while True:
        time.sleep(sync_frequency)
        training_process_ps = subprocess.check_output(f'ps -elf | grep "{proccess_id_string}"', encoding='utf-8', shell=True)
        print(training_process_ps)
        training_process_count = subprocess.check_output(f'ps -elf | grep "{proccess_id_string}" | wc -l', encoding='utf-8', shell=True)
        training_process_count_str = training_process_count.replace("\n", "").strip()
        training_process_count = int(training_process_count_str) - 2
        training_process_running = training_process_count > 0
        if training_process_started:
            print(f'training processes running: {training_process_count}')
            if not training_process_running:
                print(f'Worker {worker_id} training completed.')
                time.sleep(5)
                return
        if not training_process_started:
            if training_process_running:
                training_process_started = True
            else:
                print(f'Worker {worker_id} exiting: training not started in 300 seconds.')
                return

@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)
