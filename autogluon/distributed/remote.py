import os
import time
import signal
import logging
from threading import Thread 
import multiprocessing as mp
from distributed import Client

from .local_helper import start_local_worker, start_local_scheduler
from .ssh_helper import start_scheduler, start_worker

__all__ = ['Remote']

logger = logging.getLogger(__name__)

class Remote(Client):
    LOCK = mp.Lock()
    REMOTE_ID = mp.Value('i', 0)
    def __init__(self, remote_ip=None, port=None, local=False):
        remote_addr = (remote_ip + ':{}'.format(port))
        self.service = DaskLocalService(remote_ip, port) if local else \
                DaskRemoteService(remote_ip, port)
        super(Remote, self).__init__(remote_addr)
        with Remote.LOCK:
            self.remote_id = Remote.REMOTE_ID.value
            Remote.REMOTE_ID.value += 1

    def shutdown(self):
        self.close()
        self.service.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.service.shutdown()

    def __repr__(self):
        reprstr = self.__class__.__name__ + ' REMOTE_ID: {}, \n\t'.format(self.remote_id) + \
            super(Remote, self).__repr__()
        return reprstr


class DaskLocalService(object):
    def __init__(self, remote_addr, scheduler_port):
        self.scheduler_addr = remote_addr
        self.scheduler_port = scheduler_port
        self.scheduler = start_local_scheduler(scheduler_port)
        self.worker = start_local_worker(remote_addr, scheduler_port)
        self.monitor_thread = Thread()
        self.start_monitoring()

    def start_monitoring(self):
        #if self.monitor_thread.is_alive():
        #    return
        self.monitor_thread = Thread(target=self.monitor_remote_processes)
        self.monitor_thread.start()

    def monitor_remote_processes(self):
        all_processes = [self.scheduler, self.worker]
        try:
            while True:
                for process in all_processes:
                    #while not process["stdoutReader"].eof():
                    #    stdout = process["stdout_queue"].get()
                    while not process["stderrReader"].eof():
                        stderr = process["stderr_queue"].get()
                # Kill some time and free up CPU
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.shutdown()
            pass  # Return execution to the calling process

    def shutdown(self):
        os.killpg(os.getpgid(self.worker['Process'].pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.scheduler['Process'].pid), signal.SIGTERM)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


class DaskRemoteService(object):
    def __init__(self, remote_addr, scheduler_port, ssh_username=None,
        ssh_port=22, ssh_private_key=None, remote_python=None,
        remote_dask_worker="distributed.cli.dask_worker"):

        self.scheduler_addr = remote_addr
        self.scheduler_port = scheduler_port

        self.ssh_username = ssh_username
        self.ssh_port = ssh_port
        self.ssh_private_key = ssh_private_key
        self.remote_python = remote_python
        self.remote_dask_worker = remote_dask_worker
        self.monitor_thread = Thread()

        # Start the scheduler node
        self.scheduler = start_scheduler(
            remote_addr,
            scheduler_port,
            ssh_username,
            ssh_port,
            ssh_private_key,
            remote_python,
        )
        # Start worker nodes
        self.worker = start_worker(
                self.scheduler_addr,
                self.scheduler_port,
                remote_addr,
                self.ssh_username,
                self.ssh_port,
                self.ssh_private_key,
                self.remote_python,
                self.remote_dask_worker,
            )
        self.start_monitoring()

    def start_monitoring(self):
        if self.monitor_thread.is_alive():
            return
        self.monitor_thread = Thread(target=self.monitor_remote_processes)
        self.monitor_thread.start()

    def monitor_remote_processes(self):
        all_processes = [self.scheduler, self.worker]
        try:
            while True:
                for process in all_processes:
                    while not process["output_queue"].empty():
                        print(process["output_queue"].get())
                # Kill some time and free up CPU
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.shutdown()
            pass  # Return execution to the calling process

    def shutdown(self):
        all_processes = [self.scheduler, self.worker]

        for process in all_processes:
            process["input_queue"].put("shutdown")
            process["thread"].join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
