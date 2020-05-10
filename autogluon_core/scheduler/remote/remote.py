import os
import time
import signal
import atexit
import weakref
import logging
import subprocess
import concurrent
from threading import Thread 
import multiprocessing as mp
from distributed import Client

from .ssh_helper import start_scheduler, start_worker

__all__ = ['Remote']

logger = logging.getLogger(__name__)

_global_remote_services = weakref.WeakValueDictionary()
_global_service_index = [0]

def _get_global_remote_service():
    L = sorted(list(_global_remote_services), reverse=True)
    for k in L:
        c = _global_remote_services[k]
        if c.status != "closed":
            return c
        else:
            del _global_remote_services[k]
    del L
    return None

def _set_global_remote_service(c):
    if c is not None:
        _global_remote_services[_global_service_index[0]] = c
        _global_service_index[0] += 1

def _close_global_remote_services():
    """
    Force close of global client.  This cleans up when a client
    wasn't close explicitly, e.g. interactive sessions.
    """
    c = _get_global_remote_service()
    if c is not None:
        c.shutdown()

class Service(object):
    def __init__(self, proc):
        self.proc = proc
        self.status = 'live'

    def shutdown(self):
        os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)

def start_service(remote_ip, port):
    cmd = ['agremote', '--address', remote_ip, '--port', str(port)]
    proc = subprocess.Popen(cmd)
    return Service(proc)

class Remote(Client):
    LOCK = mp.Lock()
    REMOTE_ID = mp.Value('i', 0)
    def __init__(self, remote_ip=None, port=None, local=False, ssh_username=None,
            ssh_port=22, ssh_private_key=None, remote_python=None):
        self.service = None
        if local:
            super().__init__(processes=False)
        else:
            remote_addr = (remote_ip + ':{}'.format(port))
            self.service = start_service(remote_ip, port)
            _set_global_remote_service(self.service)
            import time
            time.sleep(10)
            super().__init__(remote_addr)
        with Remote.LOCK:
            self.remote_id = Remote.REMOTE_ID.value
            Remote.REMOTE_ID.value += 1

    def close(self, timeout=2):
        if self.service:
            self.service.shutdown()
        super().close(timeout)
        self.shutdown()

    def upload_files(self, files, **kwargs):
        for filename in files:
            self.upload_file(filename, **kwargs)

    def __repr__(self):
        reprstr = self.__class__.__name__ + ' REMOTE_ID: {}, \n\t'.format(self.remote_id) + \
            super().__repr__()
        return reprstr

class DaskRemoteService(object):
    def __init__(self, remote_addr, scheduler_port, ssh_username=None,
        ssh_port=22, ssh_private_key=None, remote_python=None):

        self.scheduler_addr = remote_addr
        self.scheduler_port = scheduler_port

        self.ssh_username = ssh_username
        self.ssh_port = ssh_port
        self.ssh_private_key = ssh_private_key
        self.remote_python = remote_python
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
        )
        self.start_monitoring()
        self.status = "live"

    def start_monitoring(self):
        if self.monitor_thread.is_alive():
            return
        self.monitor_thread = Thread(target=self.monitor_remote_processes)
        #self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def monitor_remote_processes(self):
        all_processes = [self.scheduler, self.worker]
        try:
            while True:
                for process in all_processes:
                    while not process["output_queue"].empty():
                        try:
                            msg = process["output_queue"].get()
                            if 'distributed.' in msg:
                                msg = msg.replace('distributed', 'autogluon')

                            print(msg)
                        except Exception as e:
                            print(f'Exception happend {e}, terminating the remote.')
                            break
                # Kill some time and free up CPU
                time.sleep(0.1)

        except KeyboardInterrupt:
            pass

    def shutdown(self):
        all_processes = [self.worker, self.scheduler]

        for process in all_processes:
            process["input_queue"].put("shutdown")
            process["thread"].join()
        self.status = "closed"

atexit.register(_close_global_remote_services)
