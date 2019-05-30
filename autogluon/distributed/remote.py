import os
import sys
import socket
import signal 
import logging
#import subprocess
from threading import Thread 
import multiprocessing as mp
from distributed import Scheduler, Worker, Client
from tornado.ioloop import IOLoop

from ..resource import DistributedResourceManager

__all__ = ['RemoteManager', 'Remote']

logger = logging.getLogger(__name__)

class RemoteManager(object):
    NODES = []
    LOCK = mp.Lock()
    DASK_SCHEDULED_WORKER = []
    PORT_ID = mp.Value('i', 8780)
    MASTER_IP = socket.gethostbyname(socket.gethostname())
    def __init__(self, ip_addrs=[]):
        RemoteManager.start_local_node()
        for ip_addr in ip_addrs:
            RemoteManager.add_remote_node(ip_addr)

    @classmethod
    def exit(cls):
        os.system('pkill dask-scheduler')
        os.system('pkill dask-worker')
        os.system('pkill dask-ssh')

    @classmethod
    def get_port_id(cls):
        with cls.LOCK:
            cls.PORT_ID.value += 1
            return cls.PORT_ID.value

    @classmethod
    def get_remotes(cls):
        return cls.NODES

    @classmethod
    def create_dist_resource_mamager(cls):
        return DistributedResourceManager(cls.NODES)

    @classmethod
    def start_local_node(cls):
        port = cls.get_port_id()
        logger.debug('dask-scheduler --port {}'.format(port))
        p1 = mp.Process(target=RemoteManager._start_dask_local_scheduler,
                    args=(port,), daemon=False)
        p1.start()
        logger.debug('dask-worker {}:{} --no-nanny'.format(cls.MASTER_IP, port))
        p2 = mp.Process(target=RemoteManager._start_dask_local_worker,
                    args=(cls.MASTER_IP, port), daemon=False)
        p2.start()
        cls.DASK_SCHEDULED_WORKER.append(p1)
        cls.DASK_SCHEDULED_WORKER.append(p2)
        # add local dask client
        remote = Remote(cls.MASTER_IP, port)
        cls.NODES.append(remote)

    @classmethod
    def add_remote_node(cls, node_ip):
        port = cls.get_port_id()
        # start remote dask worker and scheduler
        logger.debug('dask-ssh {} --scheduler-port {}'.format(
                     node_ip, port))
        p = mp.Process(target=RemoteManager._start_dask_remote_scheduler,
                       args=(node_ip, port), daemon=False)
        p.start()
        cls.DASK_SCHEDULED_WORKER.append(p)
        # add local dask client
        remote = Remote(node_ip, port)
        cls.NODES.append(remote)
    
    @staticmethod
    def _start_dask_local_worker(master_ip, port):
        os.system('dask-worker {}:{} --no-nanny'.format(master_ip, port))

    @staticmethod
    def _start_dask_local_scheduler(port):
        os.system('dask-scheduler --port {}'.format(port))

    @staticmethod
    def _start_dask_remote_scheduler(node_ip, port):
        os.system('dask-ssh {} --scheduler-port {}'.format(
                  node_ip, port))

    @staticmethod
    def _start_dask_worker(master_ip, port):
        pass

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n\n'
        for node in self.NODES:
           reprstr += '{}, \n\n'.format(node)
        reprstr += ')\n\n'
        return reprstr

class Remote(Client):
    LOCK = mp.Lock()
    REMOTE_ID = mp.Value('i', 0)
    def __init__(self, master_ip=None, port=None):
        remote_addr = (master_ip + ':{}'.format(port)) if port else None
        super(Remote, self).__init__(remote_addr)
        self.ip = master_ip
        self.port = port
        with Remote.LOCK:
            self.remote_id = Remote.REMOTE_ID.value
            Remote.REMOTE_ID.value += 1

    def __repr__(self):
        reprstr = self.__class__.__name__ + ' REMOTE_ID: {}, '.format(self.remote_id) + \
            super(Remote, self).__repr__()
        return reprstr
