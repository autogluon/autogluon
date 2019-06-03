import os
import socket
import logging
#import subprocess
from threading import Thread 
import multiprocessing as mp

from ..resource import DistributedResourceManager
from .remote import Remote

__all__ = ['RemoteManager']

logger = logging.getLogger(__name__)

class RemoteManager(object):
    NODES = []
    LOCK = mp.Lock()
    PORT_ID = mp.Value('i', 8780)
    MASTER_IP = socket.gethostbyname(socket.gethostname())
    def __init__(self, ip_addrs=[]):
        RemoteManager.start_local_node()
        for ip_addr in ip_addrs:
            RemoteManager.add_remote_node(ip_addr)

    @classmethod
    def start_local_node(cls):
        port = cls.get_port_id()
        remote = Remote(cls.MASTER_IP, port, local=True)
        cls.NODES.append(remote)

    @classmethod
    def add_remote_node(cls, node_ip):
        port = cls.get_port_id()
        remote = Remote(node_ip, port)
        cls.NODES.append(remote)
    
    @classmethod
    def shutdown(cls):
        for node in cls.NODES:
            node.shutdown()

    @classmethod
    def get_port_id(cls):
        with cls.LOCK:
            cls.PORT_ID.value += 1
            return cls.PORT_ID.value

    @classmethod
    def get_remotes(cls):
        return cls.NODES

    @classmethod
    def create_resource_mamager(cls):
        return DistributedResourceManager(cls.NODES)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for node in self.NODES:
            node.shutdown()

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n'
        for node in self.NODES:
           reprstr += '{}, \n'.format(node)
        reprstr += ')\n'
        return reprstr
