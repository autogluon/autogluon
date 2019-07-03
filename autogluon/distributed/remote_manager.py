import os
import socket
import logging
#import subprocess
from threading import Thread 
import multiprocessing as mp

from .remote import Remote
from ..utils import get_ip

__all__ = ['RemoteManager']

logger = logging.getLogger(__name__)

class RemoteManager(object):
    NODES = {}
    LOCK = mp.Lock()
    PORT_ID = mp.Value('i', 8780)
    MASTER_IP = None
    __instance = None
    def __new__(cls):
        # Singleton
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
            cls.MASTER_IP = get_ip()
            cls.start_local_node()
        return cls.__instance

    @classmethod
    def start_local_node(cls):
        port = cls.get_port_id()
        remote = Remote.create_local_node(cls.MASTER_IP, port)
        with cls.LOCK:
            cls.NODES[cls.MASTER_IP] = remote

    @classmethod
    def launch_each(cls, launch_fn, *args, **kwargs):
        for node in cls.NODES.values():
            node.submit(launch_fn, *args, **kwargs)

    @classmethod
    def get_remotes(cls):
        return list(cls.NODES.values())

    @classmethod
    def add_remote_nodes(cls, ip_addrs):
        ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
        remotes = []
        for node_ip in ip_addrs:
            if node_ip in cls.NODES.keys():
                logger.warning('Already added remote {}'.format(node_ip))
                continue
            port = cls.get_port_id()
            remote = Remote(node_ip, port)
            with cls.LOCK:
                cls.NODES[node_ip] = remote
            remotes.append(remote)
        return remotes
    
    @classmethod
    def shutdown(cls):
        for node in cls.NODES.values():
            node.close()

    @classmethod
    def get_port_id(cls):
        with cls.LOCK:
            cls.PORT_ID.value += 1
            return cls.PORT_ID.value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for node in cls.NODES.values():
            node.shutdown()

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n'
        for node in cls.NODES.values():
           reprstr += '{}, \n'.format(node)
        reprstr += ')\n'
        return reprstr
