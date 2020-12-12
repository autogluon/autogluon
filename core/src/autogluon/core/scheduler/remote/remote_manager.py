import logging
import multiprocessing as mp
import socket

from .remote import Remote
from ...utils import warning_filter

__all__ = ['RemoteManager']

from ...utils.multiprocessing_utils import AtomicCounter, RWLock, read_lock, write_lock

logger = logging.getLogger(__name__)


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


class RemoteManager(object):
    NODES = {}
    LOCK = RWLock()
    PORT_ID = AtomicCounter(8700)
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
    def get_master_node(cls):
        with read_lock(cls.LOCK):
            ip = cls.NODES[cls.MASTER_IP]
        return ip

    @classmethod
    def start_local_node(cls):
        port = cls.get_port_id()
        with warning_filter():
            remote = Remote(cls.MASTER_IP, port, local=True)
        with write_lock(cls.LOCK):
            cls.NODES[cls.MASTER_IP] = remote

    @classmethod
    def launch_each(cls, launch_fn, *args, **kwargs):
        with read_lock(cls.LOCK):
            for node in cls.NODES.values():
                node.submit(launch_fn, *args, **kwargs)

    @classmethod
    def get_remotes(cls):
        with read_lock(cls.LOCK):
            nodes = list(cls.NODES.values())
        return nodes

    @classmethod
    def upload_files(cls, files, **kwargs):
        if isinstance(files, str):
            files = [files]
        with read_lock(cls.LOCK):
            for node in cls.NODES.values():
                node.upload_files(files, **kwargs)

    @classmethod
    def add_remote_nodes(cls, ip_addrs):
        with write_lock(cls.LOCK):
            ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
            remotes = []
            for node_ip in ip_addrs:
                if node_ip in cls.NODES.keys():
                    logger.warning('Already added remote {}'.format(node_ip))
                    continue
                port = cls.get_port_id()
                remote = Remote(node_ip, port)
                cls.NODES[node_ip] = remote
                remotes.append(remote)
        return remotes
    
    @classmethod
    def shutdown(cls):
        with write_lock(cls.LOCK):
            for node in cls.NODES.values():
                node.shutdown()
            cls.NODES = {}
            cls.__instance = None

    @classmethod
    def get_port_id(cls):
        return cls.PORT_ID.increment_and_get()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        with read_lock(self.LOCK):
            for node in self.NODES.values():
                node.shutdown()

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n'
        for node in self.NODES.values():
           reprstr += '{}, \n'.format(node)
        reprstr += ')\n'
        return reprstr
