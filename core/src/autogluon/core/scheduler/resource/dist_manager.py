import logging
import multiprocessing as mp
from .resource import *
from ...utils import Queue

__all__ = ['DistributedResourceManager', 'NodeResourceManager']

logger = logging.getLogger(__name__)

class DistributedResourceManager(object):
    LOCK = mp.Lock()
    REQUESTING_STACK = []
    MAX_CPU_COUNT = 0
    MAX_GPU_COUNT = 0
    NODE_RESOURCE_MANAGER = {}
    __instance = None
    def __new__(cls):
        # Singleton
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    @classmethod
    def add_remote(cls, remotes):
        """Enables dynamically removing nodes
        """
        remotes = remotes if isinstance(remotes, list) else [remotes]
        for remote in remotes:
            cls.NODE_RESOURCE_MANAGER[remote] = NodeResourceManager(remote)
        cls._refresh_resource()

    @classmethod
    def reserve_resource(cls, remote, resource):
        node_manager = cls.NODE_RESOURCE_MANAGER[remote]
        with cls.LOCK:
            if not node_manager.check_availability(resource):
                return False
            node_manager._request(remote, resource)
        logger.info('Reserved {} in {}'.format(resource, remote))
        return True

    @classmethod
    def release_reserved_resource(cls, remote, resource):
        node_manager = cls.NODE_RESOURCE_MANAGER[remote]
        node_manager._release(resource)
        cls._evoke_request()

    @classmethod
    def _refresh_resource(cls):
        cls.MAX_CPU_COUNT = max([x.get_all_resources()[0] for x in cls.NODE_RESOURCE_MANAGER.values()])
        cls.MAX_GPU_COUNT = max([x.get_all_resources()[1] for x in cls.NODE_RESOURCE_MANAGER.values()])

    @classmethod
    def _request(cls, resource):
        """ResourceManager, we recommand using scheduler instead of creating your own
        resource manager.
        """
        assert cls.check_possible(resource), \
            'Requested num_cpu={} and num_gpu={} should be less than or equal to' + \
            'largest node availability CPUs={}, GPUs={}'. \
            format(resource.num_cpus, resource.num_gpus, cls.MAX_GPU_COUNT, cls.MAX_CPU_COUNT)
       
        with cls.LOCK:
            node = cls.check_availability(resource)
            if node is not None:
                cls.NODE_RESOURCE_MANAGER[node]._request(node, resource)
                return

        logger.debug('Appending {} to Request Stack'.format(resource))
        request_semaphore = mp.Semaphore(0)
        with cls.LOCK:
            cls.REQUESTING_STACK.append((resource, request_semaphore))
        request_semaphore.acquire()
        return

    @classmethod
    def _release(cls, resource):
        logger.debug('\nReleasing resource {}'.format(resource))
        cls.NODE_RESOURCE_MANAGER[resource.node]._release(resource)
        cls._evoke_request()

    @classmethod
    def _evoke_request(cls):
        succeed = False
        with cls.LOCK:
            if len(cls.REQUESTING_STACK) > 0:
                resource, request_semaphore = cls.REQUESTING_STACK.pop()
                node = cls.check_availability(resource)
                if node is not None:
                    cls.NODE_RESOURCE_MANAGER[node]._request(node, resource)
                    logger.debug('\nEvoking requesting resource {}'.format(resource))
                    request_semaphore.release()
                    succeed = True
                else:
                    cls.REQUESTING_STACK.append((resource, request_semaphore))
                    return
        if succeed:
            cls._evoke_request()

    @classmethod
    def check_availability(cls, resource):
        """Unsafe check
        """
        candidate_nodes = cls._get_possible_nodes(resource)
        for node in candidate_nodes:
            if cls.NODE_RESOURCE_MANAGER[node].check_availability(resource):
                #logger.debug('\nSuccessfully find node {}'.format(node))
                return node
        return None

    @classmethod
    def check_possible(cls, resource):
        assert isinstance(resource, DistributedResource), \
            'Only support autogluon.resource.DistributedResource'
        if resource.num_cpus > cls.MAX_CPU_COUNT or resource.num_gpus > cls.MAX_GPU_COUNT:
            return False
        return True

    @classmethod
    def remove_remote(cls, remotes):
        #TODO 
        """Enables dynamically removing nodes
        """
        cls._refresh_resource()
        pass

    @classmethod
    def _get_possible_nodes(cls, resource):
        candidates = []
        for remote, manager in cls.NODE_RESOURCE_MANAGER.items():
            if manager.check_possible(resource):
                candidates.append(remote)
        return candidates

    def __repr__(self):
        reprstr = self.__class__.__name__ + '{\n'
        for remote, manager in self.NODE_RESOURCE_MANAGER.items():
            reprstr += '(Remote: {}, Resource: {})\n'.format(remote, manager)
        reprstr += '}'
        return reprstr


class NodeResourceManager(object):
    """Remote Resource Manager to keep track of the cpu and gpu usage
    """
    def __init__(self, remote):
        self.LOCK = mp.Lock()
        self.MAX_CPU_COUNT = get_remote_cpu_count(remote)
        self.MAX_GPU_COUNT = get_remote_gpu_count(remote)
        self.CPU_QUEUE = Queue()
        self.GPU_QUEUE = Queue()
        for cid in range(self.MAX_CPU_COUNT):
            self.CPU_QUEUE.put(cid)
        for gid in range(self.MAX_GPU_COUNT):
            self.GPU_QUEUE.put(gid)

    def _request(self, remote, resource):
        """ResourceManager, we recommand using scheduler instead of creating your own
        resource manager.
        """
        assert self.check_possible(resource), \
            'Requested num_cpu={} and num_gpu={} should be less than or equal to' + \
            'system availability CPUs={}, GPUs={}'. \
            format(resource.num_cpus, resource.num_gpus, self.MAX_GPU_COUNT, self.MAX_CPU_COUNT)

        with self.LOCK:
            cpu_ids = [self.CPU_QUEUE.get() for i in range(resource.num_cpus)]
            gpu_ids = [self.GPU_QUEUE.get() for i in range(resource.num_gpus)]
            resource._ready(remote, cpu_ids, gpu_ids)
            #logger.debug("\nReqeust succeed {}".format(resource))
            return
 
    def _release(self, resource):
        cpu_ids = resource.cpu_ids
        gpu_ids = resource.gpu_ids
        resource._release()
        if len(cpu_ids) > 0:
            for cid in cpu_ids:
                self.CPU_QUEUE.put(cid)
        if len(gpu_ids) > 0:
            for gid in gpu_ids:
                self.GPU_QUEUE.put(gid)

    def get_all_resources(self):
        return self.MAX_CPU_COUNT, self.MAX_GPU_COUNT

    def check_availability(self, resource):
        """Unsafe check
        """
        if resource.num_cpus > self.CPU_QUEUE.qsize() or resource.num_gpus > self.GPU_QUEUE.qsize():
            return False
        return True

    def check_possible(self, resource):
        assert isinstance(resource, DistributedResource), 'Only support autogluon.resource.Resources'
        if resource.num_cpus > self.MAX_CPU_COUNT or resource.num_gpus > self.MAX_GPU_COUNT:
            return False
        return True

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
            '{} CPUs, '.format(self.MAX_CPU_COUNT) + \
            '{} GPUs)'.format(self.MAX_GPU_COUNT)
        return reprstr
