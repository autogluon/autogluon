import logging
import multiprocessing as mp
from .resource import *
from ...utils import Queue

__all__ = ['ResourceManager']

logger = logging.getLogger(__name__)

class ResourceManager(object):
    """Resource Manager to keep track of the cpu and gpu usage
    """
    LOCK = mp.Lock()
    CPU_QUEUE = Queue()
    GPU_QUEUE = Queue()
    MAX_CPU_COUNT = get_cpu_count()
    MAX_GPU_COUNT = get_gpu_count()
    for cid in range(MAX_CPU_COUNT):
        CPU_QUEUE.put(cid)
    for gid in range(MAX_GPU_COUNT):
        GPU_QUEUE.put(gid)

    @classmethod
    def _request(cls, resource):
        """ResourceManager, we recommand using scheduler instead of creating your own
        resource manager.
        """
        assert cls.check_possible(resource), \
            'Requested num_cpu={} and num_gpu={} should be less than or equal to' + \
            'system availability CPUs={}, GPUs={}'. \
            format(resource.num_cpus, resource.num_gpus, cls.MAX_GPU_COUNT, cls.MAX_CPU_COUNT)

        with cls.LOCK:
            cpu_ids = [cls.CPU_QUEUE.get() for i in range(resource.num_cpus)]
            gpu_ids = [cls.GPU_QUEUE.get() for i in range(resource.num_gpus)]
        resource._ready(cpu_ids, gpu_ids)
        logger.debug("Reqeust succeed {}".format(resource))
        return

    @classmethod
    def _release(cls, resource):
        cpu_ids = resource.cpu_ids
        gpu_ids = resource.gpu_ids
        resource._release()
        if len(cpu_ids) > 0:
            for cid in cpu_ids:
                cls.CPU_QUEUE.put(cid)
        if len(gpu_ids) > 0:
            for gid in gpu_ids:
                cls.GPU_QUEUE.put(gid)

    @classmethod
    def check_availability(cls, resource):
        """Unsafe check
        """
        if resource.num_cpus > self.CPU_QUEUE.qsize() or resource.num_gpus > self.GPU_QUEUE.qsize():
            return False
        return True

    @classmethod
    def check_possible(cls, resource):
        assert isinstance(resource, Resources), 'Only support autogluon.resource.Resources'
        if resource.num_cpus > cls.MAX_CPU_COUNT or resource.num_gpus > cls.MAX_GPU_COUNT:
            return False
        return True

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
            '{} CPUs, '.format(self.MAX_CPU_COUNT) + \
            '{} GPUs)'.format(self.MAX_GPU_COUNT)
        return reprstr
