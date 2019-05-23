import logging
import multiprocessing as mp
from ..utils import cpu_count, gpu_count

logger = logging.getLogger(__name__)

class ResourceManager(object):
    """Resource Manager to keep track of the cpu and gpu usage
    """
    LOCK = mp.Lock()
    CPU_QUEUE = mp.Queue()
    GPU_QUEUE = mp.Queue()
    for cid in range(cpu_count()):
        CPU_QUEUE.put(cid)
    for gid in range(gpu_count()):
        GPU_QUEUE.put(gid)

    @classmethod
    def _request(cls, resource):
        # Despite ResourceManager is thread/process safe, we do recommand
        # using single scheduler.
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
    def get_available_resources(cls):
        cpu_count = self.CPU_QUEUE.qsize()
        gpu_count = self.GPU_QUEUE.qsize()
        return cpu_count, gpu_count


class Resources(object):
    def __init__(self, num_cpus, num_gpus):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.cpu_ids = []
        self.gpu_ids = []
        self.ready = False

    def _release(self):
        self.ready = False
        self.cpu_ids = []
        self.gpu_ids = []

    def _ready(self, cids, gids):
        self.cpu_ids = cids
        self.gpu_ids = gids
        self.ready = True

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' \
            + 'nCPUs = ' + str(self.num_cpus)
        if len(self.cpu_ids) > 0:
            reprstr += ', CPU_IDs = {' + str(self.cpu_ids) + '}'
        if self.num_gpus > 0:
            reprstr += ', nGPUs = ' + str(self.num_gpus)
        if len(self.gpu_ids) > 0:
            reprstr += ', GPU_IDs = {' + str(self.gpu_ids) + '}'
        reprstr += ')'
        return reprstr
