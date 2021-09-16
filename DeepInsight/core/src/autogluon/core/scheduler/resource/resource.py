import logging
from ...utils import get_gpu_count, get_cpu_count

__all__ = ['Resources', 'DistributedResource',
           'get_remote_cpu_count', 'get_remote_gpu_count',
           'get_gpu_count', 'get_cpu_count']

logger = logging.getLogger(__name__)


class Resources(object):
    """Resource for AutoGluon Scheduler :class:`autogluon.scheduler.TaskScheduler`

    Args:
        num_cpus (int): number of cpu cores required for the training task.
        num_gpus (int): number of gpu required for the training task.

    Example:
        >>> def my_task():
        >>>     pass
        >>> resource = Resources(num_cpus=2, num_gpus=0)
        >>> task = Task(my_task, {}, resource)
    """
    def __init__(self, num_cpus=1, num_gpus=0):
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

class DistributedResource(Resources):
    """Resource for AutoGluon Distributed Scheduler :class:`autogluon.distributed.DistributedTaskScheduler`

    Args:
        num_cpus (int): number of cpu cores required for the training task.
        num_gpus (int): number of gpu required for the training task.

    Example:
        >>> def my_task():
        >>>     pass
        >>> resource = DistributedResource(num_cpus=2, num_gpus=1)
        >>> task = Task(my_task, {}, resource)
    """
    def __init__(self, num_cpus=1, num_gpus=0):
        super(DistributedResource, self).__init__(num_cpus, num_gpus)
        self.node = None
        self.is_ready = False

    def _ready(self, remote, cids, gids):
        super(DistributedResource, self)._ready(cids, gids)
        self.node = remote
        self.is_ready = True
 
    def _release(self):
        super(DistributedResource, self)._release()
        self.node = None

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n\t'
        if self.node: reprstr  += 'Node = ' + str(self.node)
        reprstr  += '\n\tnCPUs = ' + str(self.num_cpus)
        if len(self.cpu_ids) > 0:
            reprstr += ', CPU_IDs = {' + str(self.cpu_ids) + '}'
        if self.num_gpus > 0:
            reprstr += ', nGPUs = ' + str(self.num_gpus)
        if len(self.gpu_ids) > 0:
            reprstr += ', GPU_IDs = {' + str(self.gpu_ids) + '}'
        reprstr += ')'
        return reprstr


def get_remote_cpu_count(node):
    ret = node.submit(get_cpu_count)
    return ret.result()


def get_remote_gpu_count(node):
    ret = node.submit(get_gpu_count)
    return ret.result()
