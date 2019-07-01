"""Distributed Task Scheduler"""
import os
import pickle
import logging
from threading import Thread
import multiprocessing as mp
from collections import namedtuple, OrderedDict

from .remote_manager import RemoteManager
from ..resource import DistributedResourceManager
from ..scheduler import Task
from .dist_reporter import Communicator
from ..scheduler.reporter import StatusReporter

logger = logging.getLogger(__name__)

__all__ = ['DistributedTaskScheduler']

class DistributedTaskScheduler(object):
    """Distributed Task Scheduler

    Args:
        dist_ip_addrs (List): list of ip addresses for remote nodes

    Example:
        >>> def my_task():
        >>>     pass
        >>> resource = DistributedResource(num_cpus=2, num_gpus=1)
        >>> task = Task(my_task, {}, resource)
        >>> scheduler = DistributedTaskScheduler()
        >>> scheduler.add_task(task)
    """
    LOCK = mp.Lock()
    RESOURCE_MANAGER = DistributedResourceManager()
    REMOTE_MANAGER = None
    def __init__(self, dist_ip_addrs=[]):
        if DistributedTaskScheduler.REMOTE_MANAGER is None:
            DistributedTaskScheduler.REMOTE_MANAGER = RemoteManager()
            DistributedTaskScheduler.RESOURCE_MANAGER.add_remote(
                DistributedTaskScheduler.REMOTE_MANAGER.get_remotes())
        remotes = DistributedTaskScheduler.REMOTE_MANAGER.add_remote_nodes(dist_ip_addrs)
        DistributedTaskScheduler.RESOURCE_MANAGER.add_remote(remotes)
        self.scheduled_tasks = []
        self.finished_tasks = []

    def add_remote(self, ip_addrs):
        ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
        with self.LOCK:
            remotes = DistributedTaskScheduler.REMOTE_MANAGER.add_remote_nodes(ip_addrs)
            DistributedTaskScheduler.RESOURCE_MANAGER.add_remote(remotes)

    def add_task(self, task):
        """Adding a training task to the scheduler.

        Args:
            task (autogluon.scheduler.Task): a new trianing task
        """
        # adding the task
        DistributedTaskScheduler.RESOURCE_MANAGER._request(task.resources)
        p = Thread(target=DistributedTaskScheduler._start_distributed_task, args=(
                   task, DistributedTaskScheduler.RESOURCE_MANAGER))
        p.start()
        with self.LOCK:
            self.scheduled_tasks.append({'TASK_ID': task.task_id, 'Args': task.args,
                                         'Process': p})

    @staticmethod
    def _start_distributed_task(task, resource_manager):
        logger.debug('\nScheduling {}'.format(task))
        job = task.resources.node.submit(DistributedTaskScheduler._run_dist_task,
                                         task.fn, task.args, task.resources.gpu_ids)
        job.result()
        resource_manager._release(task.resources)

    @staticmethod
    def _run_dist_task(fn, args, gpu_ids):
        """Executing the task
        """
        if len(gpu_ids) > 0:
            # handle GPU devices
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
        try:
            # create local communicator
            if 'reporter' in args:
                local_reporter = StatusReporter()
                dist_reporter = args['reporter']
                args['reporter'] = local_reporter
            # start local progress
            p = mp.Process(target=fn, kwargs=args)
            p.start()
            if 'reporter' in args:
                cp = Communicator.Create(p, local_reporter, dist_reporter)
            p.join()
            #fn(**args)
        except Exception as e:
            logger.error('Exception in worker process: {}'.format(e))

    def _cleaning_tasks(self):
        with self.LOCK:
            for i, task_dick in enumerate(self.scheduled_tasks):
                if not task_dick['Process'].is_alive():
                    task_dict = self.scheduled_tasks.pop(i)
                    self.finished_tasks.append({'TASK_ID': task_dict['TASK_ID'],
                                               'Args': task_dict['Args']})

    def join_tasks(self):
        self._cleaning_tasks()
        for i, task_dic in enumerate(self.scheduled_tasks):
            task_dic['Process'].join()

    def shutdown(self):
        self.join_tasks()
        self.REMOTE_MANAGER.shutdown()

    def state_dict(self, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler
        """
        self._cleaning_tasks()
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        logger.debug('\nState_Dict self.finished_tasks: {}'.format(self.finished_tasks))
        destination['finished_tasks'] = pickle.dumps(self.finished_tasks)
        destination['TASK_ID'] = Task.TASK_ID.value
        return destination

    def load_state_dict(self, state_dict):
        self.finished_tasks = pickle.loads(state_dict['finished_tasks'])
        Task.set_id(state_dict['TASK_ID'])
        logger.debug('\nLoading finished_tasks: {} '.format(self.finished_tasks))

    @property
    def num_finished_tasks(self):
        return len(self.finished_tasks)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n' + \
            str(self.RESOURCE_MANAGER) +')\n'
        return reprstr
