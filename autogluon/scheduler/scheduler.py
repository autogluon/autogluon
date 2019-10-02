"""Distributed Task Scheduler"""
import os
import pickle
import logging
import subprocess
from threading import Thread
import multiprocessing as mp
from collections import namedtuple, OrderedDict

from .remote import RemoteManager
from .resource import DistributedResourceManager
from ..core import Task
from .reporter import StatusReporter, Communicator, DistSemaphore
from ..utils import DeprecationHelper

logger = logging.getLogger(__name__)

__all__ = ['TaskScheduler', 'DistributedTaskScheduler']

class TaskScheduler(object):
    """Distributed Task Scheduler

    Args:
        dist_ip_addrs (List): list of ip addresses for remote nodes

    Example:
        >>> def my_task():
        >>>     pass
        >>> resource = DistributedResource(num_cpus=2, num_gpus=1)
        >>> task = Task(my_task, {}, resource)
        >>> scheduler = TaskScheduler()
        >>> scheduler.add_task(task)
    """
    LOCK = mp.Lock()
    RESOURCE_MANAGER = DistributedResourceManager()
    REMOTE_MANAGER = None
    def __init__(self, dist_ip_addrs=[]):
        cls = TaskScheduler
        if cls.REMOTE_MANAGER is None:
            cls.REMOTE_MANAGER = RemoteManager()
            cls.RESOURCE_MANAGER.add_remote(
                cls.REMOTE_MANAGER.get_remotes())
        remotes = cls.REMOTE_MANAGER.add_remote_nodes(dist_ip_addrs)
        cls.RESOURCE_MANAGER.add_remote(remotes)
        self.scheduled_tasks = []
        self.finished_tasks = []
        self.env_sem = DistSemaphore(1)

    def add_remote(self, ip_addrs):
        ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
        with self.LOCK:
            remotes = TaskScheduler.REMOTE_MANAGER.add_remote_nodes(ip_addrs)
            TaskScheduler.RESOURCE_MANAGER.add_remote(remotes)

    @classmethod
    def upload_files(cls, files, **kwargs):
        """Upload files to remote machines, so that they are accessible by import or load. 
        """
        cls.REMOTE_MANAGER.upload_files(files, **kwargs)

    def add_task(self, task):
        """Adding a training task to the scheduler.

        Args:
            task (autogluon.scheduler.Task): a new trianing task
        """
        # adding the task
        cls = TaskScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        p = Thread(target=cls._start_distributed_task, args=(
                   task, cls.RESOURCE_MANAGER, self.env_sem))
        p.start()
        with self.LOCK:
            self.scheduled_tasks.append({'TASK_ID': task.task_id, 'Args': task.args,
                                         'Process': p})

    @staticmethod
    def _start_distributed_task(task, resource_manager, env_sem):
        logger.debug('\nScheduling {}'.format(task))
        job = task.resources.node.submit(TaskScheduler._run_dist_task,
                                         task.fn, task.args, task.resources.gpu_ids,
                                         env_sem)
        job.result()
        resource_manager._release(task.resources)

    @staticmethod
    def _run_dist_task(fn, args, gpu_ids, env_semaphore):
        """Executing the task
        """
        # create local communicator
        if 'reporter' in args:
            local_reporter = StatusReporter()
            dist_reporter = args['reporter']
            args['reporter'] = local_reporter
        # handle terminator
        terminator_semaphore = None
        if 'terminator_semaphore' in args:
            terminator_semaphore = args.pop('terminator_semaphore')
        try:
            env_semaphore.acquire()
            if len(gpu_ids) > 0:
                # handle GPU devices
                os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
                os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"
            # start local progress
            p = mp.Process(target=fn, kwargs=args)
            p.start()
            env_semaphore.release()
            if 'reporter' in args:
                cp = Communicator.Create(p, local_reporter, dist_reporter)
            if terminator_semaphore is not None:
                terminator_semaphore.acquire()
                if p.is_alive():
                    if 'kill' in dir(p):
                        p.kill()
                        p.join()
                    else:
                        subprocess.run(['kill', '-9', str(p.pid)])
                        subprocess.run(['kill', '-9', str(p.pid)])
                        p.join()
            else:
                p.join()
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
        #self._cleaning_tasks()
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

DistributedTaskScheduler = DeprecationHelper(TaskScheduler, 'DistributedTaskScheduler')
