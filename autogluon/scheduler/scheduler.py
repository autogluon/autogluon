"""Distributed Task Scheduler"""
import os
import pickle
import logging
import subprocess
from warnings import warn
from threading import Thread
import multiprocessing as mp
from collections import OrderedDict

from .remote import RemoteManager
from .resource import DistributedResourceManager
from ..core import Task
from .reporter import StatusReporter, Communicator, DistSemaphore
from ..utils import DeprecationHelper, AutoGluonWarning

logger = logging.getLogger(__name__)

__all__ = ['TaskScheduler']

class TaskScheduler(object):
    """Base Distributed Task Scheduler
    """
    LOCK = mp.Lock()
    RESOURCE_MANAGER = DistributedResourceManager()
    REMOTE_MANAGER = RemoteManager()
    def __init__(self, dist_ip_addrs=None):
        if dist_ip_addrs is None:
            dist_ip_addrs=[]
        cls = TaskScheduler
        remotes = cls.REMOTE_MANAGER.add_remote_nodes(dist_ip_addrs)
        cls.RESOURCE_MANAGER.add_remote(cls.REMOTE_MANAGER.get_remotes())
        self.scheduled_tasks = []
        self.finished_tasks = []
        self.env_sem = DistSemaphore(1)

    def add_remote(self, ip_addrs):
        """Add remote nodes to the scheduler computation resource.
        """
        ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
        with self.LOCK:
            remotes = TaskScheduler.REMOTE_MANAGER.add_remote_nodes(ip_addrs)
            TaskScheduler.RESOURCE_MANAGER.add_remote(remotes)

    @classmethod
    def upload_files(cls, files, **kwargs):
        """Upload files to remote machines, so that they are accessible by import or load. 
        """
        cls.REMOTE_MANAGER.upload_files(files, **kwargs)

    def _dict_from_task(self, task):
        if isinstance(task, Task):
            return {'TASK_ID': task.task_id, 'Args': task.args}
        else:
            assert isinstance(task, dict)
            return {'TASK_ID': task['TASK_ID'], 'Args': task['Args']}

    def add_task(self, task, **kwargs):
        """add_task() is now deprecated in favor of add_job().
        """
        warn("scheduler.add_task() is now deprecated in favor of scheduler.add_job().",
             AutoGluonWarning)
        self.add_job(task, **kwargs)

    def add_job(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new training task

        Relevant entries in kwargs:
        - bracket: HB bracket to be used. Has been sampled in _promote_config
        - new_config: If True, task starts new config eval, otherwise it promotes
          a config (only if type == 'promotion')
        Only if new_config == False:
        - config_key: Internal key for config
        - resume_from: config promoted from this milestone
        - milestone: config promoted to this milestone (next from resume_from)
        """
        # adding the task
        cls = TaskScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        job = cls._start_distributed_job(task, cls.RESOURCE_MANAGER, self.env_sem)
        with self.LOCK:
            new_dict = self._dict_from_task(task)
            new_dict['Job'] = job
            self.scheduled_tasks.append(new_dict)

    def run_job(self, task):
        """Run a training task to the scheduler (Sync).
        """
        cls = TaskScheduler
        cls.RESOURCE_MANAGER._request(task.resources)
        job = cls._start_distributed_job(task, cls.RESOURCE_MANAGER, self.env_sem)
        return job.result()

    @staticmethod
    def _start_distributed_job(task, resource_manager, env_sem):
        logger.debug('\nScheduling {}'.format(task))
        job = task.resources.node.submit(TaskScheduler._run_dist_job,
                                         task.fn, task.args, task.resources.gpu_ids,
                                         env_sem)
        def _release_resource_callback(fut):
            resource_manager._release(task.resources)
        job.add_done_callback(_release_resource_callback)
        return job

    @staticmethod
    def _run_dist_job(fn, args, gpu_ids, env_semaphore):
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

        manager = mp.Manager()
        return_list = manager.list()
        def _worker(return_list, **kwargs):
            ret = fn(**kwargs)
            return_list.append(ret)

        try:
            env_semaphore.acquire()
            if len(gpu_ids) > 0:
                # handle GPU devices
                os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
                os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"
            # start local progress
            p = mp.Process(target=_worker, args=(return_list,), kwargs=args)
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
                        try:
                            subprocess.run(['kill', '-9', str(p.pid)])
                            subprocess.run(['kill', '-9', str(p.pid)])
                        except Exception:
                            pass
                        p.join()
            else:
                p.join()
        except Exception as e:
            logger.error('Exception in worker process: {}'.format(e))
        ret = return_list[0] if len(return_list) > 0 else None
        return ret


    def _clean_task_internal(self, task_dict):
        pass

    def _cleaning_tasks(self):
        with self.LOCK:
            new_scheduled_tasks = []
            for task_dict in self.scheduled_tasks:
                if task_dict['Job'].done():
                    self._clean_task_internal(task_dict)
                    self.finished_tasks.append(self._dict_from_task(task_dict))
                else:
                    new_scheduled_tasks.append(task_dict)
            if len(new_scheduled_tasks) < len(self.scheduled_tasks):
                self.scheduled_tasks = new_scheduled_tasks

    def join_tasks(self):
        warn("scheduler.join_tasks() is now deprecated in favor of scheduler.join_jobs().",
             AutoGluonWarning)
        self.join_jobs()

    def join_jobs(self, timeout=None):
        """Wait all scheduled jobs to finish
        """
        self._cleaning_tasks()
        for task_dict in self.scheduled_tasks:
            try:
                task_dict['Job'].result(timeout=timeout)
            except TimeoutError as e:
                logger.error(str(e))
            self._clean_task_internal(task_dict)
        self._cleaning_tasks()

    def shutdown(self):
        """shutdown() is now deprecated in favor of :func:`autogluon.done`.
        """
        warn("scheduler.shutdown() is now deprecated in favor of autogluon.done().",
             AutoGluonWarning)
        self.join_jobs()
        self.REMOTE_MANAGER.shutdown()

    def state_dict(self, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler

        Examples
        --------
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['finished_tasks'] = pickle.dumps(self.finished_tasks)
        destination['TASK_ID'] = Task.TASK_ID.value
        return destination

    def load_state_dict(self, state_dict):
        """Load from the saved state dict.

        Examples
        --------
        >>> scheduler.load_state_dict(ag.load('checkpoint.ag'))
        """
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
