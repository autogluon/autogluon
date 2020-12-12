"""Distributed Task Scheduler"""
import logging
import multiprocessing as mp
import pickle
import sys
import time
from collections import OrderedDict
from random import random

import distributed

from .jobs import DistributedJobRunner
from .managers import TaskManagers
from .. import Task
from ..utils.multiprocessing_utils import AtomicCounter

logger = logging.getLogger(__name__)

__all__ = ['TaskScheduler']


class TaskScheduler(object):
    """
    Base Distributed Task Scheduler
    """
    LOCK = mp.RLock()

    def __init__(self, dist_ip_addrs=None):
        self.managers = TaskManagers()
        self.managers.register_dist_ip_addrs(dist_ip_addrs)
        self.scheduled_tasks = []
        self.finished_tasks = []
        self._task_id_counter = AtomicCounter()


    def add_remote(self, ip_addrs):
        """
        Add remote nodes to the scheduler computation resource.
        """
        with self.LOCK:
            self.managers.add_remote(ip_addrs)

    def upload_files(self, files, **kwargs):
        """
        Upload files to remote machines, so that they are accessible by import or load.
        """
        self.managers.upload_files(files, **kwargs)

    def add_job(self, task, **kwargs):
        """
        Adding a training task to the scheduler.

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
        if not task.resources.is_ready:
            self.managers.request_resources(task.resources)

        job_runner = DistributedJobRunner(task, self.managers)
        job = job_runner.start_distributed_job()
        new_dict = self._dict_from_task(task)
        new_dict['Job'] = job
        with self.LOCK:
            self.scheduled_tasks.append(new_dict)

    def run_job(self, task):
        """
        Run a training task to the scheduler (Sync).
        """
        cls = TaskScheduler
        self.managers.request_resources(task.resources)
        job_runner = DistributedJobRunner(task, self.managers)
        job = job_runner.start_distributed_job()
        result = job.result()
        return result

    def _new_task(self, fn, args, resources):
        """
        Safely crates Task object with assigned unique task_id
        """
        task_id = self._task_id_counter.get_and_increment()
        task = Task(task_id, fn, args, resources)
        return task

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

    def join_jobs(self, timeout=None):
        """
        Wait all scheduled jobs to finish
        """
        self._cleaning_tasks()
        for task_dict in self.scheduled_tasks:
            try:
                if timeout:
                    task_dict['Job'].result(timeout=timeout)
                else:
                    while not task_dict['Job'].done():
                        task_dict['Job'].result(timeout=60)
            except distributed.TimeoutError as e:
                logger.error(str(e))
            except:
                logger.error("Unexpected error:", sys.exc_info()[0])
                raise
            self._clean_task_internal(task_dict)
        self._cleaning_tasks()

    def state_dict(self, destination=None):
        """
        Returns a dictionary containing a whole state of the Scheduler

        Examples
        --------
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['finished_tasks'] = pickle.dumps(self.finished_tasks)
        with self.LOCK:
            destination['TASK_ID'] = self._task_id_counter.get()
        return destination

    def load_state_dict(self, state_dict):
        """
        Load from the saved state dict.

        Examples
        --------
        >>> scheduler.load_state_dict(ag.load('checkpoint.ag'))
        """
        self.finished_tasks = pickle.loads(state_dict['finished_tasks'])
        self._task_id_counter.set(state_dict['TASK_ID'])
        logger.debug('\nLoading finished_tasks: {} '.format(self.finished_tasks))

    def _dict_from_task(self, task):
        if isinstance(task, Task):
            return {'TASK_ID': task.task_id, 'Args': task.args}
        else:
            assert isinstance(task, dict)
            return {'TASK_ID': task['TASK_ID'], 'Args': task['Args']}

    @property
    def num_finished_tasks(self):
        return len(self.finished_tasks)

    def __repr__(self):
        return f'{self.__class__.__name__}(\n{self.managers.resource_manager})\n'
