"""Distributed Task Scheduler"""
import logging
import pickle
import sys
from collections import OrderedDict
from warnings import warn

import distributed

from .jobs import DistributedJobRunner
from .managers import TaskManagers
from .. import Task
from ..utils import AutoGluonWarning

logger = logging.getLogger(__name__)

__all__ = ['TaskScheduler']


class ClassProperty(object):

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class TaskScheduler(object):
    """Base Distributed Task Scheduler
    """
    managers = TaskManagers()
    jobs = DistributedJobRunner()

    def __init__(self, dist_ip_addrs=None):
        cls = TaskScheduler
        cls.managers.register_dist_ip_addrs(dist_ip_addrs)
        self.scheduled_tasks = []
        self.finished_tasks = []

    def add_remote(self, ip_addrs):
        """Add remote nodes to the scheduler computation resource.
        """
        self.managers.add_remote(ip_addrs)

    @classmethod
    def upload_files(cls, files, **kwargs):
        """Upload files to remote machines, so that they are accessible by import or load.
        """
        cls.managers.upload_files(files, **kwargs)

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
        if not task.resources.is_ready:
            cls.managers.request_resources(task.resources)
        job = cls.jobs.start_distributed_job(task, cls.managers)
        new_dict = self._dict_from_task(task)
        new_dict['Job'] = job
        with self.managers.lock:
            self.scheduled_tasks.append(new_dict)

    def run_job(self, task):
        """Run a training task to the scheduler (Sync).
        """
        cls = TaskScheduler
        cls.managers.request_resources(task.resources)
        job = cls.jobs.start_distributed_job(task, cls.managers)
        return job.result()

    def _clean_task_internal(self, task_dict):
        pass

    def _cleaning_tasks(self):
        with self.managers.lock:
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
            except distributed.TimeoutError as e:
                logger.error(str(e))
            except:
                logger.error("Unexpected error:", sys.exc_info()[0])
                raise
            self._clean_task_internal(task_dict)
        self._cleaning_tasks()

    def shutdown(self):
        """shutdown() is now deprecated in favor of :func:`autogluon.done`.
        """
        warn("scheduler.shutdown() is now deprecated in favor of autogluon.done().",
             AutoGluonWarning)
        self.join_jobs()
        self.remote_manager.shutdown()

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
                  str(self.managers.resource_manager) + ')\n'
        return reprstr
