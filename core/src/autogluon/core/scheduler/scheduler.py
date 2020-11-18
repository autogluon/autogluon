"""Distributed Task Scheduler"""
import contextlib
import os
import pickle
import logging
import shutil
import sys
import tempfile
from functools import partial

import dill
import distributed
from warnings import warn
import multiprocessing as mp
from collections import OrderedDict

from .remote import RemoteManager
from .resource import DistributedResourceManager
from .. import Task
from .reporter import *
from ..utils import AutoGluonWarning, AutoGluonEarlyStop, CustomProcess
from ..utils.multiprocessing_utils import is_fork_enabled

SYS_ERR_OUT_FILE = 'sys_err.out'
SYS_STD_OUT_FILE = 'sys_std.out'

logger = logging.getLogger(__name__)

__all__ = ['TaskScheduler']


@contextlib.contextmanager
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class ClassProperty(object):

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class TaskScheduler(object):
    """Base Distributed Task Scheduler
    """
    LOCK = mp.Lock()
    _resource_manager = None
    _remote_manager = None

    @ClassProperty
    def resource_manager(cls):
        if cls._resource_manager is None:
            cls._resource_manager = DistributedResourceManager()
        return cls._resource_manager

    @ClassProperty
    def remote_manager(cls):
        if cls._remote_manager is None:
            cls._remote_manager = RemoteManager()
        return cls._remote_manager

    def __init__(self, dist_ip_addrs=None):
        if dist_ip_addrs is None:
            dist_ip_addrs=[]
        cls = TaskScheduler
        remotes = cls.remote_manager.add_remote_nodes(dist_ip_addrs)
        cls.resource_manager.add_remote(cls.remote_manager.get_remotes())
        self.scheduled_tasks = []
        self.finished_tasks = []

    def add_remote(self, ip_addrs):
        """Add remote nodes to the scheduler computation resource.
        """
        ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
        with self.LOCK:
            remotes = TaskScheduler.remote_manager.add_remote_nodes(ip_addrs)
            TaskScheduler.resource_manager.add_remote(remotes)

    @classmethod
    def upload_files(cls, files, **kwargs):
        """Upload files to remote machines, so that they are accessible by import or load.
        """
        cls.remote_manager.upload_files(files, **kwargs)

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
            cls.resource_manager._request(task.resources)
        job = cls._start_distributed_job(task, cls.resource_manager)
        new_dict = self._dict_from_task(task)
        new_dict['Job'] = job
        with self.LOCK:
            self.scheduled_tasks.append(new_dict)

    def run_job(self, task):
        """Run a training task to the scheduler (Sync).
        """
        cls = TaskScheduler
        cls.resource_manager._request(task.resources)
        job = cls._start_distributed_job(task, cls.resource_manager)
        return job.result()

    @staticmethod
    def _start_distributed_job(task, resource_manager):
        """Async Execute the job in remote and release the resources
        """
        logger.debug('\nScheduling {}'.format(task))
        job = task.resources.node.submit(TaskScheduler._run_dist_job,
                                         task.fn, task.args, task.resources.gpu_ids)
        def _release_resource_callback(fut):
            logger.debug('Start Releasing Resource')
            resource_manager._release(task.resources)
        job.add_done_callback(_release_resource_callback)
        return job


    @staticmethod
    def _wrapper(tempdir, task, *args, **kwargs):
        with open(os.path.join(tempdir, SYS_STD_OUT_FILE), 'w') as std_out:
            with open(os.path.join(tempdir, SYS_ERR_OUT_FILE), 'w') as err_out:
                sys.stdout = std_out
                sys.stderr = err_out
                task(*args, **kwargs)

    @staticmethod
    def _wrap(tempdir, task):
        return partial(TaskScheduler._wrapper, tempdir, task)

    @staticmethod
    def _worker(pickled_fn, pickled_args, return_list, gpu_ids, args):
        """Worker function in the client
        """

        # Only fork mode allows passing non-picklable objects
        fn = pickled_fn if is_fork_enabled() else dill.loads(pickled_fn)
        args = {**pickled_args, **args} if is_fork_enabled() else {**dill.loads(pickled_args), **args}

        if len(gpu_ids) > 0:
            # handle GPU devices
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        # running
        try:
            ret = fn(**args)
        except AutoGluonEarlyStop:
            ret = None
        return_list.append(ret)


    @staticmethod
    def _run_dist_job(fn, args, gpu_ids):
        """Remote function Executing the task
        """
        if '_default_config' in args['args']:
            args['args'].pop('_default_config')

        if 'reporter' in args:
            local_reporter = LocalStatusReporter()
            dist_reporter = args['reporter']
            args['reporter'] = local_reporter

        manager = mp.Manager()
        return_list = manager.list()

        try:
            # Starting local process
            # Note: we have to use dill here because every argument passed to a child process over spawn or forkserver
            # has to be pickled. fork mode does not require this because memory sharing, but it is unusable for CUDA
            # applications (CUDA does not support fork) and multithreading issues (hanged threads).
            # Usage of decorators makes standard pickling unusable (https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)
            # Dill enables sending of decorated classes. Please note if some classes are used in the training function,
            # those classes are best be defined inside the function - this way those can be constructed 'on-the-other-side'
            # after deserialization.
            pickled_fn = fn if is_fork_enabled() else dill.dumps(fn)

            # Reporter has to be separated since it's used for cross-process communication and has to be passed as-is
            args_ = {k: v for (k, v) in args.items() if k not in ['reporter']}
            pickled_args = args_ if is_fork_enabled() else dill.dumps(args_)

            cross_process_args = {k: v for (k, v) in args.items() if k not in ['fn', 'args']}

            with make_temp_directory() as tempdir:
                p = CustomProcess(
                    target=TaskScheduler._wrap(tempdir, partial(TaskScheduler._worker, pickled_fn, pickled_args)),
                    args=(return_list, gpu_ids, cross_process_args)
                )
                p.start()
                if 'reporter' in args:
                    cp = Communicator.Create(p, local_reporter, dist_reporter)
                p.join()
                # Get processes outputs
                with open(os.path.join(tempdir, SYS_STD_OUT_FILE)) as f:
                    print(f.read(), file=sys.stdout, end = '')
                with open(os.path.join(tempdir, SYS_ERR_OUT_FILE)) as f:
                    print(f.read(), file=sys.stderr, end = '')
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
            str(self.resource_manager) +')\n'
        return reprstr
