import logging
import multiprocessing as mp
import os
import sys
from functools import partial

import dill

from autogluon.core.scheduler.managers import TaskManagers
from autogluon.core.scheduler.reporter import Communicator, LocalStatusReporter
from autogluon.core.utils import CustomProcess, AutoGluonEarlyStop
from autogluon.core.utils.files import make_temp_directory
from autogluon.core.utils.multiprocessing_utils import is_fork_enabled

logger = logging.getLogger(__name__)

__all__ = ['DistributedJobRunner']

SYS_ERR_OUT_FILE = 'sys_err.out'
SYS_STD_OUT_FILE = 'sys_std.out'


class DistributedJobRunner(object):

    @classmethod
    def start_distributed_job(cls, task, manager: TaskManagers):
        """Async Execute the job in remote and release the resources
        """
        logger.debug('\nScheduling {}'.format(task))
        job = task.resources.node.submit(cls._run_dist_job, task.task_id, task.fn, task.args, task.resources.gpu_ids)

        def _release_resource_callback(fut):
            logger.debug('Start Releasing Resource')
            manager.release_resources(task.resources)

        job.add_done_callback(_release_resource_callback)
        return job

    @classmethod
    def _worker(cls, tempdir, task_id, pickled_fn, pickled_args, return_list, gpu_ids, args):
        """Worker function in the client
        """

        with open(os.path.join(tempdir, f'{task_id}.out'), 'w') as std_out:
            with open(os.path.join(tempdir, f'{task_id}.err'), 'w') as err_out:

                # redirect stdout/strerr into a file so the main process can read it after the job is completed
                if not is_fork_enabled():
                    sys.stdout = std_out
                    sys.stderr = err_out

                # Only fork mode allows passing non-picklable objects
                fn = pickled_fn if is_fork_enabled() else dill.loads(pickled_fn)
                args = {**pickled_args, **args} if is_fork_enabled() else {**dill.loads(pickled_args), **args}

                DistributedJobRunner.set_cuda_environment(gpu_ids)

                # running
                try:
                    ret = fn(**args)
                except AutoGluonEarlyStop:
                    ret = None
                return_list.append(ret)

                sys.stdout.flush()
                sys.stderr.flush()

    @classmethod
    def set_cuda_environment(cls, gpu_ids):
        if len(gpu_ids) > 0:
            # handle GPU devices
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


    @classmethod
    def _run_dist_job(cls, task_id, fn, args, gpu_ids):
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
                    target=partial(cls._worker, tempdir, task_id, pickled_fn, pickled_args),
                    args=(return_list, gpu_ids, cross_process_args)
                )
                p.start()
                if 'reporter' in args:
                    cp = Communicator.Create(p, local_reporter, dist_reporter)
                p.join()
                # Get processes outputs
                if not is_fork_enabled():
                    cls.__print(tempdir, task_id, 'out')
                    cls.__print(tempdir, task_id, 'err')
        except Exception as e:
            logger.error('Exception in worker process: {}'.format(e))
        ret = return_list[0] if len(return_list) > 0 else None
        return ret

    @classmethod
    def __print(cls, tempdir, task_id, out):
        with open(os.path.join(tempdir, f'{task_id}.{out}')) as f:
            out = f.read()
            file = sys.stderr if out is 'err' else sys.stdout
            if out:
                print(f'(task:{task_id})\t{out}', file=file, end='')
