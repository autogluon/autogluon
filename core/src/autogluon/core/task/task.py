import copy
import logging
import argparse
import multiprocessing as mp

logger = logging.getLogger(__name__)
import time
import random

__all__ = ['Task']

class Task(object):
    """Individual training task, containing the lauch function, default arguments and
    required resources.

    Args:
        fn (callable): Lauch function for the training task.
        args (argparse.ArgumentParser): Default function arguments.
        resources (autogluon.scheduler.Resources): Required resources for lauching the task.

    Example:
        >>> def my_task():
        >>>     pass
        >>> resource = Resources(num_cpus=2, num_gpus=0)
        >>> task = Task(my_task, {}, resource)
    """
    TASK_ID = mp.Value('i', 0)
    LOCK = mp.Lock()

    def __init__(self, fn, args, resources):
        self.fn = fn
        self.args = copy.deepcopy(args)
        self.resources = resources
        with Task.LOCK:
            # Adding random jitter to reduce likelihood of tasks starting at the same time
            # This should significantly reduce generation of duplicate values (observed earlier)
            time.sleep(0.1 + random.random())
            self.task_id = Task.TASK_ID.value
            if 'args' in self.args:
                if isinstance(self.args['args'], (argparse.Namespace, argparse.ArgumentParser)):
                    args_dict = vars(self.args['args'])
                else:
                    args_dict = self.args['args']
                args_dict.update({'task_id': self.task_id})
            Task.TASK_ID.value += 1

    @classmethod
    def set_id(cls, taskid):
        logger.info('Setting TASK ID: {}'.format(taskid))
        with Task.LOCK:
            cls.TASK_ID.value = taskid

    def __repr__(self):
        reprstr = self.__class__.__name__ +  \
            ' (' + 'task_id: ' + str(self.task_id) + \
            ',\n\tfn: ' + str(self.fn) + \
            ',\n\targs: {'
        for k, v in self.args.items():
            data = str(v)
            info = (data[:100] + '..') if len(data) > 100 else data
            reprstr += '{}'.format(k) + ': ' + info + ', '
        reprstr += '},\n\tresource: ' + str(self.resources) + ')\n'
        return reprstr
