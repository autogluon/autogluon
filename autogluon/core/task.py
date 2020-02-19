import argparse
import copy
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)

__all__ = ['Task']


class Task:
    """Individual training task, containing the launch function, default arguments and
    required resources.

    Args:
        fn (callable): Launch function for the training task.
        args (argparse.ArgumentParser): Default function arguments.
        resources (autogluon.scheduler.Resources): Required resources for launching the task.

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
        logger.info(f'Seting TASK ID: {taskid}')
        cls.TASK_ID.value = taskid

    def __repr__(self):
        reprstr = (
            f'{self.__class__.__name__} ('
            f'task_id: {self.task_id},'
            f'\n\tfn: {self.fn},'
            f'\n\targs: {{'
        )

        for k, v in self.args.items():
            data = str(v)
            info = (data[:100] + '..') if len(data) > 100 else data
            reprstr += f'{k}: {info}, '
        reprstr += f'}},\n\tresource: {self.resources})\n'
        return reprstr
