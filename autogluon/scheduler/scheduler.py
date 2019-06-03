"""Task Scheduler"""
import os
import copy
import pickle
import logging
import argparse
import multiprocessing as mp
from collections import namedtuple, OrderedDict
from ..resource import ResourceManager

__all__ = ['TaskScheduler', 'Task']

logger = logging.getLogger(__name__)

#BasicTask = namedtuple('Task', 'fn args resources')

class Task(object):
    """Individual training task, containing the lauch function, default arguments and
    required resources.
    Args:
        fn (callable): Lauch function for the training task.
        args (argparse.ArgumentParser): Default function arguments.
        resources (autogluon.scheduler.Resources): Required resources for lauching the task.
    """
    TASK_ID = mp.Value('i', 0)
    LOCK = mp.Lock()
    #def __new__(cls, fn, args, resources):
    def __init__(self, fn, args, resources):
        #self = super(Task, cls).__new__(cls, fn, args, resources)
        self.fn = fn
        self.args = copy.deepcopy(args)
        self.resources = resources
        with Task.LOCK:
            self.task_id = Task.TASK_ID.value
            #logger.debug('creating task_id: {}'.format(self.task_id))
            if 'args' in self.args:
                vars(self.args['args']).update({'task_id': self.task_id})
            Task.TASK_ID.value += 1
        #return self

    @classmethod
    def set_id(cls, taskid):
        logger.info('Seting TASK ID: {}'.format(taskid))
        cls.TASK_ID.value = taskid

    def __repr__(self):
        reprstr = self.__class__.__name__ +  \
            ' (' + 'task_id: ' + str(self.task_id) + \
            ',\n\tfn: ' + str(self.fn) + \
            ',\n\targs: {'
        for k, v in self.args.items():
            reprstr +=  '{}'.format(k) + ': ' + str(v) + ', '
            #if isinstance(v, argparse.ArgumentParser):
            #    reprstr +=  '{}'.format(k) + ': ' + str(vars(v)) + ', '
            #else:
            #    reprstr +=  '{}'.format(k) + ': ' + str(v) + ', '
        reprstr +=  '},\n\tresource: ' + str(self.resources) + ')\n'
        return reprstr

class TaskScheduler(object):
    """Basic Task Scheduler
    """
    LOCK = mp.Lock()
    RESOURCE_MANAGER = ResourceManager()
    def __init__(self):
        self.scheduled_tasks = []
        self.finished_tasks = []

    def add_task(self, task):
        """Adding a training task to the scheduler.
        Args:
            task (autogluon.scheduler.Task): a new trianing task
        """
        # adding the task
        logger.debug("\nAdding A New Task {}".format(task))
        TaskScheduler.RESOURCE_MANAGER._request(task.resources)
        p = mp.Process(target=TaskScheduler._run_task, args=(
                       task.fn, task.args, task.resources,
                       TaskScheduler.RESOURCE_MANAGER))
        p.start()
        with self.LOCK:
            self.scheduled_tasks.append({'TASK_ID': task.task_id, 'Args': task.args,
                                         'Process': p})

    @staticmethod
    def _run_task(fn, args, resources, resource_manager):
        """Executing the task
        """
        if resources.num_gpus > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str,resources.gpu_ids))
        try:
            fn(**args)
        except Exception as e:
            logger.error(
                'Uncaught exception in worker process: {}'.format(e))
        resource_manager._release(resources)

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
        #while not TaskScheduler.ERROR_QUEUE.empty():
        #    e = TaskScheduler.ERROR_QUEUE.get()
        #    logger.error(str(e))

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
