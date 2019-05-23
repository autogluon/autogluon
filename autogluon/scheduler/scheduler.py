"""Task Scheduler"""
import os
import pickle
import logging
import multiprocessing as mp
from collections import namedtuple, OrderedDict
from .resource_manager import ResourceManager

__all__ = ['TaskScheduler', 'Task']

logger = logging.getLogger(__name__)

BasicTask = namedtuple('Task', 'fn args resources')

class Task(BasicTask):
    """Individual training task, containing the lauch function, default arguments and
    required resources.
    Args:
        fn (callable): Lauch function for the training task.
        args (argparse.ArgumentParser): Default function arguments.
        resources (autogluon.scheduler.Resources): Required resources for lauching the task.
    """
    TASK_ID = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __new__(cls, fn, args, resources):
        self = super(Task, cls).__new__(cls, fn, args, resources)
        with Task.LOCK:
            self.task_id = Task.TASK_ID.value
            Task.TASK_ID.value += 1
        return self

    @classmethod
    def set_id(cls, taskid):
        logger.info('Seting TASK ID: {}'.format(taskid))
        cls.TASK_ID.value = taskid

    def __repr__(self):
        reprstr = self.__class__.__name__ +  \
            '(' + 'TASK_ID: ' + str(self.task_id) + ') ' + \
            super(Task, self).__repr__() + ')'
        return reprstr

class TaskScheduler(object):
    """Basic Task Scheduler w/o Searcher
    """
    LOCK = mp.Lock()
    RESOURCE_MANAGER = ResourceManager()
    ERROR_QUEUE = mp.Queue()
    def __init__(self):
        self.scheduler_tasks = []
        self.finished_tasks = []

    def add_task(self, task):
        """Adding a training task to the scheduler.
        Args:
            task (autogluon.scheduler.Task): a new trianing task
        """
        # adding the task
        logger.debug("Adding A New Task {}".format(task))
        TaskScheduler.RESOURCE_MANAGER._request(task.resources)
        p = mp.Process(target=TaskScheduler._run_task, args=(
                       task.fn, task.args, task.resources,
                       TaskScheduler.RESOURCE_MANAGER))
        p.start()
        with self.LOCK:
            self.scheduler_tasks.append({'TASK_ID': task.task_id, 'Config': task.args['config'], 'Process': p})

    @staticmethod
    def _run_task(fn, args, resources, resource_manager):
        """Executing the task
        """
        if resources.num_gpus > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str,resources.gpu_ids))
        try:
            fn(**args)
        except Exception as e:
            logging.error(
                'Uncaught exception in worker process: {}'.format(e))
            TaskScheduler.ERROR_QUEUE.put(e)
        resource_manager._release(resources)

    def _cleaning_tasks(self):
        with self.LOCK:
            for i, task_dick in enumerate(self.scheduler_tasks):
                if not task_dick['Process'].is_alive():
                    task_dict = self.scheduler_tasks.pop(i)
                    self.finished_tasks.append({'TASK_ID': task_dict['TASK_ID'],
                                               'Config': task_dict['Config']})

    def join_tasks(self):
        self._cleaning_tasks()
        for i, task_dic in enumerate(self.scheduler_tasks):
            task_dic['Process'].join()
        while not TaskScheduler.ERROR_QUEUE.empty():
            e = TaskScheduler.ERROR_QUEUE.get()
            logger.error(str(e))

    def state_dict(self, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler
        """
        self._cleaning_tasks()
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        logger.debug('State_Dict self.finished_tasks: {}'.format(self.finished_tasks))
        destination['finished_tasks'] = pickle.dumps(self.finished_tasks)
        destination['TASK_ID'] = Task.TASK_ID.value
        return destination

    @classmethod
    def load_state_dict(self, state_dict):
        self.finished_tasks = pickle.loads(state_dict['finished_tasks'])
        Task.set_id(state_dict['TASK_ID'])
        logger.debug('Loading finished_tasks: {} '.format(self.finished_tasks))

    @property
    def num_finished_tasks(self):
        return len(self.finished_tasks)
