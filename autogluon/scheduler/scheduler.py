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
        logger.info('Seeting TASK ID: {}'.format(taskid))
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
    SCHEDULED_TASKS = []
    FINISHED_TASKS = []
    ERROR_QUEUE = mp.Queue()
    def add_task(self, task):
        """Adding a training task to the scheduler.
        Args:
            task (autogluon.scheduler.Task): a new trianing task
        """
        # adding the task
        logger.debug("Adding A New Task {}".format(task))
        TaskScheduler.RESOURCE_MANAGER._request(task.resources)
        if task.resources.num_gpus > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(task.resources.gpu_ids)[1:-1]
        p = mp.Process(target=TaskScheduler._run_task, args=(
                       task.fn, task.args, task.resources,
                       TaskScheduler.RESOURCE_MANAGER))
        p.start()
        with self.LOCK:
            self.SCHEDULED_TASKS.append({'TASK_ID': task.task_id, 'Config': task.args['config'], 'Process': p})

    @staticmethod
    def _run_task(fn, args, resources, resource_manager):
        """Executing the task
        """
        try:
            fn(**args)
        except Exception as e:
            logging.error(
                'Uncaught exception in worker process: {}'.format(e))
            TaskScheduler.ERROR_QUEUE.put(e)
        resource_manager._release(resources)

    @classmethod
    def _cleaning_tasks(cls):
        with cls.LOCK:
            for i, task_dick in enumerate(cls.SCHEDULED_TASKS):
                if not task_dick['Process'].is_alive():
                    task_dict = cls.SCHEDULED_TASKS.pop(i)
                    cls.FINISHED_TASKS.append({'TASK_ID': task_dict['TASK_ID'],
                                               'Config': task_dict['Config']})

    @classmethod
    def join_tasks(cls):
        cls._cleaning_tasks()
        for i, task_dic in enumerate(cls.SCHEDULED_TASKS):
            task_dic['Process'].join()

    @classmethod
    def state_dict(cls, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler
        """
        cls._cleaning_tasks()
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        logger.debug('State_Dict cls.FINISHED_TASKS: {}'.format(cls.FINISHED_TASKS))
        destination['FINISHED_TASKS'] = pickle.dumps(cls.FINISHED_TASKS)
        destination['TASK_ID'] = Task.TASK_ID.value
        return destination

    @classmethod
    def load_state_dict(cls, state_dict):
        cls.FINISHED_TASKS = pickle.loads(state_dict['FINISHED_TASKS'])
        Task.set_id(state_dict['TASK_ID'])
        logger.debug('Loading FINISHED_TASKS: {} '.format(cls.FINISHED_TASKS))

    @property
    def num_finished_tasks(self):
        return len(self.FINISHED_TASKS)
