import os
import logging
import threading
import multiprocessing as mp

from .scheduler import *
from .resource_manager import Resources
from .reporter import StatusReporter

__all__ = ['FIFO_Scheduler']

logger = logging.getLogger(__name__)

class FIFO_Scheduler(TaskScheduler):
    """Simple scheduler that just runs trials in submission order.
    Args:
        train_fn (callable): A task launch function for training.
            Note: please add the `@autogluon_method` decorater to the original function.
        args (object): Default arguments for launching train_fn.
        resource (dict): Computation resources.
            For example, `{'num_cpus':2, 'num_gpus':1}`
        searcher (object): Autogluon searcher.
            For example, autogluon.searcher.RandomSampling
    """
    def __init__(self, train_fn, args, resource, searcher):
        self.train_fn = train_fn
        self.args = args
        self.resource = resource
        self.searcher = searcher

    def run(self, num_trials):
        """Run multiple number of trials
        """
        for i in range(num_trials):
            self.schedule_next()
        self.join_tasks()

    def schedule_next(self):
        """Schedule next searcher suggested task
        """
        task = Task(self.train_fn, {'args': self.args, 'config': self.searcher.get_config()},
                    Resources(**self.resource))
        self.add_task(task)

    def add_task(self, task):
        # adding the task
        logger.debug("Adding A New Task {}".format(task))
        FIFO_Scheduler.RESOURCE_MANAGER._request(task.resources)
        with self.LOCK:
            if task.resources.num_gpus > 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(task.resources.gpu_ids)[1:-1]
            reporter = StatusReporter()
            task.args['reporter'] = reporter
            tp = mp.Process(target=FIFO_Scheduler._run_task, args=(
                           task.fn, task.args, task.resources,
                           FIFO_Scheduler.RESOURCE_MANAGER))
            rp = threading.Thread(target=FIFO_Scheduler._run_reporter, args=(tp, reporter, self.searcher))
            tp.start()
            rp.start()
            self.SCHEDULED_TASKS.append({'Task': task, 'Process': tp, 'ReporterProcess': rp})

    @staticmethod
    def _run_reporter(task_process, reporter, searcher):
        last_result = None
        while task_process.is_alive():
            kwargs = reporter.fetch()
            if 'done' in kwargs and kwargs['done'] is True: break
            reporter.move_on()
            last_result = kwargs
        searcher.update(**last_result)
