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
    def __init__(self, train_fn, args, resource, searcher):
        self.train_fn = train_fn
        self.args = args
        self.resource = resource
        self.searcher = searcher

    def run(self, num_trials):
        for i in range(num_trials):
            self.schedule_next()

    def schedule_next(self):
        task = Task(self.train_fn, {'args': self.args, 'config': self.searcher.get_config()},
                    Resources(**self.resource))
        self.add_task(task)

    def add_task(self, task):
        # adding the task
        logger.debug("Adding A New Task {}".format(task))
        TaskScheduler.RESOURCE_MANAGER._request(task.resources)
        with self.LOCK:
            if task.resources.num_gpus > 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(task.resources.gpu_ids)[1:-1]
            reporter = StatusReporter()
            task.args['reporter'] = reporter
            tp = mp.Process(target=TaskScheduler._run_task, args=(
                           task.fn, task.args, task.resources,
                           FIFO_Scheduler.RESOURCE_MANAGER))
            rp = threading.Thread(target=FIFO_Scheduler._run_reporter, args=(tp, reporter, self.searcher))
            tp.start()
            rp.start()
            self.SCHEDULED_TASKS.append({'Task': task, 'Process': tp, 'ReporterProcess': rp})

    @staticmethod
    def _run_reporter(process, reporter, searcher):
        last_result = None
        while process.is_alive():
            kwargs = reporter.fetch()
            if 'done' in kwargs and kwargs['done'] is True: break
            last_result = kwargs
        searcher.update(**last_result)

    @staticmethod
    def _run_task(fn, args, resources, resource_manager):
        """Executing the task
        """
        fn(**args)
        resource_manager._release(resources)
